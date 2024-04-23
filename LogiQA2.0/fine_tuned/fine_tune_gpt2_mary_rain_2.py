import sys
import os
import json
from typing import List, Dict
from transformer_lens import HookedTransformer
from transformer_lens.evals import evaluate
from transformer_lens.train import HookedTransformerTrainConfig, train
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
import torch.nn as nn
import tqdm
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer
print("is cuda available?", torch.cuda.is_available())

sys.path.append("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")
os.chdir("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")

# Load the LogiQA2.0 train dataset
with open("train_mary_rain.txt", "r") as file:
    train_data = [json.loads(line) for line in file]

# Remove indices that are too long
train_data = [item for i, item in enumerate(train_data)]


# Load the LogiQA2.0 train dataset
with open("test_mary_rain.txt", "r") as file:
    validation_data = [json.loads(line) for line in file]

# Remove indices that are too long
validation_data = [item for i, item in enumerate(validation_data)]


# Load the pre-trained GPT-2 model
model = HookedTransformer.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Prepare the dataset for training
def prepare_data(data):
    tokenized_data = []
    for item in data:
        prompt = item["prompt"]
        completion = item["completion"]
        label = item["label"]
        
        # Tokenize prompt and completion
        tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        tokenized_completion = tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")
        
        tokenized_data.append({"prompt": tokenized_prompt.squeeze(0),
                               "completion": tokenized_completion.squeeze(0),
                               "label": label})
    return tokenized_data


def pad_collate(batch):
    # Assume that each element in "batch" is a tuple (data, label).
    # You might need to adjust this depending on how your data is structured.
    data = [item[0] for item in batch]  # Extracting data
    labels = [item[1] for item in batch]  # Extracting labels

    # Padding the sequences to the maximum length in the batch
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    
    # Convert labels to a tensor, if they are not already
    labels = torch.tensor(labels, dtype=torch.long)

    return data_padded, labels

train_dataset = prepare_data(train_data)

# Create a custom dataset class
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"tokens": item["prompt"], "completion": item["completion"], "label": item["label"]}

# Create a DataLoader
train_dataset = TextDataset(train_dataset)


#the custom collate function is for padding the 'prompt' sequences
def custom_collate_fn(batch):
    # Extract 'prompt', 'completion', and 'label' from the batch
    prompts = [item['prompt'] for item in batch]
    completions = [item['completion'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Pad the 'prompt' sequences so they all have the same length
    prompts_padded = pad_sequence(prompts, batch_first=True)

    # Since 'completion' and 'labels' are already uniform in length or don't require padding,
    # we can just use default_collate for them. This step might need adjustments based on your specific needs.
    completions_collated = completions
    labels_collated = labels

    # Return the collated batch as a dict
    return {'tokens': prompts_padded, 'completion': completions_collated, 'label': labels_collated}

def custom_collate_eval_fn(batch):
    # Extract 'prompt', 'completion', and 'label' from the batch
    prompts = [item['tokens'].to(device) for item in batch]
    completions = [item['completion'].to(device) for item in batch]
    labels = [item['label'] for item in batch]
    
    # Pad the 'prompt' sequences so they all have the same length
    prompts_padded = pad_sequence(prompts, batch_first=True)

    # Since 'completion' and 'labels' are already uniform in length or don't require padding,
    # we can just use default_collate for them. This step might need adjustments based on your specific needs.
    completions_collated = completions
    labels_collated = labels

    # Return the collated batch as a dict
    return {'tokens': prompts_padded, 'completion': completions_collated, 'label': labels_collated}


#use the custom collate function in the DataLoader
dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=custom_collate_fn, shuffle=True)

# new_dataset = []

# for batch in dataloader:
#     print("bathc length", len(batch['tokens']))
#     for i in range(len(batch['tokens'])):
#         item = {
#             'tokens': batch['tokens'][i],
#             'completion': batch['completion'][i],
#             'label': batch['label'][i]
#         }
#         new_dataset.append(item)

# print("new_data_set length:", len(new_dataset[0]['tokens']))
# print("new_data_set length:", len(new_dataset[1]['tokens']))
# print("new_data_set length:", len(new_dataset[2]['tokens']))
# print("new_data_set length:", len(new_dataset[3]['tokens']))
# print("new_data_set length:", len(new_dataset[11]['tokens']))
# print("new_data_set length:", len(new_dataset[20]['tokens']))


class PaddedTextDataset(Dataset):
    def __init__(self, data):
        # 'data' should be a list of dictionaries
        self.data = data
        #max length for both tokens and completion
        self.max_length_tokens = max(len(item['tokens']) for item in data)
        self.max_length_completion = max(len(item['completion']) for item in data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'tokens' in item: #pad the prompt sequence
            tokens = (item['tokens']).tolist()
            padded_tokens = tokens + [0] * (self.max_length_tokens - len(tokens))
            item['tokens'] = torch.tensor(padded_tokens, dtype=torch.long)
        if 'completion' in item: #pad the completion sequence
            completion = (item['completion']).tolist()
            padded_completions = completion + [0] * (self.max_length_completion - len(completion))
            item['completion'] = torch.tensor(padded_completions, dtype=torch.long)    
            
        return item

#define hyperparameters through HTL's configs
config = HookedTransformerTrainConfig(
    num_epochs=2,  # Number of epochs to train for
    batch_size=10,  # Adjust the batch size according to your computational resources
    lr=0.00019,  # Learning rate
    seed=42,  # A seed for reproducibility
    momentum=0.0,  # Momentum (not typically used with AdamW, but here for completeness)
    max_grad_norm=None,  # Maximum gradient norm (optional, adjust as needed)
    weight_decay=0.01,  # Weight decay for regularization
    optimizer_name="AdamW",  # Optimizer
    device="cpu",  # Automatically use GPU if available
    warmup_steps=0, # Number of warmup steps for the learning rate scheduler
    save_every=None,  # How often to save the model (optional, adjust as needed)
    save_dir="/Users/crayhippo/deduct/LogiQA2.0/fine_tuned",  # Directory to save models (optional, adjust as needed)
    wandb=False,  # Toggle Weights & Biases logging
    wandb_project_name=None,  # Weights & Biases project name (optional)
    print_every=50,  # How often to print training progress
    max_steps=None  # Optional, set a limit for steps per epoch for debugging or faster iterations
)

# #create instance of PaddedTextDataset
padded_dataset = PaddedTextDataset(train_dataset)

print("padded dataset example", padded_dataset[0])

for i, sample in enumerate(padded_dataset):
    print(f"Completion padded {i}: {sample['completion'].size()}")


#train the model
fine_tuned_model = train(model, config, padded_dataset).to(device)
print("training complete")

#create validation Dataset and Dataloader, with padding
validation_dataset = PaddedTextDataset(TextDataset(prepare_data(validation_data)))
print("validation dataset", validation_dataset)
print("validation dataset", len(validation_dataset))
print("validation dataset", validation_dataset[0])
# validation_dataset = TextDataset(validation_dataset)
# validation_dataset = PaddedTextDataset(validation_dataset)
validation_dataloader = DataLoader(validation_dataset, batch_size=10, collate_fn=custom_collate_eval_fn, shuffle=True)

@torch.inference_mode()
def evaluate_on_dataset_internal(model, data_loader, truncate=100, device="cpu"):
    running_loss = 0
    total = 0
    for batch in tqdm.tqdm(data_loader):
        loss = model(batch["tokens"].to(device), return_type="loss").mean()
        running_loss += loss.item()
        total += 1
        if total > truncate:
            break
    return running_loss / total

eval_performance = evaluate_on_dataset_internal(fine_tuned_model, validation_dataloader)
print(eval_performance)
print("validation evaluation complete")
print(fine_tuned_model)


# Save the fine-tuned model
#torch.save(fine_tuned_model.state_dict(), "/Users/crayhippo/deduct/LogiQA2.0/fine_tuned/fine_tuned_model_test.pth")