#use CPU as a fallback for unsupported operations like isin

import sys
import os
import json
from typing import List, Dict
from transformer_lens import HookedTransformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer

sys.path.append("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")
os.chdir("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")

# Load the LogiQA2.0 train dataset
with open("train_mary_rain.txt", "r") as file:
    train_data = [json.loads(line) for line in file]

# Remove indices that are too long
train_data = [item for i, item in enumerate(train_data)]

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

train_dataset = prepare_data(train_data)

# Create a custom dataset class
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"prompt": item["prompt"], "completion": item["completion"], "label": item["label"]}

# Create a DataLoader
train_dataset = TextDataset(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: {
    "prompt": pad_sequence([item["prompt"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
    "completion": pad_sequence([item["completion"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
    "label": [item["label"] for item in batch]
})


#define hhyperparameters
optimizer = AdamW(model.parameters(), lr=0.0003, eps=1e-7, weight_decay=0.01)
criterion = CrossEntropyLoss()
num_epochs = 3
early_stopping_patience = 10
best_eval_loss = float("inf")
patience_counter = 0

#set requires_grad to true
for param in model.parameters():
    param.requires_grad = True

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        prompts = batch['prompt'].to(device)
        completions = batch['completion'].to(device)
        completions = completions.float()
        print("completions type:", type(completions))

        labels = batch['label']
    
        optimizer.zero_grad()

        # Generate completions
        model.train()
        
        #define prompt length
        prompt_length = prompts.size(1)
        #print("prompt_length", prompt_length)

        #outputs = model(prompts).logit //when using Transformers package
        print(model)
        outputs = model.generate(prompts, max_new_tokens=prompt_length + 2, stop_at_eos=True, return_type="logits")
        print("output requires grad?", outputs.requires_grad)
        outputs = outputs.float()

        print("completions:", completions.size())
        print("input tensor size: ", outputs[:,prompt_length:prompt_length+2].size())
        print("input tensor dtype: ", outputs[:,prompt_length:prompt_length+2].dtype)
        # Calculate loss
        loss = criterion(outputs[:,prompt_length:prompt_length+2], completions)
    
        print(any(p.requires_grad for p in model.parameters()))  # Should be True
        print(outputs.requires_grad)  # Also should be True

        loss.backward()
        optimizer.step()


        # Evaluate the model
        model.eval()
        with torch.no_grad():
            generated_completions = model.generate(prompts, max_length=prompts.size(1) + 2)
            accuracy = evaluate_answer(generated_completions, completions)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        model.train()
        loss.backward()
        optimizer.step()
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            generated_completions = model.generate(prompts, max_length=prompts.size(1) + 2)
            accuracy = evaluate_answer(generated_completions, completions)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        model.train()

# Evaluate answer function
def evaluate_answer(generated_completions, actual_completions):
    correct_count = 0
    total_count = generated_completions.size(0)
    
    for generated, actual in zip(generated_completions, actual_completions):
        generated_text = tokenizer.decode(generated[generated.size(0)-2:], skip_special_tokens=True)
        actual_text = tokenizer.decode(actual, skip_special_tokens=True)
        
        if actual_text.strip() in generated_text.strip():
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy

# Save the fine-tuned model
model.save_pretrained("finetuned_gpt2_logiqa")

