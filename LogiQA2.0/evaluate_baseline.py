from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
import os
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")
os.chdir("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")

# Load the pre-trained GPT-2 small model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Load the LogiQA2.0 test dataset
with open("test.txt", "r") as file:
    test_data = [json.loads(line) for line in file]

# Evaluate the model on the test dataset
y_true = []
y_pred = []

for i, line_dict in enumerate(test_data):
    label = 0 if line_dict['label'] == "not entailed" else 1
    maj_premise = ' '.join(line_dict['major_premise'])
    min_premise = ' '.join(line_dict['minor_premise'])
    hypo = line_dict['conclusion']
    
    prompt_input = "Given the fact: " + maj_premise + ' ' + min_premise + " Does it follow that: " + hypo + " Yes or no?"
    y_true.append(label)
    
    # Tokenize the input and generate the attention mask
    inputs = tokenizer.encode_plus(prompt_input, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Maximum token ID:", torch.max(input_ids).item())
    print("Minimum token ID:", torch.min(input_ids).item())
    
    # Generate the model's prediction
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1024 + 20,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    
    y_pred.append(1 if "yes" in pred else 0)

f_score = f1_score(y_true, y_pred)
p_score = precision_score(y_true, y_pred)
r_score = recall_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("F1 Score:", f_score)
print("Precision Score:", p_score)
print("Recall Score:", r_score)
print("Accuracy:", acc)