from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import json
import sys
import os
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random

sys.path.append("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")
os.chdir("/Users/crayhippo/deduct/LogiQA2.0/logiqa2nli/data/QA2NLI")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])

# Load the LogiQA2.0 test dataset
with open("test.txt", "r") as file:
    test_data = [json.loads(line) for line in file]

#remove indices that are too long
indices_to_remove = [1058, 778, 399, 20, 117, 211, 288, 434, 581, 88, 89]
test_data = [item for i, item in enumerate(test_data) if i not in indices_to_remove]

# evaluate the model on the first 300 examples
n = 300  # Number of examples in the subset
test_data_subset = random.sample(test_data, n)
#test_data_subset = test_data[:300]
# Evaluate the model on the test dataset
y_true = []
y_pred = []

removed_indices = []

for i, line_dict in enumerate(test_data_subset):
    label = 0 if line_dict['label'] == "not entailed" else 1
    maj_premise = ' '.join(line_dict['major_premise'])
    min_premise = ' '.join(line_dict['minor_premise'])
    hypo = line_dict['conclusion']
    
    prompt_input = "Given the fact: " + maj_premise + ' ' + min_premise + " Does it follow that: " + hypo + " Yes or no?"
    y_true.append(label)
    
    input_ids = tokenizer.encode(prompt_input, return_tensors="pt")
    input_length = len(input_ids[0])
    print("input length:", input_length)
    
    # filter by input length
    if(input_length > 1000):
        print("input length exceeds 500 in line, we're skipping", i)
        removed_indices.append(i)
        continue
    
    output = model.generate(input_ids, max_length=input_length+20, num_return_sequences=1)
    pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    
    y_pred.append(1 if "yes" in pred else 0)

y_true = [item for i, item in enumerate(y_true) if i not in removed_indices]

f_score = f1_score(y_true, y_pred)
p_score = precision_score(y_true, y_pred)
r_score = recall_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("F1 Score:", f_score)
print("Precision Score:", p_score)
print("Recall Score:", r_score)
print("Accuracy:", acc)