import torch
from transformers import AutoTokenizer

from __raven import Raven



"""
Model Inference Script

This script demonstrates how to use the trained Raven model for inference. It loads the trained model weights, sets the model to evaluation mode, and processes a sample input text. The script tokenizes the input using the BERT tokenizer, passes it through the model to get logits, applies softmax to obtain probabilities, and predicts labels for each token. It then prints the tokens with their corresponding predicted labels.

Usage:
- Load trained weights from the "weights" directory.
- Tokenize input text using `AutoTokenizer` from Transformers.
- Pass tokenized inputs through the model to get logits.
- Apply softmax and argmax to determine predicted labels.
- Map tokens to predicted labels for analysis.

Run this script to see how to use the trained model for making predictions on new text.
"""

model = Raven(bert_model='bert-base-multilingual-cased', n_classes=5, dropout_rate=0.2)

device = 'cpu'
model.to(device)

model_path = 'weights/raven_13_03_24_02_2024.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)

input_text = "what did you say last week about bridges. Adjust my preference for apples. I just started training Jiu Jitsu."

inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

with torch.no_grad():
    logits = model(input_ids, attention_mask)

probabilities = torch.softmax(logits, dim=-1)

predicted_labels = torch.argmax(probabilities, dim=-1)

predicted_labels_list = predicted_labels.tolist()[0]

tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

token_label_pairs = list(zip(tokens, predicted_labels_list))

print("probabilities:")
print(probabilities)
print('')
print('predicted labels')
print(predicted_labels)
print('')


print("Token to Predicted Label Index Mapping:")
for token, label in token_label_pairs:
    print(f"{token}: {label}")
