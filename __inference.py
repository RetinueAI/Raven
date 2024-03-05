import torch
from transformers import AutoTokenizer

from __raven import Raven


# Assuming the model class and its parameters are defined as in your training script
model = Raven(bert_model='bert-base-multilingual-cased', n_classes=5, dropout_rate=0.2)

# Move model to the appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model.to(device)

# Load the trained model weights
model_path = 'weights/raven_13_03_24_02_2024.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
model.eval()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)

# Sample input text
input_text = "what did you say last week about bridges. Adjust my preference for apples. I just started training Jiu Jitsu."

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move tensors to the same device as the model
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

with torch.no_grad():
    logits = model(input_ids, attention_mask)

# Apply softmax to get probabilities (dim=-1 ensures softmax is applied across the classes)
probabilities = torch.softmax(logits, dim=-1)

# Take argmax to get the most likely class index for each token
predicted_labels = torch.argmax(probabilities, dim=-1)

# Convert predicted labels to list for easier reading
predicted_labels_list = predicted_labels.tolist()[0]  # Assuming batch size of 1 for simplicity

# Convert token IDs back to tokens for better understanding of output
tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])

# Combine tokens with their predicted labels
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