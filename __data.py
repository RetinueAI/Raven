from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
from transformers import BertTokenizer


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import ast


class RavenDataSet(Dataset):
    def __init__(self, filename, maxlen, n_classes, split_type='train', split_ratio=0.9):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.n_classes = n_classes
        df = pd.read_csv(filename)
        
        # No need for splitting the text here
        train_df, test_df = train_test_split(df, test_size=1-split_ratio, random_state=42)
        self.df = train_df if split_type == 'train' else test_df
        self.df = self.df.reset_index(drop=True)
        self.maxlen = maxlen
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['inputs']
        labels = ast.literal_eval(row['labels'])  # Assuming labels are stored as string representations of lists
        
        # Tokenize the text
        inputs = self.tokenizer(text, add_special_tokens=True, max_length=self.maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        # Align labels with tokenized input IDs, assuming labels are at the character level and need conversion
        # This part needs to be adjusted based on your actual label format and alignment strategy
        label_ids = self._align_labels_with_tokens(labels, input_ids)
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': torch.tensor(label_ids, dtype=torch.long)  # Assuming classification task with integer labels
        }
    
    def _align_labels_with_tokens(self, labels, input_ids):
        # Decode the input IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Initialize the list for the aligned labels with default value -100
        # -100 is often used to ignore tokens during loss calculation in PyTorch
        aligned_labels = [-100] * len(tokens)
        
        # Initialize variables to keep track of the position in the original labels list
        label_index = 0
        
        for i, token in enumerate(tokens):
            # Skip special tokens added by BERT tokenizer ([CLS], [SEP], [PAD])
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            # If the current token is a subword that does not start with "##",
            # it is the start of a new word, and we should move to the next label
            if not token.startswith("##"):
                if label_index < len(labels):
                    aligned_labels[i] = labels[label_index]
                    label_index += 1
            else:
                # For subtokens (that start with "##"), use the same label as the previous token
                if label_index < len(labels):
                    aligned_labels[i] = labels[label_index - 1]
        
        return aligned_labels
