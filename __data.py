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



"""
    Raven Dataset Module
    
    This module defines the `RavenDataSet` class, a custom PyTorch Dataset for loading and processing text data for token classification tasks using BERT.
    
    The `RavenDataSet` class handles:
    - Loading data from a CSV file.
    - Tokenizing text using the BERT tokenizer (`bert-base-multilingual-cased`).
    - Aligning labels with tokenized inputs, ensuring special tokens ([CLS], [SEP], [PAD]) are labeled with -100.
    - Splitting the data into training and test sets based on a specified ratio (default: 90% training, 10% testing).
    
    Classes:
        RavenDataSet: A custom dataset class for token classification tasks.
    
    Usage:
        - Initialize with parameters such as `filename`, `maxlen`, `n_classes`, and `split_type`.
        - Use `__len__` to get the dataset size.
        - Use `__getitem__` to retrieve processed samples (input_ids, attention_mask, labels).
        - The `_align_labels_with_tokens` method ensures proper label alignment during tokenization.
    """
class RavenDataSet(Dataset):
    def __init__(self, filename, maxlen, n_classes, split_type='train', split_ratio=0.9):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.n_classes = n_classes
        df = pd.read_csv(filename)
        
        train_df, test_df = train_test_split(df, test_size=1-split_ratio, random_state=42)
        self.df = train_df if split_type == 'train' else test_df
        self.df = self.df.reset_index(drop=True)
        self.maxlen = maxlen
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['inputs']
        labels = ast.literal_eval(row['labels'])
        
        inputs = self.tokenizer(text, add_special_tokens=True, max_length=self.maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        label_ids = self._align_labels_with_tokens(labels, input_ids)
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': torch.tensor(label_ids, dtype=torch.long)  # Assuming classification task with integer labels
        }
    
    def _align_labels_with_tokens(self, labels, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        aligned_labels = [-100] * len(tokens)
        
        label_index = 0
        
        for i, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if not token.startswith("##"):
                if label_index < len(labels):
                    aligned_labels[i] = labels[label_index]
                    label_index += 1
            else:
                if label_index < len(labels):
                    aligned_labels[i] = labels[label_index - 1]
        
        return aligned_labels
