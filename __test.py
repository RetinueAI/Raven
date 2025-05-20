from torch.utils.data import DataLoader

from __data import RavenDataSet



"""
Dataset Validation Script

This script validates the Raven Dataset by checking the alignment between input tokens and their corresponding labels. It ensures:
- The number of tokens matches the number of labels for each sample.
- Special tokens ([CLS], [SEP], [PAD]) are labeled with -100.
- Non-special tokens have valid labels (not -100).

The script uses the `RavenDataSet` class to load the dataset from '__raven.csv' and creates a `DataLoader` to process the data in batches. It then performs detailed checks on each batch to detect any mismatches.

Main Function:
    test_label_alignment_data_loader_with_detailed_checks(data_loader): Performs validation checks on the dataset.

Run this script to verify the integrity of the dataset before training or inference.
"""
weights_dir = "weights"
filename = "__raven.csv"
maxlen = 512
n_classes = 5
batch_size = 32

dataset = RavenDataSet(filename=filename, maxlen=maxlen, n_classes=n_classes, split_type='test')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def test_label_alignment_data_loader_with_detailed_checks(data_loader):
    mismatch_summary = []
    
    for batch_idx, batch in enumerate(data_loader):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        for i in range(input_ids.size(0)):
            tokens = dataset.tokenizer.convert_ids_to_tokens(input_ids[i])
            label_seq = labels[i].tolist()
            
            if len(tokens) != len(label_seq):
                mismatch_summary.append({
                    'batch_idx': batch_idx,
                    'item_idx': i,
                    'reason': "Length mismatch",
                    'details': f"Token length: {len(tokens)}, Label length: {len(label_seq)}"
                })
                continue
            
            for token_idx, (token, label) in enumerate(zip(tokens, label_seq)):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    if label != -100:
                        mismatch_summary.append({
                            'batch_idx': batch_idx,
                            'item_idx': i,
                            'reason': "Special token label mismatch",
                            'details': f"Token: {token}, Expected label: -100, Found label: {label}"
                        })
                else:
                    if label == -100:
                        mismatch_summary.append({
                            'batch_idx': batch_idx,
                            'item_idx': i,
                            'reason': "Non-special token with ignore label",
                            'details': f"Token: {token}, Label: {label}"
                        })
    
    if not mismatch_summary:
        print("No mismatches found. Label alignment successful for all entries.")
    else:
        print("Mismatches found:")
        for mismatch in mismatch_summary:
            print(f"Batch {mismatch['batch_idx']}, Item {mismatch['item_idx']}: {mismatch['reason']}. {mismatch['details']}")

test_label_alignment_data_loader_with_detailed_checks(data_loader)

