from transformers import BertModel
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F


class Raven(nn.Module):
    def __init__(self, bert_model='bert-base-multilingual-cased', n_classes=5, dropout_rate=0.5):
        super(Raven, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=2, padding='same')
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        
        transformer_encoder_layer = TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=4)

        self.segment_classifier = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
    
        conv_output = F.relu(self.conv1d(outputs.permute(0, 2, 1))).permute(0, 2, 1)
        lstm_output, _ = self.lstm(conv_output)
        transformer_output = self.transformer_encoder(lstm_output)

        # Apply the classifier to each token in the sequence
        token_labels_logits = self.segment_classifier(transformer_output)  # Renamed for clarity

        return token_labels_logits
