import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from __data import RavenDataSet
from __raven import Raven 


weights_dir = "weights"
filename = "__raven.csv"
maxlen = 512
n_classes = 14
batch_size = 4
learning_rate = 3e-7
epochs = 100

current_time = time.localtime()
name = f"raven_{time.strftime('%H_%M_%d_%m_%Y', current_time)}"

wandb.init(project="Raven", name=name, config={
    "learning_rate": learning_rate,
    "architecture": "bert-base-multilingual-cased",
    "dataset": "Raven Dataset",
    "epochs": epochs,
})

train_dataset = RavenDataSet(filename=filename, maxlen=maxlen, n_classes=n_classes, split_type='train', split_ratio=0.9)
test_dataset = RavenDataSet(filename=filename, maxlen=maxlen, n_classes=n_classes, split_type='test', split_ratio=0.9)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Raven(bert_model='bert-base-multilingual-cased', n_classes=n_classes, dropout_rate=0.2).to(device)

# Initialize the loss function for sequence labeling
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Assuming -100 is used to ignore certain tokens

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
scaler = GradScaler()

best_val_loss = float('inf')

stopper = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.view(-1, n_classes), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        wandb.log({"batch_train_loss": loss.item()})

        total_loss += loss.item()

    # Validation phase
    model.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.view(-1, n_classes), labels.view(-1))
            val_total_loss += loss.item()

    current_val_loss = val_total_loss / len(test_loader)
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        model_save_path = os.path.join(weights_dir, f"{name}.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print("Saved best model with validation loss:", best_val_loss)
        stopper = 0
    else:
        stopper += 1

    scheduler.step(current_val_loss)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss / len(train_loader),
        "val_loss": current_val_loss,
        "learning_rate": optimizer.param_groups[0]['lr'], 
    })

    if stopper > 9:
        break

wandb.finish()