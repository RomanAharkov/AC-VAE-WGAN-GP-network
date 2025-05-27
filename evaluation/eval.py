import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassCohenKappa


def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    # Overall Accuracy
    oa = np.trace(cm) / np.sum(cm)
    # Average Accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        aa = np.nanmean(per_class_acc)
    # Kappa
    preds = torch.tensor(y_pred)
    target = torch.tensor(y_true)
    metric = MulticlassCohenKappa(num_classes=num_classes)
    kappa = metric(preds, target).item()
    return oa, aa, kappa


def evaluate(train_dataset: Dataset, test_dataset: Dataset, batch_size: int, model: Module, learning_rate: float,
             num_epochs: int, num_classes: int, device: str) -> tuple[float, float, float]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # (B, C) / (B, 1)
            labels = labels.squeeze(1)  # (B,)
            inputs = inputs.unsqueeze(1)  # (B, 1, C)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)
            inputs = inputs.unsqueeze(1)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B,)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    oa, aa, kappa = compute_metrics(y_true, y_pred, num_classes)
    return oa, aa, kappa
