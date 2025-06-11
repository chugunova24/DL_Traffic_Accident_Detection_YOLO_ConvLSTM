import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc
)
from torch.utils.data import DataLoader
from torchvision import transforms

from model.loader import CCDSequenceDataset
from model.model import YOLOBackboneConvLSTM
from model.utils import ApplyToSequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = ApplyToSequence(transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
]))

test_dataset = CCDSequenceDataset('/split_CCD/test', sequence_length=50, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Test dataset size: {len(test_dataset)} sequences")


model = YOLOBackboneConvLSTM(
    yolo_ckpt='yolo11m.pt',
    hidden_dim=256,
    num_layers=1,
    bidirectional=False,
    sequence_length=50,
    img_size=256
)

model.load_state_dict(torch.load('best_model.pth'))

model.eval()
model = model.to(device)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(int)

    metrics = {
        'precision': precision_score(all_targets, binary_preds),
        'recall': recall_score(all_targets, binary_preds),
        'f1': f1_score(all_targets, binary_preds)
    }

    return metrics, all_preds, all_targets


metrics, all_preds, all_targets = evaluate_model(model, test_loader, device)
print("\nМетрики на тестовом наборе:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")


precision, recall, _ = precision_recall_curve(all_targets, all_preds)
pr_auc = auc(recall, precision)

plt.figure(figsize=(4, 4))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая (Precision-Recall)')
plt.legend(loc="lower left")
plt.show()