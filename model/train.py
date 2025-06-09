import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.config import TRAIN_VERSION
from model.loader import CCDSequenceDataset
from model.model import YOLOBackboneConvLSTM
from model.utils import ApplyToSequence, SequenceTransform, FocalLoss, EarlyStopping, compute_metrics, log_metrics, \
    save_checkpoint


def train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=20,
        lr=1e-4,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        log_dir=f"workspace/logs_v{TRAIN_VERSION}/run_v{TRAIN_VERSION}",
        ckpt_dir=f"workspace/checkpoints_v{TRAIN_VERSION}"
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True, save_path=ckpt_dir + 'best_model.pth')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_losses, all_preds, all_targets = [], [], []
        current_lr = optimizer.param_groups[0]['lr']

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            avg_focal_weight = criterion.last_focal_weight_mean
            writer.add_scalar("Train/Avg_Focal_Weight", avg_focal_weight, epoch)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds).ravel()
        all_targets = np.concatenate(all_targets).ravel()
        bin_preds = (all_preds >= 0.5).astype(int)

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        train_metrics = compute_metrics(all_targets, bin_preds, all_preds) if all_targets.size > 0 else {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0
        }
        log_metrics(writer, train_metrics, epoch, prefix="Train")

        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, epoch, writer)

        # Сохраняем лучшую модель по валидационной потере
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ckpt_dir, "best_model")

        save_checkpoint(model, ckpt_dir, f"epoch_{epoch + 1}")

        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"))

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered on {epoch + 1}.")
            break

    save_checkpoint(model, ckpt_dir, "last_model")
    writer.close()


def evaluate(model, loader, device, criterion, epoch, writer, log_mistakes=True, max_mistakes=10):
    model.eval()
    losses, all_preds, all_targets = [], [], []
    mistake_images = []
    mistake_preds = []
    mistake_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.append(loss.item())
            probs = torch.sigmoid(outputs).cpu().numpy()
            bin_preds = (probs >= 0.5).astype(int)
            targets_np = targets.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets_np)

            # Логгирование ошибок (FP/FN)
            if log_mistakes and len(mistake_images) < max_mistakes:
                # Поэлементное сравнение предсказаний и целей
                mismatch_mask = bin_preds != targets_np
                mismatch_indices = np.where(mismatch_mask)[0]

                for i in mismatch_indices[:max_mistakes - len(mistake_images)]:
                    first_frame = inputs[i, 0].cpu()
                    mistake_images.append(first_frame)
                    mistake_preds.append(bin_preds[i])
                    mistake_targets.append(targets_np[i])

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()
    bin_preds = (all_preds >= 0.5).astype(int)

    metrics = compute_metrics(all_targets, bin_preds, all_preds)
    log_metrics(writer, metrics, epoch, prefix="Val")

    if writer is not None and log_mistakes:
        for i, (img, pred, true) in enumerate(zip(mistake_images, mistake_preds, mistake_targets)):
            writer.add_image(f"Val/Mistake_{i}_P{pred}_T{true}", img, global_step=epoch)

    return np.mean(losses), metrics


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = ApplyToSequence(transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
]))


train_transform = SequenceTransform(img_size=256)

train_dataset = CCDSequenceDataset('./split_CCD/train', sequence_length=50, transform=train_transform)
val_dataset = CCDSequenceDataset('./split_CCD/val', sequence_length=50, transform=val_transform)
test_dataset = CCDSequenceDataset('./split_CCD/test', sequence_length=50, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

model = YOLOBackboneConvLSTM(
    yolo_ckpt='workspace/yolo11m.pt',
    hidden_dim=256,
    num_layers=1,
    sequence_length=50,
    img_size=256
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, val_loader, device=device, num_epochs=30)
