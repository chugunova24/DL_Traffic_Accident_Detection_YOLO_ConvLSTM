import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageDraw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    confusion_matrix
)
from sklearn.metrics import (
    roc_auc_score
)
from torch import nn
from torchvision.transforms import functional as F


class SequenceTransform:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def random_params(self):
        params = {
            'hflip': random.random() < 0.5,
            'apply_color_jitter': random.random() < 0.7,
            'blur': random.random() < 0.3,
            'noise': random.random() < 0.3,
            'darken': random.random() < 0.2,
            'occlusion': random.random() < 0.3,
        }

        if params['apply_color_jitter']:
            params.update({
                'brightness': random.uniform(0.8, 1.2),
                'contrast': random.uniform(0.8, 1.2),
                'saturation': random.uniform(0.8, 1.2),
                'hue': random.uniform(-0.05, 0.05),
            })

        if params['blur']:
            params['blur_radius'] = random.uniform(0.5, 1.5)
        if params['noise']:
            params['noise_std'] = random.uniform(0.0, 0.05)
        if params['darken']:
            params['darken_factor'] = random.uniform(0.5, 0.9)
        if params['occlusion']:
            params['occ_size'] = random.uniform(0.1, 0.3)

        # Crop
        scale = random.uniform(0.8, 1.0)
        ratio = random.uniform(0.9, 1.1)
        params['crop'] = {'scale': scale, 'ratio': ratio}

        return params

    def apply_occlusion(self, img, occ_size_frac):
        w, h = img.size
        occ_w, occ_h = int(w * occ_size_frac), int(h * occ_size_frac)
        x0 = random.randint(0, w - occ_w)
        y0 = random.randint(0, h - occ_h)
        x1, y1 = x0 + occ_w, y0 + occ_h

        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
        return img

    def __call__(self, frames):
        params = self.random_params()
        transformed = []

        if len(frames) > 0:
            i, j, h, w = TF.RandomResizedCrop.get_params(
                frames[0],
                scale=(params['crop']['scale'], 1.0),
                ratio=(params['crop']['ratio'], params['crop']['ratio'])
            )

        for img in frames:
            if params['hflip']:
                img = TF.hflip(img)

            img = TF.resized_crop(img, i, j, h, w, size=[self.img_size, self.img_size])

            if params['apply_color_jitter']:
                img = TF.adjust_brightness(img, params['brightness'])
                img = TF.adjust_contrast(img, params['contrast'])
                img = TF.adjust_saturation(img, params['saturation'])
                img = TF.adjust_hue(img, params['hue'])

            if params['darken']:
                img = TF.adjust_brightness(img, params['darken_factor'])
                img = TF.adjust_contrast(img, params['darken_factor'])

            if params['blur']:
                img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_radius']))

            if params['occlusion']:
                img = self.apply_occlusion(img, params['occ_size'])

            img = TF.to_tensor(img)

            if params['noise']:
                noise = torch.randn_like(img) * params['noise_std']
                img = torch.clamp(img + noise, 0., 1.)

            transformed.append(img)

        return torch.stack(transformed, dim=0)  # (T, C, H, W)



class ApplyToSequence:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, frames):
        return torch.stack([self.transform(f) for f in frames], dim=0)  # (T, C, H, W)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.last_focal_weight_mean = None

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        self.last_focal_weight_mean = focal_weight.mean().item()

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, save_path=None):
        """
        patience: сколько эпох ждать без улучшения
        min_delta: минимальное улучшение, чтобы считать, что модель стала лучше
        save_path: путь, куда сохранять лучшую модель
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path

        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"Validation loss decreased. Saving model to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def save_metrics_to_csv(file_path, metrics_data, header=False):
    fieldnames = [
        'epoch', 'train_loss', 'val_loss',
        'train_acc', 'val_acc',
        'train_precision', 'val_precision',
        'train_recall', 'val_recall',
        'train_f1', 'val_f1',
        'train_roc_auc', 'val_roc_auc',
        'lr'
    ]
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or header:
            writer.writeheader()

        writer.writerow(metrics_data)


def save_checkpoint(model, ckpt_dir, name):
    path = os.path.join(ckpt_dir, f"{name}.pth")
    torch.save(model.state_dict(), path)
    print(f"Сохранена модель: {path}")


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"{title} Confusion Matrix")
    return fig


def log_misclassified_images(writer, inputs, preds, targets, step):
    preds_bin = (preds >= 0.5).astype(int)
    for i in range(len(preds)):
        pred, target = preds_bin[i], targets[i]
        if pred != target:
            img_grid = torchvision.utils.make_grid(inputs[i])
            label = f"Pred: {pred}, True: {target}"
            writer.add_image(label, img_grid[0], global_step=step)


def log_metrics(writer, metrics, epoch, prefix=""):
    for key, value in metrics.items():
        if key == "confusion_matrix":
            fig = plot_confusion_matrix(value, prefix)
            writer.add_figure(f"{prefix}/Confusion_Matrix", fig, epoch)
        else:
            writer.add_scalar(f"{prefix}/{key}", value, epoch)


def compute_metrics(targets, preds, prob_preds):
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    roc_auc = roc_auc_score(targets, prob_preds)
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TP": tp
    }


def plot_error_analysis(error_samples, epoch, save_dir, top_n=20):
    # Сортируем ошибки по уверенности модели
    fp_errors = [e for e in error_samples if e[3] == 0]  # False Positive
    fn_errors = [e for e in error_samples if e[3] == 1]  # False Negative

    # Берем топ-N самых уверенных ошибок
    fp_errors = sorted(fp_errors, key=lambda x: x[2], reverse=True)[:top_n]
    fn_errors = sorted(fn_errors, key=lambda x: 1 - x[2], reverse=True)[:top_n]

    # Сохраняем информацию об ошибках
    with open(os.path.join(save_dir, f"error_samples_epoch_{epoch}.txt"), "w") as f:
        f.write("False Positives (FP):\n")
        for clip_idx, frame_idx, confidence, _ in fp_errors:
            f.write(f"Clip {clip_idx}, Frame {frame_idx}: Confidence {confidence:.4f}\n")

        f.write("\nFalse Negatives (FN):\n")
        for clip_idx, frame_idx, confidence, _ in fn_errors:
            f.write(f"Clip {clip_idx}, Frame {frame_idx}: Confidence {confidence:.4f}\n")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    confidences = [e[2] for e in fp_errors]
    plt.hist(confidences, bins=20, color='red', alpha=0.7)
    plt.title('False Positives Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    confidences = [e[2] for e in fn_errors]
    plt.hist(confidences, bins=20, color='blue', alpha=0.7)
    plt.title('False Negatives Confidence Distribution')
    plt.xlabel('Confidence')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"error_distribution_epoch_{epoch}.png"))
    plt.close()