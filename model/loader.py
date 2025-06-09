import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CCDSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=50, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        for label_name in ['Crash', 'Normal']:
            label_dir = os.path.join(root_dir, label_name)
            label = 1 if label_name == 'Crash' else 0

            for video_name in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_name)
                if not os.path.isdir(video_path):
                    continue

                frame_files = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(('.jpg', '.jpeg', '.png'))
                ])

                num_sequences = len(frame_files) // sequence_length
                for i in range(num_sequences):
                    start = i * sequence_length
                    sequence = frame_files[start:start + sequence_length]
                    if len(sequence) == sequence_length:
                        self.samples.append((sequence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label = self.samples[idx]

        frames = []
        for frame_path in sequence:
            img = Image.open(frame_path).convert('RGB')
            frames.append(img)

        if self.transform:
            frames = self.transform(frames)

        if not isinstance(frames, torch.Tensor):
            frames = torch.stack(frames)

        # Метка на каждый кадр
        sequence_length = frames.shape[0]  # T
        labels = torch.full((sequence_length,), label, dtype=torch.float32)

        return frames, labels



