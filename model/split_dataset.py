import os
import random
import shutil

import cv2
from tqdm import tqdm


def extract_frames_from_videos(video_dir, output_dir, max_videos=1500):
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:max_videos]

    for i, video_file in enumerate(tqdm(video_files, desc=f'Processing {video_dir}')):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(video_file)[0]
        output_subdir = os.path.join(output_dir, f'{video_name}')
        os.makedirs(output_subdir, exist_ok=True)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_subdir, f'{frame_idx:05d}.jpg')
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_idx += 1
        cap.release()


def split_dataset(root_dir, output_dir, split_ratio=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    assert sum(split_ratio) == 1.0, "Сумма коэффициентов должна быть 1.0"

    classes = ['Crash', 'Normal']
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            raise ValueError(f"Класс {cls} не найден в {root_dir}")

        video_folders = sorted([f for f in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, f))])
        random.shuffle(video_folders)

        n_total = len(video_folders)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)
        n_test = n_total - n_train - n_val

        print(f"{cls}: {n_total} видео → train: {n_train}, val: {n_val}, test: {n_test}")

        split_map = {
            'train': video_folders[:n_train],
            'val': video_folders[n_train:n_train + n_val],
            'test': video_folders[n_train + n_val:]
        }

        for split, videos in split_map.items():
            for video in videos:
                src = os.path.join(cls_dir, video)
                dst = os.path.join(output_dir, split, cls, video)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)

    print(f"\nРазделение завершено. Данные в {output_dir}")


def main():
    os.makedirs("./frames_CCD/Crash", exist_ok=True)
    os.makedirs("./frames_CCD/Normal", exist_ok=True)

    # Пример использования
    extract_frames_from_videos(
        video_dir='./dataset_CCD/Crash-1500',
        output_dir='./frames_CCD/Crash',
        max_videos=1500
    )
    extract_frames_from_videos(
        video_dir='./dataset_CCD/Normal',
        output_dir='./frames_CCD/Normal',
        max_videos=1500
    )

    split_dataset(
        root_dir='./frames_CCD',            # Где лежат Crash и Normal
        output_dir='./split_CCD',    # Куда сохранить train/val/test
        split_ratio=(0.7, 0.15, 0.15),
        seed=42
    )
