import os

import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.model import YOLOBackboneConvLSTM
from model.utils import ApplyToSequence


class VideoProcessor:
    def __init__(self, model_path, yolo_ckpt='yolo11m.pt', sequence_length=50, img_size=256,
                 confidence_threshold=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # Инициализация модели
        self.model = YOLOBackboneConvLSTM(
            yolo_ckpt=yolo_ckpt,
            hidden_dim=256,
            num_layers=1,
            bidirectional=False,
            sequence_length=sequence_length,
            img_size=img_size
        )

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = ApplyToSequence(transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]))

        # Буфер для хранения кадров
        self.frame_buffer = []

    def preprocess_frame(self, frame):
        # Конвертация из BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Конвертация в PIL Image
        pil_image = Image.fromarray(frame_rgb)
        return pil_image

    def process_frame_sequence(self, frames):
        # Преобразование кадров
        transformed_frames = self.transform(frames)
        # Добавление размерности батча
        transformed_frames = transformed_frames.unsqueeze(0)
        # Перемещение на GPU
        transformed_frames = transformed_frames.to(self.device)

        with torch.no_grad():
            outputs = self.model(transformed_frames)
            probs = torch.sigmoid(outputs)

        return probs.cpu().numpy()

    def get_confidence_color(self, prob):
        if prob >= self.confidence_threshold:
            return (0, 0, 255)
        elif prob >= 0.8:
            return (0, 255, 255)
        else:
            return (0, 255, 0)

    def process_video(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Открытие видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            os.path.join(output_dir, 'visualization.mp4'),
            fourcc,
            fps,
            (width, height)
        )
        frame_idx = 0
        crash_probabilities = []

        with tqdm(total=total_frames, desc="Обработка видео.") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                if len(self.frame_buffer) > 0:
                    current_sequence = self.frame_buffer[-min(len(self.frame_buffer), self.sequence_length):]
                    if len(current_sequence) < self.sequence_length:
                        padding = [current_sequence[0] for _ in range(self.sequence_length - len(current_sequence))]
                        current_sequence = padding + current_sequence
                    probs = self.process_frame_sequence(current_sequence)
                    crash_prob = probs[0, -1]
                    crash_probabilities.append(crash_prob)

                    color = self.get_confidence_color(crash_prob)
                    text = f"Crash: {crash_prob:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    confidence_bar_width = int(width * 0.3)
                    confidence_bar_height = 20
                    bar_x = 10
                    bar_y = height - 40
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + confidence_bar_width, bar_y + confidence_bar_height),
                                  (100, 100, 100), -1)
                    fill_width = int(confidence_bar_width * crash_prob)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + fill_width, bar_y + confidence_bar_height),
                                  color, -1)
                    out.write(frame)
                    if crash_prob >= self.confidence_threshold:
                        cv2.imwrite(
                            os.path.join(output_dir, f'crash_frame_{frame_idx:06d}.jpg'),
                            frame
                        )

                if len(self.frame_buffer) > self.sequence_length:
                    self.frame_buffer.pop(0)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()


def main():
    # Параметры
    model_path = 'best_model.pth'
    video_path = 'video.mp4'
    output_dir = '/workspace/output'
    confidence_threshold = 0.8
    processor = VideoProcessor(model_path, confidence_threshold=confidence_threshold)
    results = processor.process_video(video_path, output_dir)

    print(f"\nОбработка завершена. Результаты сохранены в {output_dir}")


if __name__ == '__main__':
    main()