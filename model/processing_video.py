import os
from collections import deque

import cv2
import torch
from PIL import Image
from torchvision import transforms

from model.model import YOLOBackboneConvLSTM


def load_trained_model(model_path, yolo_checkpoint, sequence_length, device):
    model = YOLOBackboneConvLSTM(yolo_checkpoint, sequence_length=sequence_length)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def predict_and_save_video_background(video_path, output_path, model, sequence_length=50, device='cpu', threshold=0.5):
    """
    Обрабатывает видео в фоновом режиме, сохраняя результаты в новый файл.
    """
    print(f"[INFO] Открытие видео: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        return

    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] Сохранение результата в: {output_path}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    frame_queue = deque(maxlen=sequence_length)
    frame_buffer = deque(maxlen=sequence_length)
    frame_count = 0
    prediction_batches = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = transform(pil_image)
        frame_queue.append(tensor)
        frame_buffer.append(frame.copy())

        if len(frame_queue) == sequence_length:
            input_tensor = torch.stack(list(frame_queue)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            damage_flags = (probs > threshold).astype(int)
            prediction_batches += 1

            for i in range(sequence_length):
                vis_frame = frame_buffer[i].copy()

                confidence = probs[i] * 100
                label = f"DAMAGE ({confidence:.1f}%)" if damage_flags[i] else f"OK ({100 - confidence:.1f}%)"
                color = (0, 0, 255) if damage_flags[i] else (0, 255, 0)

                cv2.putText(vis_frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                if damage_flags[i]:
                    cv2.rectangle(vis_frame, (5, 5), (width - 5, height - 5), (0, 0, 255), 3)

                out_writer.write(vis_frame)

    cap.release()
    out_writer.release()
    print(f"[INFO] Обработка завершена.")


def run_video_inference_background():
    MODEL_PATH = "/home/chugun/Projects/Python/car_damage_detection/models/best_model_2.pth"
    YOLO_CHECKPT = "/home/chugun/Projects/Python/car_damage_detection/models/yolo11m.pt"
    VIDEO_PATH = "/home/chugun/Projects/Python/car_damage_detection/architecture/test_data/Вологда_авария_1_прогон2.mp4"
    OUTPUT_PATH = "/home/chugun/Projects/Python/car_damage_detection/architecture/test_result/Вологда_авария_1_прогон2_output.mp4"
    SEQUENCE_LENGTH = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Используемое устройство: {device}")

    print("[INFO] Загрузка модели...")
    model = load_trained_model(MODEL_PATH, YOLO_CHECKPT, SEQUENCE_LENGTH, device)

    print("[INFO] Запуск предсказаний с сохранением видео в фоновом режиме...")
    predict_and_save_video_background(VIDEO_PATH, OUTPUT_PATH, model,
                                      sequence_length=SEQUENCE_LENGTH,
                                      device=device, threshold=0.7)


if __name__ == "__main__":
    run_video_inference_background()