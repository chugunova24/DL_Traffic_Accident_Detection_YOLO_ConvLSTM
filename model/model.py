import torch
from torch import nn


class YOLOBackboneConvLSTM(nn.Module):
    def __init__(self,
                 yolo_ckpt: str = None,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 sequence_length: int = 50,
                 img_size: int = 256):
        super().__init__()
        from ultralytics import YOLO

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.img_size = img_size

        # Загрузка YOLO
        yolo = YOLO(yolo_ckpt)
        self.yolo_model = yolo.model
        self.yolo_main = self.yolo_model.model
        self.feature_layer_idx = len(self.yolo_main) - 2
        self._hook_out = None

        def hook_fn(module, input, output):
            self._hook_out = output[-1] if isinstance(output, (list, tuple)) else output

        target_layer = self.yolo_main[self.feature_layer_idx]
        target_layer.register_forward_hook(hook_fn)

        NEW_SIZE = 256  # 256/32=8

        # Фиктивный прогон для определения размерности
        with torch.no_grad():
            dummy = torch.zeros(1, 3, NEW_SIZE, NEW_SIZE)
            _ = self.yolo_model(dummy)
            feat = self._hook_out
            if feat is None:
                raise RuntimeError("YOLO hook не сработал на dummy-запуске")
            self.C_feat, self.H_feat, self.W_feat = feat.shape[1:]

        # ConvLSTM
        self.convlstm = ConvLSTM(
            input_dim=self.C_feat,
            hidden_dim=[hidden_dim],
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            padding=(1, 1)
        )

        # Регуляризация
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        b, seq_len, C, H, W = x.size()

        # Автоматическое масштабирование до нужного размера
        if (H, W) != (self.img_size, self.img_size):
            x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear',
                                                align_corners=False)

        x_in = x.view(b * seq_len, C, H, W)

        # Прогон через YOLO
        _ = self.yolo_model(x_in)
        feats = self._hook_out
        if feats is None:
            raise RuntimeError("YOLO hook не сработал")

        # Подготовка для ConvLSTM
        feats = feats.view(b, seq_len, self.C_feat, self.H_feat, self.W_feat)

        # ConvLSTM
        output, _ = self.convlstm(feats)
        convlstm_out = output[0]

        # Пулинг и регуляризация
        pooled = convlstm_out.mean(dim=[3, 4])
        pooled = self.dropout(pooled)

        # BatchNorm
        b, seq_len, hidden_dim = pooled.shape
        pooled = pooled.view(-1, hidden_dim)
        pooled = self.batchnorm(pooled)
        pooled = pooled.view(b, seq_len, hidden_dim)

        # Классификация
        logits = self.classifier(pooled).squeeze(-1)
        return logits
