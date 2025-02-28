num_classes = 6  # Пример: 6 классов
model = Unet(
    encoder_name="resnet18",  # Используем ResNet18 в качестве энкодера
    encoder_weights="imagenet",  # Предобученные веса на ImageNet
    in_channels=1,  # 1 канал для grayscale изображений
    classes=num_classes,  # Количество классов
    activation="softmax",  # Софтмакс для многоклассовой классификации
)

# Перемещение модели на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
