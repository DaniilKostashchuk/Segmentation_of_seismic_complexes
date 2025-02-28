class SegmentationDataset(Dataset):
    def __init__(self, images, masks, num_classes, transform=None):
        self.images = images  # Массив изображений (numpy array)
        self.masks = masks    # Массив масок (numpy array)
        self.num_classes = num_classes  # Количество классов
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Нормализация изображения
        image = image / 255.0  # Нормализация изображения

        # Проверка и ограничение значений в маске
        mask = np.clip(mask, 0, self.num_classes - 1)  # Ограничение значений
        mask = np.eye(self.num_classes)[mask.astype(np.uint8)]  # One-hot encoding

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Преобразование в тензоры
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        return image, mask
