def calculate_iou(preds, masks):
    preds = torch.argmax(preds, dim=1)  # Преобразование предсказаний в индексы классов
    masks = torch.argmax(masks, dim=1)  # Преобразование масок в индексы классов
    iou = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        iou += (intersection + 1e-6) / (union + 1e-6)  # Добавляем epsilon для стабильности
    return iou / num_classes  # Среднее IoU по всем классам
