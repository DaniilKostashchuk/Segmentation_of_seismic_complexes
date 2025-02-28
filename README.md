# Segmentation of seismic facies
Этот репозиторий содержит реализацию обучения модели U-Net для задачи сегментации изображений. Модель обучается на наборе данных, содержащем суммарный сейсмический куб и куб выделенных сейсмический комплексов в формате numpy.ndarray, и может быть использована для сегментации сейсмических разрезов, полученных на этой же территории (континентальный шельф Северного моря у берегов Нидерландов), или может быть переобучена для нового набора данных.

## Кастомный датасет SegmentationDataset
Класс SegmentationDataset наследуется от torch.utils.data.Dataset и предназначен для работы с массивами изображений и масок в формате NumPy. Изображения нормализуются, а маски преобразуются в one-hot encoding.

## Архитектура модели
Используется архитектура U-Net с энкодером на основе ResNet18. Модель поддерживает многоклассовую сегментацию с активацией softmax на выходе. Веса энкодера инициализируются предобученными на ImageNet.
```bash
model = Unet(
    encoder_name="resnet18",  # Используем ResNet18 в качестве энкодера
    encoder_weights="imagenet",  # Предобученные веса на ImageNet
    in_channels=1,  # 1 канал для grayscale изображений
    classes=num_classes,  # Количество классов
    activation="softmax",  # softmax для многоклассовой классификации
)
```
## Функция потерь и оптимизатор
<p>Для многоклассовой классификации используется функция потерь CrossEntropyLoss.<p> 
<p>Оптимизатор — AdamW с L2-регуляризацией.<p>
<p>Планировщик ReduceLROnPlateau динамически изменяет скорость обучения в зависимости от качества модели на валидационной выборке.<p>
  
## Метрика качества (IoU)
Реализована функция calculate_iou для вычисления Intersection over Union (IoU) для многоклассовой сегментации, IoU вычисляется для каждого класса и усредняется.
```bash
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
```
## Ссылки
Данные для обучения были взяты:
<p>https://zenodo.org/records/3755060<p>
<p><p>
<p><p>
