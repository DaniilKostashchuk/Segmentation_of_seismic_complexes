if __name__ == "__main__":

    # Разделение данных на обучающую и валидационную выборки
    train_images, val_images = Images_train, Images_test
    train_masks, val_masks = Masks_train, Masks_test

    # Создание датасетов и DataLoader
    train_dataset = SegmentationDataset(train_images, train_masks, num_classes)
    val_dataset = SegmentationDataset(val_images, val_masks, num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Обучение
    num_epochs = 50
    best_iou = 0.0

    for epoch in range(num_epochs):
        train_loss, train_iou = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_dataloader, criterion, device)

        # Обновление планировщика скорости обучения
        scheduler.step(val_iou)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Сохранение лучшей модели
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_unet_segmentation.pth")
            
    # Сохранение финальной модели
    torch.save(model.state_dict(), "final_unet_segmentation.pth")
