criterion = nn.CrossEntropyLoss()  # Кросс-энтропия для многоклассовой классификации
optimizer = optim.AdamW(  
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,  #  L2-регуляризациz
    eps=1e-8  
)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=20, verbose=True)  # Планировщик скорости обучения
