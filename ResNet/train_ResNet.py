import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ResNet import Block, ResNet
from training_utils_ResNet import train, validate
from utils_ResNet import save_plots, get_data


def train_ResNet():

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


    epochs = 1
    batch_size = 24
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader = get_data(batch_size)


    print('Training ResNet...')
    model = ResNet(num_layers=101, block=Block,  image_channels=3, num_classes=6).to(device)
    plot_name = f'ResNet_{epochs}_{batch_size}_{learning_rate}'


    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Training parameters: {total_trainable_params:,}")


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model,
            valid_loader,
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)

    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
    print('TRAINING COMPLETE')
