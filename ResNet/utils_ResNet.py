import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

plt.style.use('ggplot')


image_path = Path(" ")
train_dir = image_path / "train"
test_dir = image_path / "test"


data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def get_data(batch_size=64):

    dataset_train = datasets.ImageFolder(root=os.path.abspath(train_dir),
                                         transform=data_transform,
                                         target_transform=None)
    dataset_valid = datasets.ImageFolder(root=os.path.abspath(test_dir),
                                         transform=data_transform,
                                         target_transform=None)

    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=False)
    return train_loader, valid_loader

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):


    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='tab:blue', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='tab:red', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', f"{name}_accuracy.png"))


    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='tab:blue', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='tab:red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', f"{name}_loss.png"))
