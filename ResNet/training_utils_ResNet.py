import torch
from tqdm import tqdm


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    for images, labels in tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
        train_running_correct += torch.sum(torch.argmax(outputs, dim=1) == labels)

    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * train_running_correct / len(trainloader.dataset)
    return epoch_loss, epoch_acc


def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        valid_running_correct += torch.sum(torch.argmax(outputs, dim=1) == labels)

    epoch_loss = valid_running_loss / len(testloader)
    epoch_acc = 100. * valid_running_correct / len(testloader.dataset)
    return epoch_loss, epoch_acc
