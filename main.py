"""
    @author: Maciej Burakowski, 258969
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

plt.rcParams.update({'font.size': 14})

# === Ustawienia modelu ===
model_args = {
    'seed': 123,
    'batch_size': 256,  # większy niż domyślnie
    'lr': 0.001,
    'momentum': 0.9,
    'epochs': 10,
    'log_interval': 100
}

# === Przygotowanie danych ===
torch.manual_seed(model_args['seed'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_subset, validation_subset = random_split(mnist_train, [50000, 10000])
test_subset = datasets.MNIST('./data', train=False, download=True, transform=transform)

loader_kwargs = {
    'batch_size': model_args['batch_size'],
    'num_workers': 4,       # przyspieszenie wczytywania danych
    'pin_memory': True,     # optymalizacja przy użyciu GPU
    'shuffle': True
}

train_loader = DataLoader(train_subset, **loader_kwargs)
validation_loader = DataLoader(validation_subset, **loader_kwargs)
test_loader = DataLoader(test_subset, **loader_kwargs)

# === Definicja modelu CNN ===
# zwiększono liczbę filtrów dla lepszej reprezentacji

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# === Funkcja treningu ===
def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % model_args['log_interval'] == 0:
            print(f'Train Epoch: {epoch_number} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')
    return train_loss / len(train_loader)

# === Funkcja testująca (walidacja / test) ===
def test(model, device, test_loader, label="Validation", return_preds=False):
    model.eval()
    test_loss = 0.
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            if return_preds:
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{label}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    if return_preds:
        return test_loss, accuracy, y_true, y_pred
    return test_loss, accuracy

# === Macierz błędów ===
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

# === Główna pętla ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = CNN().to(device)

    # === Optymalizator z opcją weight_decay (L2 regularization) ===
    #optimizer = optim.Adam(model.parameters(), lr=model_args['lr'])
    optimizer = optim.Adam(model.parameters(), lr=model_args['lr'], weight_decay=1e-4)  #L2 regularization

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, model_args['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = test(model, device, validation_loader, "Validation")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    test_loss, test_acc, y_true, y_pred = test(model, device, test_loader, "Test", return_preds=True)

    total_time = time.time() - start_time
    print(f"\nTotal training and testing time: {total_time:.2f} seconds")

    # === Wykres strat ===
    plt.plot(range(1, model_args['epochs'] + 1), train_losses, 'o-', label='Training')
    plt.plot(range(1, model_args['epochs'] + 1), val_losses, 'o-', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # === Confusion matrix ===
    plot_confusion_matrix(y_true, y_pred)
