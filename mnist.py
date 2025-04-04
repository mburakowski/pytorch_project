import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import time

# --- Ustawienia ogólne ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
seed = 42
torch.manual_seed(seed)

# --- Transformacja danych ---
# Normalizujemy obrazy MNIST do wartości w [0, 1]
transform = transforms.ToTensor()

# --- Wczytanie danych MNIST ---
# Dane treningowe (60 000 przykładów)
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
# Dane testowe (10 000 przykładów)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# --- Podział na train i validation (80/20) ---
train_size = int(0.8 * len(dataset))  # 48 000
val_size = len(dataset) - train_size  # 12 000
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# --- Tworzenie DataLoaderów ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --- Definicja modelu (MLP: Multi-Layer Perceptron) ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),               # Spłaszczenie obrazu 28x28 → 784
            nn.Linear(784, 512),        # Warstwa ukryta 1
            nn.ReLU(),
            nn.Linear(512, 512),        # Warstwa ukryta 2
            nn.ReLU(),
            nn.Linear(512, 10)          # Wyjście: 10 klas (cyfry 0–9)
        )

    def forward(self, x):
        return self.model(x)

# --- Funkcja treningu modelu ---
def train(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)                    # Forward pass
        loss = loss_fn(pred, y)           # Obliczenie straty
        optimizer.zero_grad()
        loss.backward()                   # Backpropagation
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- Ewaluacja modelu ---
def evaluate(model, loader, loss_fn, label="Val", return_preds=False):
    model.eval()
    correct, total_loss = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            if return_preds:
                y_true += y.cpu().tolist()
                y_pred += preds.cpu().tolist()
    acc = 100 * correct / len(loader.dataset)
    print(f"{label} accuracy: {acc:.2f}%  loss: {total_loss/len(loader):.4f}")
    if return_preds:
        return acc, total_loss, y_true, y_pred
    return acc, total_loss

# --- Macierz bledu ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# --- Trening modelu ---
model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
print(f"Using {device} device")
start_time = time.time()

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}")
    train_loss = train(model, train_loader, loss_fn, optimizer)
    val_acc, val_loss = evaluate(model, val_loader, loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# --- Ewaluacja na zbiorze testowym + confusion matrix ---
print("\nFinal test:")
test_acc, test_loss, y_true, y_pred = evaluate(model, test_loader, loss_fn, return_preds=True)


end_time = time.time()
print(f"\nTotal training and testing time: {end_time - start_time:.2f} seconds")

# --- Rysowanie confusion matrix ---
plot_confusion_matrix(y_true, y_pred)

# --- Rysowanie wykresu strat ---
plt.plot(range(1, epochs+1), train_losses, label="Train loss")
plt.plot(range(1, epochs+1), val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- Predykcja 9 losowych obrazków ---
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
model.eval()
for ax in axes.flat:
    idx = torch.randint(len(test_dataset), (1,)).item()
    img, label = test_dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        predicted_label = pred.argmax(1).item()
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Pred: {predicted_label}, True: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()
