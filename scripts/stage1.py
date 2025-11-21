import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from torchsummary import summary
from tqdm import tqdm
import time  


# Model definition
class CNN_simples(nn.Module):
    def __init__(self):
        super(CNN_simples, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # output 16x16
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output 8x8
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output 4x4
        )
        
        self.flatten = nn.Flatten()
        
        # Linear Layers (128 * 4 * 4 = 2048)
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # CIFAR-10, has 10 classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# One epoch train
def train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{max_epochs}]")
    
    for batch_idx, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accurate loss calculation (weighted by batch size)
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validation(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def plot_confusion_matrix(cm, class_names):
    num_classes = len(class_names)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Stage 1")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values to the matrix
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predictions')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()

# Training function
def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path):
    best_acc = 0.0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print(f"\nStarting training on {device}...")
    
    total_train_start = time.time() 

    for epoch in range(max_epochs):
        
        epoch_start = time.time() 
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Epoch results
        print(f"Val Loss: {val_loss:.4f}  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.2f}%  Train Acc: {train_acc:.2f}% | Time: {epoch_duration:.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best model saved with acc: {best_acc:.2f}%")
    
    total_train_end = time.time()
    total_train_duration = total_train_end - total_train_start
    print(f"\nTraining finished in {total_train_duration // 60:.0f}m {total_train_duration % 60:.0f}s")

    # Plotting
    plt.figure(figsize=(14,5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.show()

#   MAIN
def main():
    script_start = time.time()

    # Configuration
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    class_names = full_dataset.classes
    
    # Setup
    model = CNN_simples().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 50
    model_save_path = "models/best_stage_1.pth"
       
    # Train and Plot
    train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path)
    
    # Generate Confusion Matrix after training
    print("Generating confusion matrix on Test set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = metrics.confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    script_end = time.time()
    total_duration = script_end - script_start
    print(f"\n=== Total execution time: {total_duration // 60:.0f}m {total_duration % 60:.0f}s ===")

if __name__ == "__main__":
    main()