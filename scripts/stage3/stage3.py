import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import time
from tqdm import tqdm

# Implementation of a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False) # bias=False because BN handles it
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ModernCNN(nn.Module):
    def __init__(self):
        super(ModernCNN, self).__init__()
        
        # Initial Stem (Input processing)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 32 channels (32x32 size)
        self.layer1 = ResidualBlock(32, 32, stride=1)
        
        # 64 channels (16x16 size)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        
        # 128 channels (8x8 size)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        # 256 channels (4x4 size)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        # Classification Head
        # Global Average Pooling is more modern/efficient than flattening huge layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()
        
        # Dropout for Regularization
        self.dropout = nn.Dropout(0.3) 
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
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

        # Epoch result 
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

def main():
    # Configuration
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation
    # More aggressive transforms for the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Shift image slightly
        transforms.RandomHorizontalFlip(),    # Mirror image
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Standard transform for validation/test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Stage 3: Modern CNN
    print("\n" + "="*40)
    print("  STARTING STAGE 3: MODERN CNN")
    print("="*40)
    
    model = ModernCNN().to(device)
    summary(model, input_size=(3, 32, 32))
    
    # CrossEntropy with label_smoothing parameter
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with Weight Decay (L2 Regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    
    train(model, train_loader, val_loader, optimizer, criterion, 
          max_epochs=50, device=device, model_save_path="models/best_stage3_modern.pth")

if __name__ == "__main__":
    main()