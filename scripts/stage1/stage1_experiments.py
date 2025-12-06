import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from torchsummary import summary

class CNN_Linear(nn.Module):
    def __init__(self):
        super(CNN_Linear, self).__init__()
        # Same architecture as Baseline but REMOVED ALL ReLUs
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # nn.ReLU(), <--- REMOVED
            nn.MaxPool2d(2, 2) 
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(), <--- REMOVED
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(), <--- REMOVED
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(), <--- REMOVED
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            # nn.ReLU(), <--- REMOVED
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_experiment(model, name, train_loader, val_loader, device, epochs):
    print(f"\n--- Running Experiment: {name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Tracking metrics
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }
    
    model.to(device)
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_end = time.time()
        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | Time: {epoch_end - epoch_start:.2f}s")
        
    print(f"Experiment Finished. Total Time: {(time.time() - total_start)/60:.2f} minutes.")
    return history

# MAIN
def main():
    torch.backends.cudnn.benchmark = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    
    # Data Setup
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split 80% Training / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize Model
    exp_model = CNN_Linear().to(device)

    # Model Summary
    print("Model Architecture & Parameters:")
    try:
        summary(exp_model, input_size=(3, 32, 32))
    except Exception as e:
        print(f"Summary error: {e}")
    
    # Run Experiment
    history = train_experiment(exp_model, "No Non-Linearity (Linear Collapse)", train_loader, val_loader, device, epochs=50)
    
    # Plot Results
    plt.figure(figsize=(14, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Evolution (Linear Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution (Linear Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Stage 1 Experiment: Impact of Removing ReLU (Linear Collapse)')
    plt.tight_layout()
    plt.savefig('stage1_linear_experiment.png')
    plt.show()

if __name__ == "__main__":
    main()