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
import os


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        
        # Standard ResNet Conv1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Standard ResNet Conv2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Block injected here
        self.se = SEBlock(out_channels, reduction)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply Squeeze-and-Excitation
        out = self.se(out)
        
        # Residual connection
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SEResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SEResNet18():
    return SEResNet(SEBasicBlock, [2, 2, 2, 2])

def train_one_epoch_mixup(model, train_loader, optimizer, criterion, epoch, max_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{max_epochs}]")
    
    for batch_idx, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate MixUp Data
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
        inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate MixUp Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy calculation (Compare against the dominant class)
        _, preds = outputs.max(1)
        total += labels.size(0)
        # MixUp accuracy approximation
        correct += (lam * preds.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())

        loop.set_postfix(loss=loss.item())

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

def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path, scheduler=None):
    best_loss = float('inf') 
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print(f"\nStarting training on {device} for {max_epochs} epochs...")
    print("Methods active: SE-Blocks, MixUp, Cutout, CosineScheduler.")
    
    total_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # Train & Val steps
        train_loss, train_acc = train_one_epoch_mixup(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print ALL metrics (Train Loss, Train Acc, Val Loss, Val Acc)
        print(f"Epoch [{epoch+1}/{max_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_duration:.1f}s")

        # Logic to save the model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best model saved! (Val Loss: {best_loss:.4f})")
    
    total_end = time.time()
    total_duration = total_end - total_start
    print(f"\nTraining Finished. Total Time: {total_duration/60:.2f} minutes.")

    # Plotting
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Acc (MixUp)')
    plt.plot(val_accuracies, label='Validation Acc')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# MAIN
def main():
    # Configuration
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 200 
    LEARNING_RATE = 0.1 

    # Data Augmentation with Cutout
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model setup
    print("\n" + "="*40)
    print("  STARTING STAGE 5: IMPROVING SOTA")
    print("  Methods: SE-Blocks, MixUp, Cutout")
    print("="*40)
    
    model = SEResNet18().to(device)
    summary(model, input_size=(3, 32, 32))
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Start Training
    train(model, train_loader, val_loader, optimizer, criterion, 
          max_epochs=EPOCHS, device=device, 
          model_save_path="best_stage5_improved.pth", 
          scheduler=scheduler)

    # Final Test Evaluation
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load("best_stage5_improved.pth"))
    test_loss, test_acc = validation(model, test_loader, criterion, device)
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")

if __name__ == "__main__":
    main()