import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
# import escnn
# from escnn import gspaces
# from escnn import nn as enn

class Z2CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Z2CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(20, 20, kernel_size=4)  # No padding for 4x4 kernel

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(20)
        self.bn5 = nn.BatchNorm2d(20)
        self.bn6 = nn.BatchNorm2d(20)
        self.bn7 = nn.BatchNorm2d(20)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # After conv2
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer
        self.fc = nn.Linear(20, num_classes)  # Final layer output

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Max-pool after conv2
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        # Global average pooling
        x = torch.mean(x, dim=(2, 3))  # Reduces spatial dims to 1x1
        x = self.fc(x)
        return x

# Helper to rotate regular conv filters
def rotate_2d_tensor(tensor, k: int):
    if k % 4 == 0:
        return tensor
    elif k % 4 == 1:
        return torch.flip(tensor.transpose(-2, -1), dims=[-2])
    elif k % 4 == 2:
        return torch.flip(tensor, dims=[-2, -1])
    elif k % 4 == 3:
        return torch.flip(tensor.transpose(-2, -1), dims=[-1])

# Rotate group-aware filters: shape (Cout, Cin, 4, Kh, Kw)
def rotate_g_filter(tensor, k: int):
    Cout, Cin, G, Kh, Kw = tensor.shape
    rotated = []

    for g in range(G):
        patch = tensor[:, :, g]  # shape (Cout, Cin, Kh, Kw)
        rotated_patch = rotate_2d_tensor(patch, k)  # shape (Cout, Cin, Kh, Kw)
        rotated.append(rotated_patch)

    rotated = torch.stack(rotated, dim=2)  # (Cout, Cin, 4, Kh, Kw)
    rotated = torch.roll(rotated, shifts=k, dims=2)  # roll group axis
    return rotated

# G-CNN block
class GConvZ2toP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GConvZ2toP4, self).__init__()
        self.base_filters = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.base_filters, nonlinearity='relu')
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C, H, W = x.shape
        rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
        rotated_filters = []
        for r in rotations:
            rotated = torch.rot90(self.base_filters, r, dims=[2, 3])
            rotated_filters.append(rotated)
        filters = torch.stack(rotated_filters, dim=2)  # (out_c, in_c, 4, k, k)

        outputs = []
        for r in rotations:
            filt = filters[:, :, r, :, :]
            out = F.conv2d(x, filt, stride=self.stride, padding=self.padding)
            outputs.append(out.unsqueeze(2))  # (B, out_c, 1, H, W)

        return torch.cat(outputs, dim=2)  # (B, out_c, 4, H, W)

class GConvP4toP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GConvP4toP4, self).__init__()
        self.base_filters = nn.Parameter(torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.base_filters.view(out_channels, -1, kernel_size, kernel_size), nonlinearity='relu')
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C, G, H, W = x.shape
        filters = []
        for r_out in range(4):
            rotated_filters = []
            for r_in in range(4):
                f = torch.rot90(self.base_filters[:, :, r_in, :, :], r_out, dims=[2, 3])
                rotated_filters.append(f)
            stacked = torch.stack(rotated_filters, dim=2)  # (out_c, in_c, 4, k, k)
            filters.append(stacked)
        filters = torch.stack(filters, dim=2)  # (out_c, in_c, 4, 4, k, k)

        outputs = []
        for r_out in range(4):
            out_sum = 0
            for r_in in range(4):
                inp = x[:, :, r_in, :, :]  # (B, in_c, H, W)
                filt = filters[:, :, r_in, r_out, :, :]  # (out_c, in_c, k, k)
                out_sum += F.conv2d(inp, filt, stride=self.stride, padding=self.padding)
            outputs.append(out_sum.unsqueeze(2))  # (B, out_c, 1, H, W)

        return torch.cat(outputs, dim=2)  # (B, out_c, 4, H, W)


class GroupMaxPool(nn.Module):
    def forward(self, x):
        # x: (B, C, G, H, W)
        return torch.max(x, dim=2)[0]  # → (B, C, H, W)


class GCNN(nn.Module):
    def __init__(self):
        super(GCNN, self).__init__()

        self.gconv1 = GConvZ2toP4(1, 10)
        self.bn1 = nn.BatchNorm3d(10)

        self.gconv2 = GConvP4toP4(10, 10)
        self.bn2 = nn.BatchNorm3d(10)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gconv3 = GConvP4toP4(10, 10)
        self.bn3 = nn.BatchNorm3d(10)

        self.gconv4 = GConvP4toP4(10, 10)
        self.bn4 = nn.BatchNorm3d(10)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gconv5 = GConvP4toP4(10, 10)
        self.bn5 = nn.BatchNorm3d(10)

        self.gconv6 = GConvP4toP4(10, 10)
        self.bn6 = nn.BatchNorm3d(10)

        self.gconv7 = GConvP4toP4(10, 10, kernel_size=4, stride=1, padding=0)
        self.bn7 = nn.BatchNorm3d(10)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Final spatial collapse
        self.fc = nn.Linear(10 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.gconv1(x)))
        x = F.relu(self.bn2(self.gconv2(x)))
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.pool1(x)
        x = x.view(x.size(0), 10, 4, x.size(-2), x.size(-1))
        x = F.relu(self.bn3(self.gconv3(x)))
        x = F.relu(self.bn4(self.gconv4(x)))
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.pool2(x)
        x = x.view(x.size(0), 10, 4, x.size(-2), x.size(-1))
        x = F.relu(self.bn5(self.gconv5(x)))
        x = F.relu(self.bn6(self.gconv6(x)))
        x = F.relu(self.bn7(self.gconv7(x)))
        group_pool = GroupMaxPool()
        x = group_pool(x)        # Output shape: (B, 10, 4, 4)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# P4-W Model block
class Partial_GConvZ2toP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Partial_GConvZ2toP4, self).__init__()
        self.base_filters = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.alpha_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=True)
        nn.init.kaiming_normal_(self.base_filters, nonlinearity='relu')
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C, H, W = x.shape
        rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
        rotated_filters = []
        for r in rotations:
            rotated = torch.rot90(self.base_filters, r, dims=[2, 3])
            rotated_filters.append(rotated)
        filters = torch.stack(rotated_filters, dim=2)  # (out_c, in_c, 4, k, k)

        outputs = []
        for r in rotations:
            filt = filters[:, :, r, :, :]
            out = F.conv2d(x, filt, stride=self.stride, padding=self.padding)
            alpha = torch.sigmoid(self.alpha_logits[r])
            out = out * alpha
            outputs.append(out.unsqueeze(2))  # (B, out_c, 1, H, W)

        return torch.cat(outputs, dim=2)  # (B, out_c, 4, H, W)

class Partial_GConvP4toP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Partial_GConvP4toP4, self).__init__()
        self.base_filters = nn.Parameter(torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size))
        self.alpha_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad = True)
        nn.init.kaiming_normal_(self.base_filters.view(out_channels, -1, kernel_size, kernel_size), nonlinearity='relu')
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C, G, H, W = x.shape
        filters = []
        for r_out in range(4):
            rotated_filters = []
            for r_in in range(4):
                f = torch.rot90(self.base_filters[:, :, r_in, :, :], r_out, dims=[2, 3])
                rotated_filters.append(f)
            stacked = torch.stack(rotated_filters, dim=2)  # (out_c, in_c, 4, k, k)
            filters.append(stacked)
        filters = torch.stack(filters, dim=2)  # (out_c, in_c, 4, 4, k, k)

        outputs = []
        for r_out in range(4):
            out_sum = 0
            for r_in in range(4):
                inp = x[:, :, r_in, :, :]  # (B, in_c, H, W)
                filt = filters[:, :, r_in, r_out, :, :]  # (out_c, in_c, k, k)
                out_sum += F.conv2d(inp, filt, stride=self.stride, padding=self.padding)
            alpha = torch.sigmoid(self.alpha_logits[r_out])
            out_sum = out_sum * alpha
            outputs.append(out_sum.unsqueeze(2))  # (B, out_c, 1, H, W)

        return torch.cat(outputs, dim=2)  # (B, out_c, 4, H, W)


class GroupMaxPool(nn.Module):
    def forward(self, x):
        # x: (B, C, G, H, W)
        return torch.max(x, dim=2)[0]  # → (B, C, H, W)


class P4W_CNN(nn.Module):
    def __init__(self):
        super(P4W_CNN, self).__init__()

        self.gconv1 = Partial_GConvZ2toP4(1, 10)
        self.bn1 = nn.BatchNorm3d(10)

        self.gconv2 = Partial_GConvP4toP4(10, 10)
        self.bn2 = nn.BatchNorm3d(10)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gconv3 = Partial_GConvP4toP4(10, 10)
        self.bn3 = nn.BatchNorm3d(10)

        self.gconv4 = Partial_GConvP4toP4(10, 10)
        self.bn4 = nn.BatchNorm3d(10)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gconv5 = Partial_GConvP4toP4(10, 10)
        self.bn5 = nn.BatchNorm3d(10)

        self.gconv6 = Partial_GConvP4toP4(10, 10)
        self.bn6 = nn.BatchNorm3d(10)

        self.gconv7 = Partial_GConvP4toP4(10, 10, kernel_size=4, stride=1, padding=0)
        self.bn7 = nn.BatchNorm3d(10)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Final spatial collapse
        self.fc = nn.Linear(10 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.gconv1(x)))
        x = F.relu(self.bn2(self.gconv2(x)))
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.pool1(x)
        x = x.view(x.size(0), 10, 4, x.size(-2), x.size(-1))
        x = F.relu(self.bn3(self.gconv3(x)))
        x = F.relu(self.bn4(self.gconv4(x)))
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.pool2(x)
        x = x.view(x.size(0), 10, 4, x.size(-2), x.size(-1))
        x = F.relu(self.bn5(self.gconv5(x)))
        x = F.relu(self.bn6(self.gconv6(x)))
        x = F.relu(self.bn7(self.gconv7(x)))
        group_pool = GroupMaxPool()
        x = group_pool(x)        # Output shape: (B, 10, 4, 4)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def expected_calibration_error(conf, correct, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece  = torch.zeros(1, device=conf.device)
    for i in range(n_bins):
        mask   = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc   = correct[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece  += mask.float().mean() * (acc - bin_conf).abs()
    return ece.item()

@torch.no_grad()
def compute_ece(model, test_loader, device="cuda", n_bins=15):
    model.eval()
    all_conf, all_correct = [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                       # (batch, C)
        prob   = F.softmax(logits, dim=1)       # convert to probabilities
        conf, pred = prob.max(dim=1)            # highest prob per sample
        all_conf.append(conf)
        all_correct.append(pred.eq(y))          # Boolean tensor

    conf_tensor    = torch.cat(all_conf)        # shape (N,)
    correct_tensor = torch.cat(all_correct)     # shape (N,)

    return expected_calibration_error(conf_tensor,
                                      correct_tensor,
                                      n_bins=n_bins)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split


def get_rotated_mnist(batch_size: int = 128, seed: int = 42):

    # ---------------- transforms -----------------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean / std
        transforms.RandomRotation(degrees=(-180, 180)),   
    ])

    test_transform_no_rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean / std        
    ])

    test_transform_with_rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean / std
        transforms.RandomRotation(degrees=(-180, 180)),   
    ])

    # Option C in code (example)
    full_train = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=train_transform)

    train_ds, val_ds = random_split(
        full_train, [55_000, 5_000],
        generator=torch.Generator().manual_seed(seed)
    )

    test_ds_no_rotation = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=test_transform_no_rotation)

    test_ds_with_rotation = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform_with_rotation)

    

    # ---------------- loaders --------------------------------------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader_no_rotation  = DataLoader(test_ds_no_rotation,  batch_size=batch_size, shuffle=False)
    test_loader_with_rotation = DataLoader(test_ds_with_rotation, batch_size=batch_size, shuffle=False)    

    return train_loader, val_loader, test_loader_no_rotation, test_loader_with_rotation

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f} | Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader, test_type):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{test_type}: Average loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%\n')
    return accuracy

# Training loop
def training_loop(model, device, train_loader, test_loader_no_rotation, test_loader_with_rotation, val_loader, hyper_parameters):
    # --- Training Loop ---

    MODEL_NAME = type(model).__name__
    print(f"Training {type(model).__name__} Model...")
    print(f"Total parameters: {count_parameters(model):,}")

    LEARNING_RATE = hyper_parameters["LEARNING_RATE"]
    WEIGHT_DECAY = hyper_parameters["WEIGHT_DECAY"]
    MOMENTUM = hyper_parameters["MOMENTUM"]
    EPOCHS = hyper_parameters["EPOCHS"]
    SCHEDULER_PATIENCE = hyper_parameters["SCHEDULER_PATIENCE"]
    SCHEDULER_FACTOR = hyper_parameters["SCHEDULER_FACTOR"]
    EARLY_STOPPING_PATIENCE = hyper_parameters["EARLY_STOPPING_PATIENCE"]
    BEST_MODEL_PATH = "./models/" + MODEL_NAME + "_best_tuned.pth"

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

    print("\nStarting Training...")
    best_val_accuracy = 0.0
    epochs_no_improve = 0 # Counter for early stopping

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train(model, device, train_loader, optimizer, epoch)
        val_accuracy = test(model, device, val_loader, "Validation set") # Get validation accuracy
        
        scheduler.step()

        # Check for improvement and save best model
        if val_accuracy > best_val_accuracy:
            print(f"Validation accuracy improved ({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%). Saving model...")
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0 # Reset counter
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")

        # Early stopping check
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    print("\nTraining finished.")

    # --- Final Evaluation on Test Set ---
    print(f"\nLoading best model from {BEST_MODEL_PATH} for final evaluation...")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        test_accuracy_no_rotation = test(model, device, test_loader_no_rotation, "Test set no rotation")
        test_accuracy_with_rotation = test(model,device, test_loader_with_rotation, "Test set with rotation")
        print(f"\nFinal Test Accuracy without test augmentation (Best {type(model).__name__} Model): {test_accuracy_no_rotation:.2f}%")
        print(f"\nFinal Test Accuracy with test augmentation (Best {type(model).__name__} Model): {test_accuracy_with_rotation:.2f}%")

    except FileNotFoundError:
        print("Best model file not found. Run training first.")
    except Exception as e:
         print(f"An error occurred loading the best model: {e}")

    return test_accuracy_no_rotation, test_accuracy_with_rotation

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader_no_rotation, test_loader_with_rotation = get_rotated_mnist()

# Hyperparameters
hyper_parameters = {
  "LEARNING_RATE": 0.1,
  "WEIGHT_DECAY": 5e-4,
  "MOMENTUM": 0.9,
  "EPOCHS": 150,
  "SCHEDULER_PATIENCE": 10,
  "SCHEDULER_FACTOR": 0.5,
  "EARLY_STOPPING_PATIENCE": 25,
}

seeds   = [0, 1, 2, 3, 4]

models  = {"Z2": Z2CNN,
           "P4": GCNN,
           "P4-W": P4W_CNN}

# --------------- results dict ------------------------------------- #
results = {name: {"no_rotation": [], "rotation": []} for name in models}
ece_results = {name: {"no_rotation": [], "rotation": []} for name in models}

# ------------------------------------------------------------------ #
for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    for name, ModelClass in models.items():
        model = ModelClass().to(device)
        best_acc_no_rotation, best_acc_with_rotation = training_loop(model, device, train_loader, test_loader_no_rotation, test_loader_with_rotation, val_loader, hyper_parameters)

        results[name]["no_rotation"].append(best_acc_no_rotation)
        print(f"seed {seed} \t{name} (No Rotation)\t{best_acc_no_rotation:.2f}%")

        results[name]["rotation"].append(best_acc_with_rotation)
        print(f"seed {seed} \t{name} (With Rotation)\t{best_acc_with_rotation:.2f}%")

        ece_no_rotation = compute_ece(model, test_loader_no_rotation, device="cuda", n_bins=15)
        ece_results[name]["no_rotation"].append(ece_no_rotation)

        print(f"Expected Calibration Error no test rotation: {ece_no_rotation:.4f}")

        ece_with_rotation = compute_ece(model, test_loader_with_rotation, device="cuda", n_bins=15)
        ece_results[name]["rotation"].append(ece_with_rotation)

        print(f"Expected Calibration Error with test rotation: {ece_with_rotation:.4f}")


# ----------------- summary ------------------ #
for name, score_dict in results.items():
    mean_no_rotation = np.mean(score_dict["no_rotation"])
    std_no_rotation  = np.std(score_dict["no_rotation"])
    mean_with_rotation = np.mean(score_dict["rotation"])
    std_with_rotation  = np.std(score_dict["rotation"])
    
    print(f"{name:8s}: No Rotation - {mean_no_rotation:.2f} ± {std_no_rotation:.2f} (n={len(score_dict['no_rotation'])})")
    print(f"{name:8s}: With Rotation - {mean_with_rotation:.2f} ± {std_with_rotation:.2f} (n={len(score_dict['rotation'])})")


print("\n------- ECE Results ------------")
for name, ece_dict in ece_results.items():
    mean_no_rotation = np.mean(ece_dict["no_rotation"])
    mean_with_rotation = np.mean(ece_dict["rotation"])
    print(f"{name:8s}: No Rotation - {mean_no_rotation}")
    print(f"{name:8s}: With Rotation - {mean_with_rotation}")
    