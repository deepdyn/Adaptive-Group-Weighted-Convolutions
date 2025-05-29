import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from e2cnn import gspaces
from e2cnn import nn as enn
from e2cnn.nn import FieldType, GeometricTensor

"""# Standard ResNet Model"""

# ------------------- Basic Block --------------------- #
class BasicBlockZ2(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# ------------------- ResNet‑44  ------------------ #
class ResNet44_Z2(nn.Module):
    """
    Plain ResNet‑44 (Z²) for 28×28 Fashion‑MNIST images.
    Widths: 16‑32‑64
    """
    def __init__(self, num_classes: int = 10, widen_factor: int = 1):
        super().__init__()

        widths = [16, 32, 64]         
        widths = [w * widen_factor for w in widths]

        self.in_channels = widths[0]

        # Stem
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_channels)

        # Stages  (n = 7 blocks each)
        self.stage1 = self._make_stage(BasicBlockZ2, widths[0], 7, stride=1)
        self.stage2 = self._make_stage(BasicBlockZ2, widths[1], 7, stride=2)
        self.stage3 = self._make_stage(BasicBlockZ2, widths[2], 7, stride=2)

        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(widths[2], num_classes)

    # --------------------------------------------------------------
    def _make_stage(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    # --------------------------------------------------------------
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)           # 28×28
        out = self.stage2(out)           # 14×14
        out = self.stage3(out)           # 7×7
        out = self.avg_pool(out)         # 1×1
        out = torch.flatten(out, 1)
        return self.fc(out)


"""# ResNet P4M model"""

class ResNetBlock(torch.nn.Module):
    r"""Pre‑activation residual basic block for p4m‑ResNet."""
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType,
                 kernel_size: int, stride: int = 1):
        super().__init__()

        self.in_type  = in_type
        self.out_type = out_type

        # --- main branch ---------------------------------------------------
        self.bn1   = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type)
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.bn2   = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        # --- shortcut ------------------------------------------------------
        self.shortcut = None
        needs_proj = stride != 1 or in_type.size != out_type.size
        if needs_proj:
            # 1×1 equivariant projection to match shape and/or downsample
            self.shortcut = enn.R2Conv(
                in_type, out_type,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x

        out = self.relu1(self.bn1(x))
        out = self.conv1(out)

        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return out + identity


class ResNet44_p4m(nn.Module):
    """
    ResNet‑44 with p4m equivariance for 1‑channel 28×28 Fashion‑MNIST.
    Channel widths  [6, 13, 24].
    """
    def __init__(self, n: int = 7, num_classes: int = 10):
        super().__init__()

        # ---------------- gspace & field helpers --------------------------
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)      # p4m (rot + flip)
        channels    = [6, 13, 24]                     # smaller widths

        # 1‑channel (grayscale) input
        self.in_type = enn.FieldType(
            self.r2_act,
            1 * [self.r2_act.trivial_repr]            # Fashion‑MNIST input
        )

        # ---------------- stem -------------------------------------------
        self.conv1_out = enn.FieldType(
            self.r2_act, channels[0] * [self.r2_act.regular_repr]
        )
        self.conv1 = enn.R2Conv(
            self.in_type, self.conv1_out,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # ---------------- stages -----------------------------------------
        self.stage1 = self._make_stage(self.conv1_out, channels[0], n, stride=1)
        self.stage2 = self._make_stage(self.stage1[-1].out_type, channels[1], n,
                                       stride=2)
        self.stage3 = self._make_stage(self.stage2[-1].out_type, channels[2], n,
                                       stride=2)

        # ---------------- head -------------------------------------------
        self.bn_final   = enn.InnerBatchNorm(self.stage3[-1].out_type)
        self.relu_final = enn.ReLU(self.stage3[-1].out_type)
        self.gpool      = enn.GroupPooling(self.stage3[-1].out_type)  # |G|-avg
        self.avgpool    = nn.AdaptiveAvgPool2d(1)                     # spatial
        self.fc         = nn.Linear(channels[2], num_classes)         # 24 → 10
        
        self.apply(self._init_weights)

    # ---------------- helper to build each residual stage ---------------
    def _make_stage(self, in_type, width, num_blocks, stride):
        out_type = enn.FieldType(self.r2_act,
                                 width * [self.r2_act.regular_repr])
        layers = [ResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_type, out_type, 3, stride=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            nn.init.zeros_(m.bias)

    # ---------------- forward -------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.relu_final(self.bn_final(x))
        x = self.gpool(x)             # (B, 24, H, W)
        x = self.avgpool(x.tensor)    # (B, 24, 1, 1)
        x = torch.flatten(x, 1)       # (B, 24)
        return self.fc(x)

"""# P4m-W ResNet44 model"""

class OrientationGate(nn.Module):
    def __init__(self, field_type: enn.FieldType):
        super().__init__()
        self.field_type = field_type

        # p4m → |G| = 8 (4 rotations × 2 flips)
        self.gsize = len(field_type.gspace.fibergroup.elements)
        # start at 0 ⇒ sigmoid(0)=0.5  
        self.alpha_logits = nn.Parameter(torch.zeros(self.gsize))

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        t = x.tensor                                           # (B, C, H, W)
        if t.shape[1] % self.gsize:                            
            return x                                           

        n_rep = t.shape[1] // self.gsize
        alpha = torch.sigmoid(self.alpha_logits)               
        alpha = alpha.view(1, 1, self.gsize, 1, 1)             

        t = t.view(t.shape[0], n_rep, self.gsize,
                   t.shape[2], t.shape[3]) * alpha             # gate
        t = t.view_as(x.tensor)
        return enn.GeometricTensor(t, self.field_type)

class PartialR2Conv(nn.Module):
    def __init__(self, in_type, out_type,
                 kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv  = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.gate  = OrientationGate(out_type)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return self.gate(self.conv(x))

class PartialResNetBlock(torch.nn.Module):
    r"""Pre‑activation residual basic block for p4m‑ResNet."""
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType,
                 kernel_size: int, stride: int = 1):
        super().__init__()

        self.in_type  = in_type
        self.out_type = out_type

        # --- main branch ---------------------------------------------------
        self.bn1   = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type)
        self.conv1 = PartialR2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.bn2   = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
        self.conv2 = PartialR2Conv(
            out_type, out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        # --- shortcut ------------------------------------------------------
        self.shortcut = None
        needs_proj = stride != 1 or in_type.size != out_type.size
        if needs_proj:
            self.shortcut = PartialR2Conv(
                in_type, out_type,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x

        out = self.relu1(self.bn1(x))
        out = self.conv1(out)

        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return out + identity


# ------------------------------------------------------------------
#  P4m-W ResNet‑44 for Fashion‑MNIST
# ------------------------------------------------------------------
class P4mWResNet44(nn.Module):
    """
    ResNet‑44 with learnable α‑gates for 1‑channel 28×28 Fashion‑MNIST.
    Widths [6, 13, 24].
    """
    def __init__(self, n: int = 7, num_classes: int = 10):
        super().__init__()

        # p4m group (4 rotations × 2 flips)
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)

        channels = [6, 13, 24]          

        # ---------- input field (grayscale) --------------------------
        self.in_type = enn.FieldType(
            self.r2_act,
            1 * [self.r2_act.trivial_repr]    # 1‑channel input
        )

        # ---------- stem --------------------------------------------
        self.conv1_out = enn.FieldType(
            self.r2_act, channels[0] * [self.r2_act.regular_repr]
        )
        self.conv1 = PartialR2Conv(
            self.in_type, self.conv1_out,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # ---------- residual stages ---------------------------------
        self.stage1 = self._make_stage(self.conv1_out, channels[0], n, stride=1)
        self.stage2 = self._make_stage(self.stage1[-1].out_type,
                                       channels[1], n, stride=2)
        self.stage3 = self._make_stage(self.stage2[-1].out_type,
                                       channels[2], n, stride=2)

        # ---------- head --------------------------------------------
        self.bn_final   = enn.InnerBatchNorm(self.stage3[-1].out_type)
        self.relu_final = enn.ReLU(self.stage3[-1].out_type)
        self.gpool      = enn.GroupPooling(self.stage3[-1].out_type)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Linear(channels[2], num_classes)   # 24 → 10

        self.apply(self._init_weights)

    # --------------------------------------------------------------
    def _make_stage(self, in_type, width, num_blocks, stride):
        out_type = enn.FieldType(self.r2_act,
                                 width * [self.r2_act.regular_repr])
        layers = [PartialResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(PartialResNetBlock(out_type, out_type, 3, stride=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            nn.init.zeros_(m.bias)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.relu_final(self.bn_final(x))
        x = self.gpool(x)              # (B, 24, H, W)
        x = self.avgpool(x.tensor)     # (B, 24, 1, 1)
        x = torch.flatten(x, 1)        # (B, 24)
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


def get_fashion_mnist(batch_size: int = 128, seed: int = 42):

    # ---------------- transforms -----------------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion‑MNIST mean / std
        transforms.RandomRotation(degrees=(-180, 180)),
    ])

    test_transform_no_rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion‑MNIST mean / std        
    ])

    test_transform_with_rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion‑MNIST mean / std
        transforms.RandomRotation(degrees=(-180, 180)),
    ])

    # Option C in code (example)
    full_train = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=train_transform)
    
    train_ds, val_ds = random_split(
        full_train, [55_000, 5_000],
        generator=torch.Generator().manual_seed(seed)
    )
    
    test_ds_no_rotation = datasets.FashionMNIST(root='./data', train=False,
                                    download=True, transform=test_transform_no_rotation)
    
    test_ds_with_rotation = datasets.FashionMNIST(root='./data', train=False,
                                    download=True, transform=test_transform_with_rotation)

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
print(device)

# Load data
train_loader, val_loader, test_loader_no_rotation, test_loader_with_rotation = get_fashion_mnist()

# Hyperparameters
hyper_parameters = {
  "LEARNING_RATE": 0.1,
  "WEIGHT_DECAY": 1e-9,
  "MOMENTUM": 0.9,
  "EPOCHS": 200,
  "SCHEDULER_PATIENCE": 10,
  "SCHEDULER_FACTOR": 0.5,
  "EARLY_STOPPING_PATIENCE": 30,
}

seeds   = [0, 1, 2, 3, 4]

models  = {"Z2": ResNet44_Z2,
           "P4m": ResNet44_p4m,
           "P4m-W": P4mWResNet44
          }

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
    
