

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
from e2cnn import gspaces
from e2cnn import nn as enn
from e2cnn.nn import FieldType, GeometricTensor

"""# Standard ResNet44"""

# --- Residual Block (Z2) ---
class BasicBlockZ2PreAct(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut_conv = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(x))
        
        if self.shortcut_conv is not None:
            projected_identity = self.shortcut_conv(x)
        else:
            projected_identity = identity


        out = self.conv1(out)

        out = self.relu2(self.bn2(out)) 
        out = self.conv2(out) 

        out += projected_identity        

        return out

# --- Standard ResNet44 Model ---
class ResNet44_Z2(nn.Module):
    def __init__(self, block=BasicBlockZ2PreAct, num_blocks_per_stage=[7, 7, 7], num_classes=10, input_channels=3):
        super().__init__()
        self.in_channels_next_block = 32 # Starting number of channels for stages
        
        self.conv1 = nn.Conv2d(input_channels, self.in_channels_next_block, kernel_size=3, stride=1, padding=1, bias=False)        

        # Residual Stages
        self.stage1 = self._make_stage(block, 32, num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(block, 64, num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(block, 128, num_blocks_per_stage[2], stride=2)
        
        self.bn_final = nn.BatchNorm2d(128 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        self.apply(self._init_weights)

    def _make_stage(self, block, out_channels_stage, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels_next_block, out_channels_stage, stride=s))
            self.in_channels_next_block = out_channels_stage * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x) 

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = self.relu_final(self.bn_final(out)) 

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

"""# ResNet44_P4M"""

# --- ResNetBlock ---
class ResNetBlock(nn.Module):
    r"""Pre‑activation residual basic block for p4m‑ResNet."""
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType,
                 kernel_size: int, stride: int = 1):
        super().__init__()

        self.in_type  = in_type
        self.out_type = out_type

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

        self.shortcut = None
        needs_proj = stride != 1 or in_type.size != out_type.size
        if needs_proj:
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


# --- ResNet44_p4m_e2cnn Model ---
class ResNet44_p4m_e2cnn(torch.nn.Module):
    r"""ResNet‑44 with p4m equivariance."""
    def __init__(self, n: int = 7, num_classes: int = 10):
        super().__init__()

        self.r2_act = gspaces.FlipRot2dOnR2(N=4)        
        channels    = [11, 23, 45]

        self.in_type = enn.FieldType(
            self.r2_act,
            3 * [self.r2_act.trivial_repr]
        )

        self.conv1_out = enn.FieldType(
            self.r2_act, channels[0] * [self.r2_act.regular_repr]
        )
        self.conv1 = enn.R2Conv(
            self.in_type, self.conv1_out,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.stage1 = self._make_stage(
            self.conv1_out,
            channels[0], n, stride=1
        )
        self.stage2 = self._make_stage(
            self.stage1[-1].out_type,
            channels[1], n, stride=2
        )
        self.stage3 = self._make_stage(
            self.stage2[-1].out_type,
            channels[2], n, stride=2
        )

        self.bn_final   = enn.InnerBatchNorm(self.stage3[-1].out_type)
        self.relu_final = enn.ReLU(self.stage3[-1].out_type)
        self.gpool      = enn.GroupPooling(self.stage3[-1].out_type)
        self.avgpool    = torch.nn.AdaptiveAvgPool2d(1)
        self.fc         = torch.nn.Linear(channels[2], num_classes) 

        self.apply(self._init_weights)

    def _make_stage(self, in_type: enn.FieldType, width: int,
                    num_blocks: int, stride: int):
        out_type = enn.FieldType(
            self.r2_act, width * [self.r2_act.regular_repr]
        )
        layers = [ResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_type, out_type, 3, stride=1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                 torch.nn.init.zeros_(m.bias)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)

        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.relu_final(self.bn_final(x))

        x = self.gpool(x)
        x = self.avgpool(x.tensor)
        x = torch.flatten(x, 1)

        return self.fc(x)

"""# Partial RestNet44 P4M model"""

# --- OrientationGate ---
class OrientationGate(nn.Module):
    def __init__(self, field_type: enn.FieldType):
        super().__init__()
        self.field_type = field_type
        if not field_type.representations:
            self.gsize = 0
            self.alpha_logits = None
            return

        self.gsize = len(field_type.gspace.fibergroup.elements)
        if self.gsize > 0 : 
             self.alpha_logits = nn.Parameter(torch.zeros(self.gsize))
        else: 
            self.alpha_logits = None


    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        if self.alpha_logits is None or self.gsize == 0:
            return x

        t = x.tensor
        
        # This also implicitly checks if t.shape[1] is non-zero before modulo
        if t.shape[1] == 0 or t.shape[1] % self.gsize:
            return x

        n_rep = t.shape[1] // self.gsize
        alpha = torch.sigmoid(self.alpha_logits)
        alpha = alpha.view(1, 1, self.gsize, 1, 1)

        t_reshaped = t.view(t.shape[0], n_rep, self.gsize,
                   t.shape[2], t.shape[3])

        t_gated = t_reshaped * alpha
        t_final = t_gated.reshape_as(x.tensor) 
        return enn.GeometricTensor(t_final, self.field_type)

# --- PartialR2Conv ---
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

# --- PartialResNetBlock ---
class PartialResNetBlock(torch.nn.Module):
    r"""Pre‑activation residual basic block for p4m‑ResNet."""
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType,
                 kernel_size: int, stride: int = 1):
        super().__init__()

        self.in_type  = in_type
        self.out_type = out_type

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

# --- P4mW_ResNet44 Model ---
class P4mW_ResNet44(torch.nn.Module):
    r"""ResNet‑44 with P4m-W equivariance."""
    def __init__(self, n: int = 7, num_classes: int = 10):
        super().__init__()

        self.r2_act = gspaces.FlipRot2dOnR2(N=4)        
        channels    = [11, 23, 45]

        self.in_type = enn.FieldType(
            self.r2_act,
            3 * [self.r2_act.trivial_repr]
        )

        self.conv1_out = enn.FieldType(
            self.r2_act, channels[0] * [self.r2_act.regular_repr]
        )
        self.conv1 = PartialR2Conv(
            self.in_type, self.conv1_out,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.stage1 = self._make_stage(
            self.conv1_out,
            channels[0], n, stride=1
        )
        self.stage2 = self._make_stage(
            self.stage1[-1].out_type,
            channels[1], n, stride=2
        )
        self.stage3 = self._make_stage(
            self.stage2[-1].out_type,
            channels[2], n, stride=2
        )

        self.bn_final   = enn.InnerBatchNorm(self.stage3[-1].out_type)
        self.relu_final = enn.ReLU(self.stage3[-1].out_type)
        self.gpool      = enn.GroupPooling(self.stage3[-1].out_type)
        self.avgpool    = torch.nn.AdaptiveAvgPool2d(1)
        self.fc         = torch.nn.Linear(channels[2], num_classes) 

        self.apply(self._init_weights)

    def _make_stage(self, in_type: enn.FieldType, width: int,
                    num_blocks: int, stride: int):
        out_type = enn.FieldType(
            self.r2_act, width * [self.r2_act.regular_repr]
        )
        layers = [PartialResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(PartialResNetBlock(out_type, out_type, 3, stride=1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                 torch.nn.init.zeros_(m.bias)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)

        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.relu_final(self.bn_final(x))

        x = self.gpool(x)
        x = self.avgpool(x.tensor)
        x = torch.flatten(x, 1)

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

"""# Data loading"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_rotated_cifar10(batch_size=128, seed=42):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load the full training set
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split into 40k train + 10k val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [40000, 10000], generator=generator)

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

"""# Train function"""

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

"""# Testing function"""

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

"""# Training loop"""

# Training loop
def training_loop(model, device, train_loader, test_loader, val_loader, hyper_parameters):
    # --- Training Loop ---

    MODEL_NAME = type(model).__name__
    print(f"Training {MODEL_NAME} Model...")
    print(f"Total parameters: {count_parameters(model):,}")

    LEARNING_RATE = hyper_parameters["LEARNING_RATE"]
    WEIGHT_DECAY = hyper_parameters["WEIGHT_DECAY"]
    MOMENTUM = hyper_parameters["MOMENTUM"]
    EPOCHS = hyper_parameters["EPOCHS"]
    SCHEDULER_PATIENCE = hyper_parameters["SCHEDULER_PATIENCE"]
    SCHEDULER_FACTOR = hyper_parameters["SCHEDULER_FACTOR"]
    EARLY_STOPPING_PATIENCE = hyper_parameters["EARLY_STOPPING_PATIENCE"]
    SEED = hyper_parameters["SEED"]
    BEST_MODEL_PATH = "./models/" + MODEL_NAME + "_best_tuned.pth"

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

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
        test_accuracy = test(model, device, test_loader, "Test set")
        print(f"\nFinal Test Accuracy (Best {MODEL_NAME} Model): {test_accuracy:.2f}%")
    except FileNotFoundError:
        print("Best model file not found. Run training first.")
    except Exception as e:
         print(f"An error occurred loading the best model: {e}")
    
    return test_accuracy

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load data
train_loader, val_loader, test_loader = get_rotated_cifar10()

# Hyperparameters
hyper_parameters = {
  "LEARNING_RATE": 0.05,
  "WEIGHT_DECAY": 1e-4,
  "MOMENTUM": 0.9,
  "EPOCHS": 200,
  "SCHEDULER_PATIENCE": 10,
  "SCHEDULER_FACTOR": 0.5,
  "EARLY_STOPPING_PATIENCE": 30,
  "SEED": 42,  
}

seeds   = [0, 1, 2, 3, 4]

models  = {"Z2": ResNet44_Z2,
           "P4m": ResNet44_p4m_e2cnn,
           "P4m-W": P4mW_ResNet44}

# --------------- results dict ------------------------------------- #
results = {name: [] for name in models}
ece_results = {name: [] for name in models}

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
        best_acc = training_loop(model, device, train_loader, test_loader, val_loader, hyper_parameters)

        results[name].append(best_acc)
        print(f"seed {seed} \t{name}\t{best_acc:.2f}%")
        ece = compute_ece(model, test_loader, device="cuda", n_bins=15)
        ece_results[name].append(ece)
        print(f"Expected Calibration Error: {ece:.4f}")


# ----------------- summary ------------------ #
for name, scores in results.items():
    mean = np.mean(scores)
    std  = np.std(scores)
    print(f"{name:8s}: {mean:.2f} ± {std:.2f}  (n={len(scores)})")


print("\n------- ECE Results ------------")
for name, ece in ece_results.items():
    mean = np.mean(ece)
    print(f"{name:8s}: {mean}")