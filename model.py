"""
Self-Pruning Neural Network on CIFAR-10

Implements a feed-forward neural network that learns to prune itself during
training using learnable gate parameters and L1 sparsity regularisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ── Part 1: PrunableLinear Layer ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """Custom linear layer with learnable gate parameters for self-pruning.

    Each weight has a corresponding gate_score. During the forward pass:
      1. gates = sigmoid(gate_scores)           -> values in (0, 1)
      2. pruned_weights = weight * gates         -> element-wise gating
      3. output = x @ pruned_weights.T + bias    -> linear transformation

    Gradients flow through both weight and gate_scores via autograd.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        """Return gate values detached from the computation graph."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)


# ── Network Architecture ────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """Feed-forward classifier for CIFAR-10 using PrunableLinear layers.

    Architecture:
      Flatten(3072) -> PrunableLinear(512) -> BN -> ReLU
                    -> PrunableLinear(256) -> BN -> ReLU
                    -> PrunableLinear(10)  -> logits
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_all_gates(self):
        """Concatenate all gate values into one flat CPU tensor."""
        return torch.cat([l.get_gates().cpu().flatten()
                          for l in [self.fc1, self.fc2, self.fc3]])

    def sparsity_loss(self):
        """L1 sparsity loss: sum of all gate values across all layers."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates = torch.sigmoid(layer.gate_scores)
            loss = loss + gates.sum()
        return loss

    def overall_sparsity(self, threshold=2.5e-2):
        """Percentage of gates below threshold across all layers."""
        all_gates = self.get_all_gates()
        return (all_gates < threshold).sum().item() / all_gates.numel() * 100.0

    def per_layer_sparsity(self, threshold=2.5e-2):
        """Return sparsity info for each PrunableLinear layer."""
        info = {}
        for name, layer in [("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)]:
            gates = layer.get_gates()
            total = gates.numel()
            pruned = (gates < threshold).sum().item()
            info[name] = {
                "total": total, "pruned": int(pruned),
                "pct": pruned / total * 100.0,
                "min": gates.min().item(), "max": gates.max().item(),
            }
        return info


# ── Data Loading ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 train/test loaders with standard augmentation."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available())

    return train_loader, test_loader


# ── Part 3A: Training Loop ──────────────────────────────────────────────────

def train(model, train_loader, optimizer, scheduler, lambda_sparse,
          epochs, device, warmup_epochs=10):
    """Train with Total Loss = CE + lambda_eff * SparsityLoss.

    Lambda warmup: lambda_eff linearly ramps from 0 to lambda_sparse over the
    first warmup_epochs, letting the network learn which weights matter before
    the sparsity pressure kicks in.
    """
    criterion = nn.CrossEntropyLoss()
    history = {"ce_loss": [], "sparsity_loss": [], "total_loss": [],
               "train_acc": [], "sparsity_level": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_ce, running_sp, correct, total = 0.0, 0.0, 0, 0

        effective_lambda = lambda_sparse * min(1.0, epoch / warmup_epochs) \
            if warmup_epochs > 0 else lambda_sparse

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            class_loss = criterion(outputs, labels)
            sparse_loss = model.sparsity_loss()
            loss = class_loss + effective_lambda * sparse_loss

            loss.backward()
            optimizer.step()

            running_ce += class_loss.item()
            running_sp += sparse_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        n = len(train_loader)
        avg_ce, avg_sp = running_ce / n, running_sp / n
        train_acc = 100.0 * correct / total
        sparsity = model.overall_sparsity()

        history["ce_loss"].append(avg_ce)
        history["sparsity_loss"].append(avg_sp)
        history["total_loss"].append(avg_ce + effective_lambda * avg_sp)
        history["train_acc"].append(train_acc)
        history["sparsity_level"].append(sparsity)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            gates = model.get_all_gates()
            print(f"  Epoch [{epoch:2d}/{epochs}]  "
                  f"CE: {avg_ce:.4f}  |  "
                  f"λ_eff: {effective_lambda:.2e}  |  "
                  f"Acc: {train_acc:.2f}%  |  "
                  f"Sparsity: {sparsity:.1f}%  |  "
                  f"Gates [min={gates.min():.4f}, mean={gates.mean():.4f}, "
                  f"max={gates.max():.4f}]")

    return history


# ── Part 3B: Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_loader, device):
    """Return test accuracy (%)."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = model(images).max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_gate_distribution(results, save_path="gate_distribution.png"):
    """Histogram of gate values for each lambda experiment."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        gates = r["model"].get_all_gates().numpy()
        ax.hist(gates, bins=100, color="steelblue", edgecolor="white",
                alpha=0.85, linewidth=0.5)
        ax.axvline(x=0.025, color="red", linestyle="--", linewidth=1.5,
                   label="Prune threshold (0.025)")
        ax.set_title(f"λ = {r['lambda']}\n"
                     f"Acc = {r['accuracy']:.1f}%  |  "
                     f"Sparsity = {r['sparsity']:.1f}%", fontsize=12)
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle("Gate Value Distributions — Self-Pruning Network",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {save_path}")


def plot_training_curves(all_histories, lambda_values,
                         save_path="training_curves.png"):
    """Training curves for all lambda experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (lam, h) in enumerate(zip(lambda_values, all_histories)):
        label = f"λ = {lam}"
        ep = range(1, len(h["ce_loss"]) + 1)
        c = colors[i % len(colors)]
        axes[0, 0].plot(ep, h["ce_loss"],        color=c, label=label, lw=2)
        axes[0, 1].plot(ep, h["sparsity_level"], color=c, label=label, lw=2)
        axes[1, 0].plot(ep, h["train_acc"],      color=c, label=label, lw=2)
        axes[1, 1].plot(ep, h["total_loss"],     color=c, label=label, lw=2)

    for ax, title, yl in [
        (axes[0, 0], "Cross-Entropy Loss", "Loss"),
        (axes[0, 1], "Sparsity Level",     "Sparsity (%)"),
        (axes[1, 0], "Training Accuracy",  "Accuracy (%)"),
        (axes[1, 1], "Total Loss",         "Loss"),
    ]:
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves Across λ Values",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    LAMBDA_VALUES = [1e-5, 5e-5, 2e-4]
    NUM_EPOCHS    = 35
    WARMUP_EPOCHS = 10
    LEARNING_RATE = 1e-3
    BATCH_SIZE    = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Self-Pruning Neural Network — CIFAR-10")
    print(f"  Device           : {device}")
    print(f"  Lambda values    : {LAMBDA_VALUES}")
    print(f"  Epochs           : {NUM_EPOCHS}")
    print(f"  Warmup epochs    : {WARMUP_EPOCHS}")
    print("=" * 65)

    print("\nLoading CIFAR-10 dataset …")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    print("Dataset ready.\n")

    results, all_histories = [], []

    for lam in LAMBDA_VALUES:
        print(f"\n{'═' * 65}")
        print(f"  EXPERIMENT  —  λ = {lam}")
        print(f"{'═' * 65}")

        model = SelfPruningNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        history = train(model, train_loader, optimizer, scheduler,
                        lam, NUM_EPOCHS, device, WARMUP_EPOCHS)
        all_histories.append(history)

        test_acc = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity()

        print(f"\n  ✓ Test Accuracy   : {test_acc:.2f}%")
        print(f"  ✓ Sparsity Level  : {sparsity:.2f}%")

        for name, info in model.per_layer_sparsity().items():
            print(f"     {name}: {info['pct']:6.2f}%  "
                  f"({info['pruned']:,}/{info['total']:,})  "
                  f"gate range [{info['min']:.4f}, {info['max']:.4f}]")

        results.append({"lambda": lam, "accuracy": test_acc,
                         "sparsity": sparsity, "model": model})

    # Summary
    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")
    print(f"  {'Lambda':<12} {'Test Accuracy':>15}  {'Sparsity (%)':>14}")
    print(f"  {'─' * 44}")
    for r in results:
        print(f"  {r['lambda']:<12.1e} {r['accuracy']:>14.2f}%  "
              f"{r['sparsity']:>13.2f}%")
    print(f"{'═' * 65}")

    # Plots
    print("\nGenerating plots …")
    plot_gate_distribution(results)
    plot_training_curves(all_histories, LAMBDA_VALUES)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()