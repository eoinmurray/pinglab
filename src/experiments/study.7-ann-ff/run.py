import sys
import json
from pathlib import Path
import shutil

import torch
from torch import nn
from torch.accelerator import current_accelerator, is_available
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from plots import (
    save_line,
)


device = current_accelerator().type if is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def main() -> None:
    experiment_dir = Path(__file__).parent.resolve()

    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    training_data = MNIST(
        root=experiment_dir / "data", 
        train=True, 
        download=True,
        transform=ToTensor()
    )

    test_data = MNIST(
        root=experiment_dir / "data", 
        train=False, 
        download=True,
        transform=ToTensor()
    )

    meta = spec.get("meta", {})
    batch_size = int(meta.get("batch_size", 64))

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = int(meta.get("epochs", 5))
    losses, accuracies = [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        loss, acc = test(test_dataloader, model, loss_fn)
        losses.append(loss)
        accuracies.append(acc)

    save_line(
        data_path / "loss",
        x=list(range(1, epochs+1)),
        y=losses,
        title="Test Loss",
        xlabel="Epoch",
        ylabel="Loss",
    )

    save_line(
        data_path / "accuracy",
        x=list(range(1, epochs+1)),
        y=[a * 100 for a in accuracies],
        title="Test Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
    )

    # ── save results ──────────────────────────────────────────────────────
    trainable_params = sum(p.numel() for p in model.parameters())
    results = {
        "epochs": epochs,
        "train_samples": len(training_data),
        "test_samples": len(test_data),
        "final_test_loss": round(losses[-1], 4),
        "best_test_loss": round(min(losses), 4),
        "best_test_loss_epoch": int(losses.index(min(losses)) + 1),
        "final_test_accuracy": round(100 * accuracies[-1], 1),
        "best_test_accuracy": round(100 * max(accuracies), 1),
        "best_test_accuracy_epoch": int(accuracies.index(max(accuracies)) + 1),
        "test_losses_per_epoch": [round(x, 4) for x in losses],
        "test_accuracies_per_epoch": [round(100 * x, 1) for x in accuracies],
        "trainable_params": trainable_params,
    }
    with open(data_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done!")

    torch.save(model.state_dict(), experiment_dir / "data" / "model.pth")
    print(f"Saved PyTorch Model State to {experiment_dir / 'data' / 'model.pth'}")

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(experiment_dir / "data" / "model.pth", weights_only=True))

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = pred[0].argmax(0).item(), y
        print(f'Predicted: {predicted}, Actual: {actual}')


if __name__ == "__main__":
    main()
