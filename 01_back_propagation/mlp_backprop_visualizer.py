"""
MLP Backpropagation Visualizer
-------------------------------

A minimal, from-scratch implementation of a multilayer perceptron to illustrate:
1. Random weights initialization
2. Forward pass and error propagation (backpropagation)
3. Gradient descent weight update

Outputs a PNG frame per epoch showing:
- Loss curve (so far)
- Decision boundary (sigmoid output) with training data points
- Heatmaps / bars for W1, b1, W2, b2
- Average |dL/dx| for each input feature (how error propagates "back" to the input)

Make a video (example using ffmpeg) from the PNGs:
ffmpeg -r 30 -i frames_xor/frame_%04d.png -pix_fmt yuv420p xor_training.mp4
"""

from typing import Tuple, Any, Generator, Dict, List

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy import floating


# -----------------------
# Utilities functions
# ----------------------

def sigmoid(
    z: np.ndarray,
) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def dsigmoid_from_output(
        a: np.ndarray,
) -> np.ndarray:
    return a * (1 - a)


def tanh(
        z: np.ndarray,
) -> np.ndarray:
    return np.tanh(z)

def dtanh(
        z: np.ndarray,
)-> np.ndarray:
    t: np.ndarray = tanh(z)
    return 1.0 - t**2

def make_xor_dataset(

) -> Tuple[np.ndarray, np.ndarray]:

    X: np.ndarray = np.array([
          [-1.0, -1.0]
        , [-1.0, +1.0]
        , [+1.0, -1.0]
        , [+1.0, +1.0]
    ])
    y: np.ndarray = np.array([
          [0.0]
        , [1.0]
        , [1.0]
        , [0.0]
    ])

    return X, y

def decision_boundary(
        model_forward: Any,
        grid_lim: float = 1.5,
        steps: int = 200
) -> Tuple[np.meshgrid, np.meshgrid, np.meshgrid]:

    xs: np.ndarray = np.linspace(-grid_lim, grid_lim, steps)
    ys: np.ndarray = np.linspace(-grid_lim, grid_lim, steps)
    xx, yy = np.meshgrid(xs, ys)
    flat = np.c_[xx.ravel(), yy.ravel()]
    preds, _, _, _ = model_forward(flat)
    zz = preds.reshape(xx.shape)

    return xx, yy, zz

def ensure_outdir(
        path: Path,
) -> None:
    if not os.path.exists(path):
        path.mkdir(parents=True)


# ------------------------------------
# Model + Training (from scratch)
# -----------------------------------

class TinyMLP:

    def __init__(
            self,
            input_dim: int = 2,
            hidden_dim: int = 3,
            output_dim: int = 1,
            seed: int = 42,
    ) -> None:
        rng: Generator = np.random.default_rng(seed)

        # Random Weight Seeding
        self.W1: np.ndarray = rng.normal(0, 1.0, size=(input_dim, hidden_dim))
        self.b1: np.ndarray = rng.normal(0, 1.0, size=(1, hidden_dim))
        self.W2: np.ndarray = rng.normal(0, 1.0, size=(hidden_dim, output_dim))
        self.b2: np.ndarray = rng.normal(0, 1.0, size=(1, output_dim))


    def forward(
            self,
            X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through the model
        Z1 = X•W1 + b1
        A1 = tanh(Z1)
        Z2 = A1•W2 + b2
        A2 = sigmoid(Z2)
        """
        Z1: np.ndarray = np.dot(X, self.W1) + self.b1
        A1: np.ndarray = tanh(Z1)
        Z2: np.ndarray = np.dot(A1, self.W2) + self.b2
        A2: np.ndarray = sigmoid(Z2)
        return A2, Z1, A1, Z2


    def loss_bce(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
    ) -> floating[Any]:
        """
        Binary Cross Entropy Loss
        Using the binary cross entropy loss function because we only have two classes (0, 1)

        :param y_pred: Predicted probabilities
        :param y_true: True labels
        :return: Loss value
        """
        eps: float = 1e-12
        y_pred_clipped: np.ndarray = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))


    def backprop(
            self,
            X: np.ndarray,
            y_true: np.ndarray,
            cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict:
        """
        Backpropagation for BCE + sigmoid output

        For BCE with sigmoid, dL/dz2 = (A2 - y_true) / N

        :param X:
        :param y_true:
        :param cache:
        :return:
        """

        z1, a1, z2, a2 = cache
        N = X.shape[0]

        # Output layer gradient
        dz2 = (a2 - y_true) / N
        dw2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradient
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * dtanh(z1)
        dw1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # For illustrating `error propagation toward inputs`
        # dL/dx = dz1 • W1^T (chain rule back to inputs)
        dLdX = dz1.dot(self.W1.T)
        avg_dL_dx = np.mean(np.abs(dLdX), axis=0)

        return {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2,
            "avg_dL_dx": avg_dL_dx,
        }

    def step(
            self,
            grads: Dict,
             learning_rate: float = 0.01,
    ) -> None:
        """
        Update weights with gradient descent
        """

        self.W1 -= learning_rate * grads["dw1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dw2"]
        self.b2 -= learning_rate * grads["db2"]


# ----------------------------------
# Plotting per-epoch frame
# ----------------------------------

def plot_frame (
    epoch: int,
    losses: List[float],
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    outdir: Path,
    seed_used: int,
    grid_lim: float = 1.5,
) -> None:

    # Prepare decision boundary
    def forward_only(
            input_data
    ) -> Tuple:
        return model.forward(input_data)[0], *model.forward(input_data)[1:]

    xx, yy, zz = decision_boundary(forward_only, grid_lim=grid_lim, steps=200)

    # Forward on training points for overlay
    yhat, _, _, _ = model.forward(X)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Epoch {epoch} - Seed: {seed_used}", fontsize=14)

    # 1) Loss curve
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.plot(np.arange(len(losses)), losses)
    ax1.set_title("Loss (BCE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linewidth=0.2)

    # 2) Decision boundary
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    im = ax2.imshow(
        zz,
        origin="lower",
        extent=(-grid_lim, grid_lim, -grid_lim, grid_lim),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax2.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors="k")
    ax2.set_title("Decision Boundary (Sigmoid Output)")
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("P(y=1)")

    # 3) W1 heatmap
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    imW1 = ax3.imshow(model.W1, aspect='auto')
    ax3.set_title("W1 (2×3)")
    ax3.set_xlabel("hidden units")
    ax3.set_ylabel("inputs")
    plt.colorbar(imW1, ax=ax3)

    # 4) b1 bar
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    ax4.bar(np.arange(model.b1.shape[1]), model.b1.ravel())
    ax4.set_title("b1 (length 3)")
    ax4.set_xlabel("hidden units")
    ax4.set_ylabel("value")
    ax4.grid(True, linewidth=0.3)

    # 5) W2 heatmap
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    imW2 = ax5.imshow(model.W2, aspect='auto')
    ax5.set_title("W2 (3×1)")
    ax5.set_xlabel("output unit")
    ax5.set_ylabel("hidden units")
    plt.colorbar(imW2, ax=ax5)

    # 6) Error propagation to inputs: avg |dL/dx|
    # Compute it for this epoch to show how gradients "flow back" to the inputs.
    a2, z1, a1, z2 = model.forward(X)
    grads = model.backprop(X, y, (a2, z1, a1, z2))
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    ax6.bar([0, 1], grads['avg_dL_dx'])
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['|∂L/∂x1|', '|∂L/∂x2|'])
    ax6.set_title("Backprop to Inputs (avg |gradient|)")
    ax6.grid(True, linewidth=0.3)

    # Annotate current loss
    if len(losses):
        fig.text(0.02, 0.02, f"Current loss: {losses[-1]:.4f}", fontsize=11)

    outpath = outdir / f"frame_{epoch:04d}.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# -------------------------
# Main routine
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for gradient descent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--out", type=str, default="frames_xor", help="Output folder for frames")
    parser.add_argument("--grid_lim", type=float, default=1.5, help="Decision boundary plot extent")
    args = parser.parse_args()

    outdir = Path(args.out)
    ensure_outdir(outdir)

    # Data
    X, y = make_xor_dataset()

    # Model
    model = TinyMLP(input_dim=2, hidden_dim=3, output_dim=1, seed=args.seed)

    # Illustrate random initialization by saving epoch 0 frame before any training
    losses: List[float] = []
    plot_frame(epoch=0, losses=losses, X=X, y=y, model=model, outdir=outdir, seed_used=args.seed, grid_lim=args.grid_lim)

    # Training loop (full-batch gradient descent)
    for epoch in range(1, args.epochs + 1):
        # Forward
        y_pred, z1, a1, z2 = model.forward(X)
        loss = model.loss_bce(y_pred, y)
        losses.append(loss)

        # Backprop
        grads = model.backprop(X, y, (y_pred, z1, a1, z2))

        # Update (GD)
        model.step(grads, learning_rate=args.lr)

        # Save a frame each epoch
        plot_frame(epoch=epoch, losses=losses, X=X, y=y, model=model, outdir=outdir, seed_used=args.seed, grid_lim=args.grid_lim)

    # Final hints for the student
    print("Frames saved to:", str(outdir.resolve()))
    print("\nCreate a video (1 fps) with ffmpeg:")
    print(f"ffmpeg -r 1 -i {args.out}/frame_%04d.png -pix_fmt yuv420p xor_training.mp4")

if __name__ == "__main__":
    main()