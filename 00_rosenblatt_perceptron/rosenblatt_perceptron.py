"""
Rosenblatt Perceptron — Animated Training Demo
----------------------------------------------

This script provides a minimal, from-scratch implementation of the classic
Rosenblatt Perceptron algorithm on a simple 2D, linearly separable dataset.
It is intended as an educational illustration of how the perceptron learns
a linear decision boundary over time.

Features
--------
1. Synthetic dataset generation:
   - Creates two Gaussian blobs in 2D (labels: -1 and +1).
   - Data roughly linearly separable with adjustable spread and margin.

2. Perceptron training loop:
   - Online (sample-by-sample) updates following Rosenblatt’s rule.
   - Tracks weights, bias, misclassification errors, perceptron loss,
     and classification accuracy across epochs.

3. Visualization:
   - Saves one PNG frame per epoch showing:
     * Scatter plot of training data.
     * Current decision boundary and shaded decision regions.
     * Training progress (errors, perceptron loss, accuracy curves).

4. Video creation:
   - Compiles saved frames into an MP4 (or GIF fallback) showing the evolution
     of the decision boundary as training progresses.

Usage
-----
Run from the command line with optional arguments:

    python3 rosenblatt_perceptron.py --epochs 100 --lr 0.5 --n-per-class 200

Or, you can also run from the command line using `make` (requires `make`) with preset defaults:

    make run_perceptron

Arguments include:
    --epochs          Number of training epochs (default: 50)
    --lr              Learning rate (default: 1.0)
    --n-per-class     Samples per class (default: 100)
    --seed            Random seed (default: 42)
    --spread          Standard deviation of each blob (default: 0.8)
    --margin          Distance between class centers (default: 2.0)
    --outdir          Directory for output frames (default: perceptron_frames)
    --clear-outdir    Clear output directory before saving new frames
    --fps             Frames per second for video (default: 12)
    --video           Output video filename (default: perceptron_training.mp4)
    --no-shuffle      Disable shuffling between epochs

Creating a video manually (alternative to built-in export):
    ffmpeg -r 30 -i perceptron_frames/frame_%04d.png -pix_fmt yuv420p perceptron_training.mp4

Dependencies
------------
- numpy
- matplotlib
- imageio (with ffmpeg support for MP4 export)

This script is intended for learning and demonstration purposes.
"""


import argparse
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Data generation
# --------------------------
def make_2d_blobs(n_per_class=100, seed=42, spread=0.8, margin=2.0):
    """
    Create two 2D Gaussian blobs, roughly linearly separable.
    y ∈ {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    mean_pos = np.array([ margin,  margin], dtype=float)
    mean_neg = np.array([-margin, -margin], dtype=float)
    cov = np.eye(2) * (spread ** 2)

    X_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class)
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class)
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_per_class, dtype=int), -np.ones(n_per_class, dtype=int)])
    return X, y

# --------------------------
# Perceptron (Rosenblatt)
# --------------------------
def perceptron_train(X, y, epochs=50, lr=1.0, shuffle=True, seed=42):
    """
    Classic Rosenblatt training loop.
    Returns model_history of (w, b, errors, perceptron_loss, accuracy).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    hist = {
        "w": [],
        "b": [],
        "errors": [],
        "perceptron_loss": [],
        "accuracy": [],
    }

    idx = np.arange(n)
    for ep in range(epochs):
        if shuffle:
            rng.shuffle(idx)

        errors = 0
        # Online updates
        for i in idx:
            s = y[i] * (np.dot(w, X[i]) + b)
            if s <= 0:
                # misclassified or on the boundary: update
                w += lr * y[i] * X[i]
                b += lr * y[i]
                errors += 1

        # Compute diagnostics AFTER the epoch's updates
        margins = y * (X @ w + b)
        p_loss = np.maximum(0.0, -margins).sum()
        acc = (margins > 0).mean()

        # Save snapshot
        hist["w"].append(w.copy())
        hist["b"].append(float(b))
        hist["errors"].append(int(errors))
        hist["perceptron_loss"].append(float(p_loss))
        hist["accuracy"].append(float(acc))

    return hist

# --------------------------
# Plotting helpers
# --------------------------
def compute_bounds(X, pad=1.0):
    xmin, ymin = X.min(axis=0) - pad
    xmax, ymax = X.max(axis=0) + pad
    return float(xmin), float(xmax), float(ymin), float(ymax)

def plot_epoch_frame(
    epoch, X, y, w, b, errors_hist, loss_hist, acc_hist, outdir,
    bounds=None, grid_step=300
):
    """
    Save a frame (PNG) showing:
    - Data points
    - Decision regions + boundary
    - Loss curves over epochs so far
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame_path = outdir / f"frame_{epoch:04d}.png"

    if bounds is None:
        xmin, xmax, ymin, ymax = compute_bounds(X, pad=1.0)
    else:
        xmin, xmax, ymin, ymax = bounds

    # Prepare mesh for decision regions
    xs = np.linspace(xmin, xmax, grid_step)
    ys = np.linspace(ymin, ymax, grid_step)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)

    # If w ~ 0 initially, prediction is ambiguous; handle gracefully
    if np.linalg.norm(w) < 1e-12:
        ZZ.fill(0.0)
    else:
        Z_lin = w[0] * XX + w[1] * YY + b
        ZZ = np.sign(Z_lin)

    # Figure layout
    fig = plt.figure(figsize=(11, 5), dpi=120)
    # Left: data & decision boundary
    ax1 = fig.add_subplot(1, 2, 1)

    # Decision regions (light shading)
    ax1.contourf(XX, YY, ZZ, levels=[-np.inf, 0, np.inf], alpha=0.15)

    # Scatter points
    cls_pos = y == 1
    cls_neg = ~cls_pos
    ax1.scatter(X[cls_pos, 0], X[cls_pos, 1], label="+1", marker="o")
    ax1.scatter(X[cls_neg, 0], X[cls_neg, 1], label="-1", marker="x")

    # Decision boundary line (w·x + b = 0)
    if abs(w[1]) > 1e-12:
        x_line = np.array([xmin, xmax])
        y_line = -(w[0] * x_line + b) / w[1]
        ax1.plot(x_line, y_line, linewidth=2)
    elif abs(w[0]) > 1e-12:
        # vertical line x = -b/w1
        x_vert = -b / w[0]
        ax1.plot([x_vert, x_vert], [ymin, ymax], linewidth=2)

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(f"Decision Boundary — Epoch {epoch+1}")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(loc="upper left")

    # Right: loss curves
    ax2 = fig.add_subplot(1, 2, 2)
    epochs_so_far = np.arange(1, epoch + 2)
    ax2.plot(epochs_so_far, errors_hist, label="Errors (0–1 loss)")
    ax2.plot(epochs_so_far, loss_hist, label="Perceptron loss")
    ax2.plot(epochs_so_far, acc_hist, label="Accuracy", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Training Progress")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # Suptitle with quick stats
    ax2_text = (
        f"Epoch {epoch+1} | Errors: {errors_hist[-1]} | "
        f"Perceptron loss: {loss_hist[-1]:.2f} | Acc: {acc_hist[-1]*100:.1f}%"
    )
    fig.suptitle(ax2_text, fontsize=11)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(frame_path)
    plt.close(fig)

    return frame_path

# --------------------------
# Video creation
# --------------------------
def create_video_from_frames(outdir, video_path, fps=12):
    """
    Tries to create an MP4 (requires ffmpeg via imageio-ffmpeg).
    Falls back to GIF if MP4 is not available.
    """
    outdir = Path(outdir)
    frames = sorted(outdir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames found to create a video.")

    try:
        import imageio.v3 as iio
        # Read into memory (safe for tens/hundreds of frames)
        imgs = [iio.imread(str(fp)) for fp in frames]
        iio.imwrite(str(video_path), imgs, fps=fps)  # MP4 if plugin available
        made = "mp4"
    except Exception as e_mp4:
        # Fall back to GIF
        try:
            import imageio
            imgs = [imageio.v2.imread(str(fp)) for fp in frames]
            gif_path = Path(video_path).with_suffix(".gif")
            imageio.mimsave(str(gif_path), imgs, duration=1.0 / fps)
            made = "gif"
            video_path = gif_path
        except Exception as e_gif:
            raise RuntimeError(
                f"Failed to create video. MP4 error: {e_mp4}; GIF error: {e_gif}"
            )
    return Path(video_path), made

# --------------------------
# Main / CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Rosenblatt Perceptron — Animated Training")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    p.add_argument("--n-per-class", type=int, default=100, help="Samples per class")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--spread", type=float, default=0.8, help="Gaussian std deviation")
    p.add_argument("--margin", type=float, default=2.0, help="Class centers distance")
    p.add_argument("--outdir", type=str, default="perceptron_frames", help="Frames dir")
    p.add_argument("--clear-outdir", action="store_true", help="Clear frames dir first")
    p.add_argument("--fps", type=int, default=12, help="Frames per second for video")
    p.add_argument("--video", type=str, default="perceptron_training.mp4", help="Output video filename (mp4/gif)")
    p.add_argument("--no-shuffle", action="store_true", help="Disable per-epoch shuffle")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)

    if args.clear_outdir and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    X, y = make_2d_blobs(
        n_per_class=args.n_per_class,
        seed=args.seed,
        spread=args.spread,
        margin=args.margin
    )
    bounds = compute_bounds(X, pad=1.0)

    # Train
    hist = perceptron_train(
        X, y, epochs=args.epochs, lr=args.lr,
        shuffle=not args.no_shuffle, seed=args.seed
    )

    # Frames per epoch
    errors_hist, loss_hist, acc_hist = [], [], []
    for ep in range(args.epochs):
        w = hist["w"][ep]
        b = hist["b"][ep]
        errors_hist.append(hist["errors"][ep])
        loss_hist.append(hist["perceptron_loss"][ep])
        acc_hist.append(hist["accuracy"][ep])
        plot_epoch_frame(
            epoch=ep, X=X, y=y, w=w, b=b,
            errors_hist=errors_hist,
            loss_hist=loss_hist,
            acc_hist=acc_hist,
            outdir=outdir,
            bounds=bounds
        )

    # Video
    video_path, kind = create_video_from_frames(outdir, Path(args.video), fps=args.fps)
    print(f"Saved {len(list(outdir.glob('frame_*.png')))} frames in: {outdir}")
    print(f"Created {kind.upper()} video: {video_path}")

if __name__ == "__main__":
    main()
