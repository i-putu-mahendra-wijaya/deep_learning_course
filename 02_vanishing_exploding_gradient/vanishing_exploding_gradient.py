"""
vanishing_exploding_gradient.py

--------------------------------------------------------------------------------------

Demonstrates vanishing vs exploding gradients during deep learning training.

What it does:
- Builds TWO deep MLPs trained on the same synthetic 2D classification task:
  (A) Vanishing-prone: many layers + sigmoid.
  (B) Exploding-prone: many layers + ReLU, large init, and extra per-layer scaling.

- Tracks for every epoch:
  * Loss (train)
  * Per-node gradient magnitudes w.r.t. pre-activations z (dL/dz) for each hidden layer.
    (We aggregate across the batch as mean |grad| per node.)

- Saves a frame per epoch with:
  * Loss curves (both models)
  * Heatmaps (epochs × nodes) of log10(|grad|) for 3 representative layers
  * A histogram of |grad| at the current epoch
  * % nodes considered "vanished" (< VANISH_THR) or "exploded" (> EXPLODE_THR)

- Combines frames to MP4 via ffmpeg.

Run:
  python vanishing_exploding_gradient.py

Optional tweaks at the CONFIG section.
"""

import os
import math
import shutil
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
SEED               = 7
N_SAMPLES          = 4000
EPOCHS             = 60
BATCH_SIZE         = 128
DEPTH              = 10        # number of hidden layers
WIDTH              = 128       # units per hidden layer
INIT_STD_VANISH    = 0.1       # weight init std for vanishing model (sigmoid)
INIT_STD_EXPLODE   = 1.0       # weight init std for exploding model (relu)
SCALE_PER_LAYER    = 1.05       # multiply pre-activation z by this per layer (exploding model)
VANISHING_LR                 = 1e-4
EXPLODING_LR                 = 1e-4
FRAME_DIR          = "frames_vegd"
VIDEO_OUT          = "vanishing_exploding_gradient.mp4"

# Thresholds to tag "vanished" / "exploded" per-node gradients
VANISH_THR         = 1e-6
EXPLODE_THR        = 1.0

# Heatmap display: we plot log10(|grad|); clamp to this range for consistent colors
LOG_MIN, LOG_MAX   = -12.0, 2.0

# Layers to visualize in heatmaps (0 = first, -1 = last hidden, middle layer as well)
def chosen_layers(depth):
    return [0, depth // 2, depth - 1]

# -----------------------
# Utils
# -----------------------
rng = np.random.RandomState(SEED)
tf.random.set_seed(SEED)

def make_toy_data(n=N_SAMPLES):
    """Two Gaussian blobs, linearly separable-ish but with variance."""
    n2 = n // 2
    c0 = rng.normal(loc=[-1.5, 0.0], scale=0.35, size=(n2, 2)).astype(np.float32)
    c1 = rng.normal(loc=[+1.5, 0.0], scale=0.35, size=(n - n2, 2)).astype(np.float32)
    X = np.vstack([c0, c1]).astype(np.float32)
    y = np.array([0]*n2 + [1]*(n - n2), dtype=np.int32)
    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    # y_oh = tf.one_hot(y, depth=2)
    y_oh = np.eye(2, dtype=np.float32)[y]
    return X, y_oh

def batches(X, Y, batch_size):
    n = X.shape[0]
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        j = idx[start:start+batch_size]
        yield tf.convert_to_tensor(X[j]), tf.convert_to_tensor(Y[j])

# -----------------------
# Manual deep MLP (weights as tf.Variables so we can track z and grads easily)
# -----------------------
class DeepMLP:
    def __init__(self, in_dim, out_dim, depth, width, activation, init_std=0.1, scale_per_layer=1.0, name="model"):
        self.name = name
        self.depth = depth
        self.width = width
        self.activation = activation
        self.scale_per_layer = scale_per_layer

        self.W = []
        self.b = []
        prev = in_dim
        for l in range(depth):
            W = tf.Variable(tf.random.normal([prev, width], stddev=init_std, seed=SEED+l), name=f"{name}_W{l}")
            b = tf.Variable(tf.zeros([width]), name=f"{name}_b{l}")
            self.W.append(W)
            self.b.append(b)
            prev = width
        self.Wo = tf.Variable(tf.random.normal([prev, out_dim], stddev=init_std, seed=SEED+999), name=f"{name}_Wout")
        self.bo = tf.Variable(tf.zeros([out_dim]), name=f"{name}_bout")

        self.params = self.W + self.b + [self.Wo, self.bo]

    def forward(self, x):
        """Returns (logits, z_list) where z_list has pre-activations for hidden layers."""
        z_list = []
        a = x
        for l in range(self.depth):
            z = tf.matmul(a, self.W[l]) + self.b[l]
            if self.scale_per_layer != 1.0:
                z = z * self.scale_per_layer
            z_list.append(z)
            if self.activation == "sigmoid":
                a = tf.nn.sigmoid(z)
            elif self.activation == "tanh":
                a = tf.nn.tanh(z)
            elif self.activation == "relu":
                a = tf.nn.relu(z)
            else:
                raise ValueError("Unsupported activation")
        logits = tf.matmul(a, self.Wo) + self.bo
        return logits, z_list

# -----------------------
# Training + gradient tracking
# -----------------------
@tf.function(reduce_retracing=True)
def batch_step(model, xb, yb, optimizer):
    """One optimizer step + collect mean |dL/dz| per node for each hidden layer on this batch."""
    with tf.GradientTape(persistent=True) as tape:
        logits, z_list = model.forward(xb)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yb, logits=logits))

    # --- parameter gradients (apply update) ---
    grads = tape.gradient(loss, model.params)

    # sanitize + clip to avoid NaN/Inf blow-ups
    safe_grads = []
    for g in grads:
        if g is None:
            safe_grads.append(None)
        else:
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            safe_grads.append(g)
    safe_grads, _ = tf.clip_by_global_norm([g for g in safe_grads if g is not None], 5.0)

    # re-zip with params, preserving None if any
    zipped = []
    it = iter(safe_grads)
    for p, g in zip(model.params, grads):
        if g is None:
            zipped.append((None, p))  # will be skipped by apply_gradients
        else:
            zipped.append((next(it), p))
    optimizer.apply_gradients([(g, p) for g, p in zipped if g is not None])

    # --- grads wrt pre-activations z for visualization ---
    grad_z_list = tape.gradient(loss, z_list)
    del tape

    per_layer_node_means = []
    for gz in grad_z_list:
        # replace non-finite with huge-but-finite cap for stats, then take mean |.| over batch
        gz = tf.where(tf.math.is_finite(gz), gz, tf.zeros_like(gz))
        g = tf.reduce_mean(tf.abs(gz), axis=0)  # shape: (width,)
        per_layer_node_means.append(g)
    return loss, per_layer_node_means


def train_one_epoch(model, X, Y, optimizer):
    """Returns: epoch_loss (float), per_layer_node_means (list of np arrays, length=depth, each (width,))"""
    n = X.shape[0]
    # Accumulate weighted by batch size to get a true epoch mean
    accum = [tf.zeros([model.width], dtype=tf.float32) for _ in range(model.depth)]
    total_loss = 0.0
    seen = 0
    for xb, yb in batches(X, Y, BATCH_SIZE):
        bs = tf.shape(xb)[0]
        loss, per_layer = batch_step(model, xb, yb, optimizer)
        total_loss += float(loss) * int(bs)
        seen += int(bs)
        for l in range(model.depth):
            accum[l] = accum[l] + per_layer[l] * tf.cast(bs, tf.float32)
    epoch_means = [ (accum[l] / float(seen)).numpy() for l in range(model.depth) ]
    return total_loss / float(seen), epoch_means

# -----------------------
# Plotting frames
# -----------------------
def ensure_dir(d):
    if os.path.exists(d):
        return
    os.makedirs(d, exist_ok=True)

def plot_frame(epoch_idx,
               losses_v, losses_e,
               grad_hist_v, grad_hist_e,
               model_v: DeepMLP, model_e: DeepMLP):
    """
    Draw a frame with:
      - Loss curves
      - Heatmaps for 3 representative layers for each model (epochs × nodes, log10 |grad|)
      - Histograms of |grad| at current epoch for each model
      - % vanished / exploded nodes at current epoch
    """
    # Assemble epoch × nodes arrays for the chosen layers
    layers = chosen_layers(model_v.depth)
    # Stack up to current epoch inclusive
    def stack_grad_matrix(grad_hist):
        # grad_hist: list length EPOCHS; each item is list length depth of (width, ) arrays
        mats = []
        for li in layers:
            rows = []
            for e in range(epoch_idx + 1):
                rows.append(np.log10(np.clip(grad_hist[e][li], 1e-12, None)))
            mat = np.stack(rows, axis=0)  # shape: (epoch_idx+1, width)
            mats.append(mat)
        return mats

    mats_v = stack_grad_matrix(grad_hist_v)
    mats_e = stack_grad_matrix(grad_hist_e)

    # Collect current-epoch grad magnitudes (all hidden nodes, all layers) for histogram + stats
    def flatten_epoch(grad_hist):
        cur = grad_hist[epoch_idx]
        arr = np.concatenate([g for g in cur], axis=0)  # shape: depth*width
        return arr

    flat_v = flatten_epoch(grad_hist_v)
    flat_e = flatten_epoch(grad_hist_e)
    flat_v = np.where(np.isfinite(flat_v), flat_v, 0.0)
    flat_e = np.where(np.isfinite(flat_e), flat_e, 1_000_000_000.0)

    pct_vanished_v = 100.0 * float((flat_v < VANISH_THR).sum()) / float(flat_v.size)
    pct_exploded_v = 100.0 * float((flat_v > EXPLODE_THR).sum()) / float(flat_v.size)
    pct_vanished_e = 100.0 * float((flat_e < VANISH_THR).sum()) / float(flat_e.size)
    pct_exploded_e = 100.0 * float((flat_e > EXPLODE_THR).sum()) / float(flat_e.size)

    # Figure layout
    plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(3, 4, height_ratios=[1.2, 1.5, 1.5], hspace=0.4, wspace=0.35)

    # --- Loss curves (span all columns)
    ax_loss = plt.subplot(gs[0, :])
    ax_loss.plot(range(1, epoch_idx+2), losses_v[:epoch_idx+1], label="Vanishing-prone (sigmoid)")
    ax_loss.plot(range(1, epoch_idx+2), losses_e[:epoch_idx+1], label="Exploding-prone (relu+big init+scale)")
    ax_loss.set_title("Training loss over epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper center")
    ax_loss.grid(True, alpha=0.3)

    # --- Heatmaps: 3 layers × 2 models (left=vanish, right=explode), rows 1-2 of grid
    titles = [f"Layer {layers[0]} (early)", f"Layer {layers[1]} (middle)", f"Layer {layers[2]} (late)"]
    for i in range(3):
        # Vanishing model heatmap
        ax = plt.subplot(gs[1, i])
        im = ax.imshow(mats_v[i], aspect='auto', origin='lower', vmin=LOG_MIN, vmax=LOG_MAX)
        ax.set_title(f"Vanishing (sigmoid) – {titles[i]}")
        ax.set_xlabel("Node index")
        ax.set_ylabel("Epoch")
        # Exploding model heatmap
        ax2 = plt.subplot(gs[2, i])
        im2 = ax2.imshow(mats_e[i], aspect='auto', origin='lower', vmin=LOG_MIN, vmax=LOG_MAX)
        ax2.set_title(f"Exploding (relu) – {titles[i]}")
        ax2.set_xlabel("Node index")
        ax2.set_ylabel("Epoch")

    # Shared colorbar for the heatmaps (rightmost column)
    cax = plt.subplot(gs[1:, 3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("log10(|∂L/∂z|)")

    # --- Histograms (current epoch) & stats (top-right small box)
    # Place histogram above colorbar in the top-right area
    axh = ax_loss.inset_axes([0.8, 0.15, 0.18, 0.7])  # relative to loss axes
    bins = np.linspace(-12, 2, 50)
    axh.hist(np.log10(np.clip(flat_v, 1e-12, None)), bins=bins, alpha=0.7, label="vanish")
    axh.hist(np.log10(np.clip(flat_e, 1e-12, None)), bins=bins, alpha=0.7, label="explode")
    axh.set_title("log10(|∂L/∂z|) @ epoch")
    axh.set_xlabel("")
    axh.set_yticks([])
    axh.legend(fontsize=8)

    # Text box with percentages
    txt = (
        f"Vanishing model:\n"
        f"  % nodes < {VANISH_THR:g}: {pct_vanished_v:.1f}%\n"
        f"  % nodes > {EXPLODE_THR:g}: {pct_exploded_v:.1f}%\n\n"
        f"Exploding model:\n"
        f"  % nodes < {VANISH_THR:g}: {pct_vanished_e:.1f}%\n"
        f"  % nodes > {EXPLODE_THR:g}: {pct_exploded_e:.1f}%"
    )
    ax_loss.text(0.015, 0.35, txt, transform=ax_loss.transAxes,
                 bbox=dict(boxstyle="round", alpha=0.1), fontsize=9)

    plt.suptitle(f"Vanishing vs Exploding Gradients — Epoch {epoch_idx+1}/{EPOCHS}", y=0.99)
    outpath = os.path.join(FRAME_DIR, f"frame_{epoch_idx:04d}.png")
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close()

# -----------------------
# Main
# -----------------------
def main():
    print("Generating data…")
    X, Y = make_toy_data(N_SAMPLES)

    if os.path.isdir(FRAME_DIR):
        print(f"Clearing existing frames in {FRAME_DIR} …")
        for f in os.listdir(FRAME_DIR):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(FRAME_DIR, f))
                except:
                    pass
    ensure_dir(FRAME_DIR)

    # Build the two models
    model_v = DeepMLP(in_dim=2, out_dim=2, depth=DEPTH, width=WIDTH,
                      activation="sigmoid", init_std=INIT_STD_VANISH, scale_per_layer=1.0, name="vanish")

    model_e = DeepMLP(in_dim=2, out_dim=2, depth=DEPTH, width=WIDTH,
                      activation="relu", init_std=INIT_STD_EXPLODE, scale_per_layer=SCALE_PER_LAYER, name="explode")

    opt_v = tf.keras.optimizers.SGD(learning_rate=VANISHING_LR)
    opt_e = tf.keras.optimizers.SGD(learning_rate=EXPLODING_LR)

    # Storage: losses and gradient history
    losses_v, losses_e = [], []
    grad_hist_v = []  # list of length EPOCHS; each item = list length depth of (width,) arrays
    grad_hist_e = []

    print("Training…")
    for epoch in range(EPOCHS):
        # One epoch on each model (same data each epoch)
        loss_v, grads_v = train_one_epoch(model_v, X, Y, opt_v)
        loss_e, grads_e = train_one_epoch(model_e, X, Y, opt_e)
        losses_v.append(loss_v)
        losses_e.append(loss_e)
        grad_hist_v.append(grads_v)
        grad_hist_e.append(grads_e)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
              f"Loss vanish={loss_v:.4f}  explode={loss_e:.4f}")

        # Render frame
        plot_frame(epoch, losses_v, losses_e, grad_hist_v, grad_hist_e, model_v, model_e)

    print("\nAll frames written to:", FRAME_DIR)

    # Try to stitch video with ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("\n[!] ffmpeg not found on PATH. Install it to build the video, e.g.:")
        print("    macOS:  brew install ffmpeg")
        print("    Ubuntu: sudo apt-get install ffmpeg")
        print("Then run manually:")
        print(f'    ffmpeg -y -framerate 30 -i {FRAME_DIR}/frame_%04d.png -pix_fmt yuv420p {VIDEO_OUT}')
    else:
        print("\nStitching frames into video with ffmpeg…")
        cmd = [
            ffmpeg
            , "-y"
            , "-framerate", "30"
            , "-i", os.path.join(FRAME_DIR, "frame_%04d.png")
            , "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2"
            , "-pix_fmt", "yuv420p",
            VIDEO_OUT
        ]
        subprocess.run(cmd, check=True)
        print("Video saved to:", VIDEO_OUT)

if __name__ == "__main__":
    main()
