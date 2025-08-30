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

from typing import Tuple, Any, Generator, Dict

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
        return Z1, A1, Z2, A2


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

        a2, z1, a1, z2 = cache
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