"""Shared NumPy MLP for research baselines; historical BaselineMLP is untouched."""
import numpy as np
from baselines.mlp import BaselineMLP


class ResearchMLP(BaselineMLP):
    param_names = ("W1", "b1", "W2", "b2")

    def __init__(self, lr=0.01, **kwargs):
        super().__init__(lr=lr, **kwargs)
        # Make the intended historical fp32 architecture explicit across NumPy versions.
        for name in self.param_names:
            setattr(self, name, getattr(self, name).astype(np.float32))
        self.updates = 0
        self.replay_draws = 0

    def parameters(self):
        return {name: getattr(self, name) for name in self.param_names}

    def logits(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.n_in)
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def gradients(self, x, d_logits, h):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.n_in)
        dh = (d_logits @ self.W2.T) * (h > 0)
        return {"W1": x.T @ dh, "b1": dh.sum(axis=0),
                "W2": h.T @ d_logits, "b2": d_logits.sum(axis=0)}

    def ce_gradients(self, x, y):
        h, z = self.logits(x)
        probabilities = np.exp(z - z.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = np.asarray(y).reshape(-1)
        probabilities[np.arange(len(labels)), labels] -= 1
        probabilities /= len(labels)
        return self.gradients(x, probabilities, h), z.copy()

    def apply_gradients(self, gradients):
        for name, parameter in self.parameters().items():
            parameter -= self.lr * gradients[name]
            if not np.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite MLP parameter: {name}")
        self.updates += 1

    def observe(self, raw_x, y):
        gradients, _ = self.ce_gradients(normalize(raw_x), [y])
        self.apply_gradients(gradients)


def normalize(raw_x):
    return np.asarray(raw_x, dtype=np.float32) / np.float32(255)


def add_gradients(destination, source, scale=1.0):
    for name in destination:
        destination[name] += scale * source[name]
