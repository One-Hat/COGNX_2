"""Task-wise diagonal empirical-Fisher EWC (Kirkpatrick et al., 2017).

Each old task retains its own Fisher and optimum; no online-EWC approximation.
The empirical Fisher uses observed labels, not model-sampled labels. It is an
explicit approximation to the model Fisher, computed without weight updates.
"""
import numpy as np
from baselines.research_mlp import ResearchMLP, normalize


class EWC(ResearchMLP):
    def __init__(self, ewc_lambda=100.0, **kwargs):
        super().__init__(**kwargs)
        self.ewc_lambda = ewc_lambda
        self.fishers = []
        self.optima = []
        self.fisher_sample_counts = []

    def penalty_gradients(self):
        result = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        # d/dtheta [lambda/2 sum_tasks sum_i F_i (theta_i - old_i)^2].
        for fisher, optimum in zip(self.fishers, self.optima):
            for name, parameter in self.parameters().items():
                result[name] += self.ewc_lambda * fisher[name] * (parameter - optimum[name])
        return result

    def observe(self, raw_x, y):
        gradients, _ = self.ce_gradients(normalize(raw_x), [y])
        penalty = self.penalty_gradients()
        for name in gradients:
            gradients[name] += penalty[name]
        self.apply_gradients(gradients)

    def consolidate(self, raw_images, labels, indices):
        if len(indices) == 0:
            raise ValueError("Fisher estimation needs training samples")
        fisher = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        # Square each sample's gradient BEFORE averaging. Squaring a minibatch
        # mean would introduce cross terms and is not diagonal empirical Fisher.
        for index in indices:
            gradient, _ = self.ce_gradients(normalize(raw_images[index]), [int(labels[index])])
            for name in fisher:
                fisher[name] += gradient[name] ** 2 / len(indices)
        self.fishers.append(fisher)
        self.optima.append({k: v.copy() for k, v in self.parameters().items()})
        self.fisher_sample_counts.append(len(indices))
