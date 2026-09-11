"""DER++: current CE + alpha * logit MSE + beta * replay CE.

Reference: Buzzega et al., NeurIPS 2020; authors' Mammoth models/derpp.py.
One independently drawn reservoir item for each replay loss; insert afterwards.
"""
from baselines.research_mlp import ResearchMLP, normalize, add_gradients
from baselines.research_replay import ReservoirBuffer


class DERPP(ResearchMLP):
    def __init__(self, alpha=0.5, beta=0.5, capacity=300, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta = alpha, beta
        self.buffer = ReservoirBuffer(capacity, n_in=self.n_in, n_out=self.n_out,
                                      store_logits=True, seed=seed)

    def observe(self, raw_x, y):
        gradients, observed_logits = self.ce_gradients(normalize(raw_x), [y])
        if self.buffer.size:
            past = self.buffer.draw()
            replay_x = normalize(past["image"])
            h, current_logits = self.logits(replay_x)
            # MSE averages over all ten logits; no softmax or class masking.
            derivative = 2 * self.alpha * (current_logits - past["logits"]) / self.n_out
            add_gradients(gradients, self.gradients(replay_x, derivative, h))
            labelled_past = self.buffer.draw()
            replay_gradient, _ = self.ce_gradients(normalize(labelled_past["image"]),
                                                   [int(labelled_past["label"])])
            add_gradients(gradients, replay_gradient, self.beta)
            self.replay_draws += 2
        self.apply_gradients(gradients)
        # Logits were produced when originally observed, BEFORE the SGD update.
        self.buffer.add(raw_x, y, observed_logits[0])
