"""Research ER: reservoir sampling, replay before insertion, one joint SGD step."""
import numpy as np
from baselines.research_mlp import ResearchMLP, normalize


class ReservoirBuffer:
    def __init__(self, capacity=300, budget_bytes=None, n_in=784, n_out=10,
                 store_logits=False, seed=0):
        # All data-dependent buffer metadata live in this array: seen and occupied.
        self.counters = np.zeros(2, dtype=np.int64)
        row_dtype = np.dtype([("image", np.uint8, (n_in,)), ("label", np.uint8)]
                             + ([("logits", np.float32, (n_out,))] if store_logits else []))
        if budget_bytes is not None:
            capacity = (budget_bytes - self.counters.nbytes) // row_dtype.itemsize
        if capacity < 1:
            raise ValueError("Buffer budget cannot hold one item and counters")
        self.rows = np.zeros(capacity, dtype=row_dtype)
        self.budget_bytes = budget_bytes
        self.rng = np.random.default_rng(seed)
        self.assert_budget()

    @property
    def size(self):
        return int(self.counters[1])

    @property
    def memory_bytes(self):
        # Allocated capacity, not just populated slots; no hidden image list.
        return self.rows.nbytes + self.counters.nbytes

    def assert_budget(self):
        if self.budget_bytes is not None:
            assert self.memory_bytes <= self.budget_bytes

    def draw(self):
        if not self.size:
            raise ValueError("Cannot draw from empty buffer")
        return self.rows[int(self.rng.integers(self.size))]

    def add(self, raw_x, y, logits=None):
        self.counters[0] += 1
        seen = int(self.counters[0])
        capacity = len(self.rows)
        index = seen - 1 if seen <= capacity else int(self.rng.integers(seen))
        if index < capacity:
            self.rows[index]["image"] = raw_x
            self.rows[index]["label"] = y
            if "logits" in self.rows.dtype.names:
                self.rows[index]["logits"] = logits
        self.counters[1] = min(seen, capacity)
        self.assert_budget()

    def memory(self):
        # Field views' nbytes count payload, not structured-array stride padding.
        return {"replay_items": self.size, "replay_capacity": len(self.rows),
                "image_bytes": self.rows["image"].nbytes,
                "label_bytes": self.rows["label"].nbytes,
                "logit_bytes": self.rows["logits"].nbytes if "logits" in self.rows.dtype.names else 0,
                "buffer_metadata_bytes": self.counters.nbytes,
                "buffer_bytes": self.memory_bytes,
                "populated_buffer_bytes": self.size * self.rows.dtype.itemsize + self.counters.nbytes}


class ReplayMLP(ResearchMLP):
    def __init__(self, capacity=300, budget_bytes=None, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ReservoirBuffer(capacity, budget_bytes, self.n_in, self.n_out, seed=seed)

    def observe(self, raw_x, y):
        # Mean CE over current + one past image; both gradients at the same weights.
        if self.buffer.size:
            past = self.buffer.draw()
            x = normalize(np.stack((raw_x, past["image"])))
            labels = [y, int(past["label"])]
            self.replay_draws += 1
        else:
            x, labels = normalize(raw_x), [y]
        gradients, _ = self.ce_gradients(x, labels)
        self.apply_gradients(gradients)
        self.buffer.add(raw_x, y)
