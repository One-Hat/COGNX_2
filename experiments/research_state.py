"""Stateless checkpoint inference at the experiment layer; no MNEMA edits."""
import hashlib
import pickle
import random
import numpy as np
from baselines.research_mlp import normalize


def state_hash(model):
    """Hash ALL stored object attributes (arrays/scalars), including RNG state.

    This deliberately covers more than the small inference restoration set, so
    newly introduced mutations elsewhere fail the evaluation integrity check.
    """
    def state(value):
        if isinstance(value, np.random.Generator):
            return value.bit_generator.state
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, dict):
            return {k: state(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [state(v) for v in value]
        if hasattr(value, "__dict__"):
            return (type(value).__name__, state(vars(value)))
        return value
    payload = (state(model), np.random.get_state(), random.getstate())
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


class CheckpointInference:
    def __init__(self, model):
        self.model = model
        # Only these attributes can change in the audited native inference path.
        fields = [(model.encoder, "theta"), (model.encoder, "x_ref"),
                  (model.separator, "theta"), (model.separator, "running_rate"),
                  (model.cortex, "v"), (model.cortex, "b")]
        fields += [(model.controller.bus, name) for name in vars(model.controller.bus)]
        self.saved = []
        for owner, name in fields:
            original = getattr(owner, name)
            value = original.copy() if isinstance(original, np.ndarray) else original
            self.saved.append((owner, name, original, value))
        self.numpy_rng = np.random.get_state()
        self.python_rng = random.getstate()

    @property
    def snapshot_array_bytes(self):
        return sum(value.nbytes for _, _, _, value in self.saved if isinstance(value, np.ndarray))

    def restore(self):
        for owner, name, original, value in self.saved:
            if isinstance(value, np.ndarray):
                np.copyto(original, value)
            # Restore original references too: several native methods rebind arrays.
            setattr(owner, name, original)
        np.random.set_state(self.numpy_rng)
        random.setstate(self.python_rng)

    def predict(self, raw_image, instrument=None):
        try:
            output = self.model.step(normalize(raw_image), is_training=False, instrument=instrument)
            probabilities = np.asarray(output["probabilities"])
            assert probabilities.shape == (10,) and np.isfinite(probabilities).all()
            prediction = int(output["prediction"])
            assert 0 <= prediction <= 9
            return prediction
        finally:
            # Also restore after exceptions. Every image begins at the same checkpoint.
            self.restore()


def evaluate(model, images, indices, is_mnema=False, instrument=None, reverse_check=False):
    # No labels or task IDs are accepted by this prediction API.
    before = state_hash(model)
    checkpoint = CheckpointInference(model) if is_mnema else None

    def predict(index, charge=True):
        inst = instrument if charge else None
        if checkpoint is not None:
            return checkpoint.predict(images[index], inst)
        _, probabilities = model.forward(normalize(images[index]).reshape(1, -1), instrument=inst)
        assert probabilities.shape == (1, 10) and np.isfinite(probabilities).all()
        return int(probabilities.argmax())

    predictions = np.array([predict(index) for index in indices], dtype=np.uint8)
    if reverse_check:
        reversed_predictions = np.array([predict(index, False) for index in indices[::-1]], dtype=np.uint8)
        assert np.array_equal(predictions, reversed_predictions[::-1]), "Test order changed predictions"
    after = state_hash(model)
    assert before == after, "Evaluation modified model or RNG state"
    return predictions, {"state_before": before, "state_after": after,
                         "state_equal": before == after, "reverse_order_checked": reverse_check,
                         "snapshot_array_bytes": checkpoint.snapshot_array_bytes if checkpoint else 0}
