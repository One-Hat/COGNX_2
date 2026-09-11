"""Full MNIST in uint8 (~55 MB), shared by all sequential method runs."""
import gzip
import hashlib
import struct
from pathlib import Path
import numpy as np
from benchmarks.split_mnist import download_mnist

EXPECTED_MD5 = {
    "train_img": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train_lbl": "d53e105ee54ea40749a09fcbcd1e9432",
    "test_img": "9fb629c4189551a2d022fa330f9573f3",
    "test_lbl": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_dataset(directory):
    paths = download_mnist(str(directory))
    data, hashes = {}, {}
    for key, path in paths.items():
        compressed = Path(path).read_bytes()
        if hashlib.md5(compressed).hexdigest() != EXPECTED_MD5[key]:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        hashes[Path(path).name] = hashlib.sha256(compressed).hexdigest()
        del compressed
        with gzip.open(path, "rb") as handle:
            magic, count = struct.unpack(">II", handle.read(8))
            if key.endswith("img"):
                if magic != 2051 or struct.unpack(">II", handle.read(8)) != (28, 28):
                    raise ValueError("Invalid MNIST image header")
                array = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, 784)
            else:
                if magic != 2049:
                    raise ValueError("Invalid MNIST label header")
                array = np.frombuffer(handle.read(), dtype=np.uint8)
                if array.shape != (count,) or not np.isin(array, np.arange(10)).all():
                    raise ValueError("Invalid MNIST labels")
        data[key] = array
    assert len(data["train_img"]) == len(data["train_lbl"]) == 60000
    assert len(data["test_img"]) == len(data["test_lbl"]) == 10000
    data["hashes"] = hashes
    return data


def task_indices(data, seed, quick=False, quick_train=100, quick_test=100):
    # Generate ONCE per seed, independently of model RNG. Same arrays for all methods.
    stream_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    test_rng = np.random.default_rng(2026)
    trains, tests, sizes = [], [], []
    for task in range(5):
        classes = [2 * task, 2 * task + 1]
        train = np.flatnonzero(np.isin(data["train_lbl"], classes))
        test = np.flatnonzero(np.isin(data["test_lbl"], classes))
        sizes.append({"classes": classes, "full_train": len(train), "full_test": len(test)})
        train = stream_rng.permutation(train)
        if quick:
            train = train[:quick_train]
            test = test_rng.permutation(test)[:quick_test]
        trains.append(train)
        tests.append(test)
        sizes[-1].update(train=len(train), test=len(test))
    return trains, tests, sizes


def sequence_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
