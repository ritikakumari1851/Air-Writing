"""The training script runs end to end (tiny settings)."""
import json
import sys

import numpy as np

import train_model
from airwriting.cnn import CNN


def test_quick_training_run(tmp_path, monkeypatch):
    out = tmp_path / "m.npz"
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--quick", "--fonts", "none", "--model-size", "small",
                                      "--out", str(out), "--report-dir", str(tmp_path)])
    train_model.main()
    net, meta = CNN.load(out)
    assert meta["n_classes"] == 36
    report = json.loads((tmp_path / "m_report.json").read_text())
    assert "val_acc" in report


def test_mnist_and_emnist_loaders(tmp_path):
    import gzip
    import struct
    from airwriting import datagen

    def idx(path, arr):
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">HBB", 0, 8, arr.ndim) + struct.pack(">" + "I" * arr.ndim, *arr.shape) + arr.tobytes())

    imgs = np.zeros((3, 28, 28), np.uint8)
    imgs[:, 5:23, 12:16] = 255
    for name in ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"):
        idx(tmp_path / name, imgs)
    for name in ("train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        idx(tmp_path / name, np.array([1, 7, 1], np.uint8))
    xtr, ytr, xte, yte = datagen.load_mnist(str(tmp_path))
    assert xtr.shape == (3, 28, 28) and list(ytr) == [1, 7, 1]

    idx(tmp_path / "emnist-letters-train-images-idx3-ubyte.gz", imgs.transpose(0, 2, 1).copy())
    idx(tmp_path / "emnist-letters-train-labels-idx1-ubyte.gz", np.array([1, 9, 26], np.uint8))
    x, chars = datagen.load_emnist(str(tmp_path))
    assert list(chars) == ["A", "I", "Z"] and np.array_equal(x[0], imgs[0])
    cloud = datagen.raster_to_cloud(x[0])
    assert cloud is not None and len(cloud) > 10
