"""The NumPy CNN: shapes, exact gradients, learning, save/load."""
import types
from pathlib import Path

import numpy as np

from airwriting.cnn import CNN, AdamW, cross_entropy

SMALL_ARCH = [["conv", 1, 3], ["relu"], ["pool"], ["conv", 3, 4], ["relu"], ["pool"], ["flatten"],
              ["dense", 4 * 2 * 2, 5], ["relu"], ["dense", 5, None]]


def float64_module():
    """The same CNN code with float64 maths, so finite differences are precise."""
    src = (Path(__file__).resolve().parent.parent / "airwriting" / "cnn.py").read_text()
    mod = types.ModuleType("cnn64")
    exec(compile(src.replace("np.float32", "np.float64"), "cnn64", "exec"), mod.__dict__)
    return mod


def test_forward_shapes():
    net = CNN(36)
    out = net.forward(np.zeros((5, 28, 28), np.float32))
    assert out.shape == (5, 36)
    p = net.predict_proba(np.random.rand(3, 28, 28))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_gradients_match_finite_differences():
    m = float64_module()
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(3):
        net = m.CNN(3, arch=SMALL_ARCH, seed=trial)
        x, y = rng.standard_normal((4, 9, 9)), rng.integers(0, 3, 4)
        _, d = m.cross_entropy(net.forward(x, train=True), y, smoothing=0.1)
        net.backward(d)
        for key, layer, arr in net.parameters():
            flat, g = arr.reshape(-1), layer.grads[key[1]].reshape(-1)
            for i in range(flat.size):
                old = flat[i]
                flat[i] = old + 1e-6
                lp = m.cross_entropy(net.forward(x), y, smoothing=0.1)[0]
                flat[i] = old - 1e-6
                lm = m.cross_entropy(net.forward(x), y, smoothing=0.1)[0]
                flat[i] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - g[i]) / max(1e-6, abs(num) + abs(g[i])))
    assert worst < 1e-4


def test_learns_a_toy_problem():
    rng = np.random.default_rng(0)
    x = np.zeros((200, 28, 28), np.float32)
    y = rng.integers(0, 2, 200)
    x[y == 0, 10:18, 4:24] = 1.0     # horizontal bar
    x[y == 1, 4:24, 10:18] = 1.0     # vertical bar
    net = CNN(2, seed=0)
    opt = AdamW(net, lr=3e-3)
    for _ in range(30):
        loss, d = cross_entropy(net.forward(x, train=True), y)
        net.backward(d)
        opt.step()
    assert (net.predict_proba(x).argmax(axis=1) == y).mean() == 1.0


def test_save_and_load_roundtrip(tmp_path):
    net = CNN(4, seed=3)
    x = np.random.rand(2, 28, 28).astype(np.float32)
    path = tmp_path / "m.npz"
    net.save(path, list("ABCD"), extra={"thickness": 2.5})
    net2, meta = CNN.load(path)
    assert meta["classes"] == list("ABCD") and meta["thickness"] == 2.5
    assert np.allclose(net.predict_proba(x), net2.predict_proba(x), atol=1e-6)
