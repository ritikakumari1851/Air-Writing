"""A small convolutional neural network written with NumPy only.

No TensorFlow / PyTorch needed: the same code trains the model
(train_model.py) and runs it inside the app. Tensors are channels-last
(batch, height, width, channels) and float32.

Architecture (input 28x28x1), shown for the "large" size that train_model.py
uses by default; "default" and "small" have fewer filters (see ARCHS):
    conv3x3(24) -> ReLU -> maxpool2   -> 14x14x24
    conv3x3(48) -> ReLU -> maxpool2   -> 7x7x48
    conv3x3(96) -> ReLU -> maxpool2   -> 3x3x96
    flatten -> dense(192) -> ReLU -> dropout(0.4) -> dense(n_classes)
"""
from __future__ import annotations

import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def make_arch(c1=16, c2=32, c3=64, hidden=128, dropout=0.4):
    return [
        ["conv", 1, c1], ["relu"], ["pool"],
        ["conv", c1, c2], ["relu"], ["pool"],
        ["conv", c2, c3], ["relu"], ["pool"],
        ["flatten"],
        ["dense", c3 * 3 * 3, hidden], ["relu"], ["dropout", dropout],
        ["dense", hidden, None],       # None -> number of classes
    ]


ARCHS = {
    "small": make_arch(12, 24, 48, 96),
    "default": make_arch(16, 32, 64, 128),
    "large": make_arch(24, 48, 96, 192),
}
DEFAULT_ARCH = ARCHS["default"]


class Conv2D:
    """3x3 'same' convolution: im2col followed by a single matrix multiply."""

    def __init__(self, c_in, c_out, k=3, rng=None, first=False):
        rng = rng or np.random.default_rng(0)
        fan_in = c_in * k * k
        # weight layout (c_out, c_in, k, k) matches the im2col column order (c, ki, kj)
        self.W = (rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(c_out, np.float32)
        self.k, self.pad, self.first = k, k // 2, first
        self.params = {"W": self.W, "b": self.b}

    def forward(self, x, train=False):
        N, H, W, C = x.shape
        p, k = self.pad, self.k
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        cols = sliding_window_view(xp, (k, k), axis=(1, 2)).reshape(N * H * W, C * k * k)
        out = cols @ self.W.reshape(len(self.b), -1).T
        out += self.b
        if train:
            self.cache = (cols, x.shape)
        return out.reshape(N, H, W, -1)

    def backward(self, dout):
        cols, (N, H, W, C) = self.cache
        F, k, p = len(self.b), self.k, self.pad
        d = dout.reshape(-1, F)
        self.grads = {"W": (d.T @ cols).reshape(self.W.shape), "b": d.sum(axis=0)}
        self.cache = None
        if self.first:                      # gradient w.r.t. the image itself is not needed
            return None
        dcols = (d @ self.W.reshape(F, -1)).reshape(N, H, W, C, k, k)
        dxp = np.zeros((N, H + 2 * p, W + 2 * p, C), np.float32)
        for i in range(k):
            for j in range(k):
                dxp[:, i:i + H, j:j + W, :] += dcols[:, :, :, :, i, j]
        return dxp[:, p:p + H, p:p + W, :]


class ReLU:
    params: dict = {}

    def forward(self, x, train=False):
        if train:
            self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        return np.multiply(dout, self.mask, out=dout)


class MaxPool2:
    """2x2 max pooling with stride 2 (an odd last row/column is dropped: 7 -> 3)."""
    params: dict = {}

    def forward(self, x, train=False):
        H2, W2 = x.shape[1] // 2, x.shape[2] // 2
        a = x[:, 0:2 * H2:2, 0:2 * W2:2]
        b = x[:, 0:2 * H2:2, 1:2 * W2:2]
        c = x[:, 1:2 * H2:2, 0:2 * W2:2]
        d = x[:, 1:2 * H2:2, 1:2 * W2:2]
        out = np.maximum(np.maximum(a, b), np.maximum(c, d))
        if train:
            ma = a == out
            mb = (b == out) & ~ma
            mc = (c == out) & ~(ma | mb)
            md = ~(ma | mb | mc)
            self.cache = (ma, mb, mc, md, x.shape)
        return out

    def backward(self, dout):
        ma, mb, mc, md, shape = self.cache
        H2, W2 = shape[1] // 2, shape[2] // 2
        dx = np.zeros(shape, np.float32)
        dx[:, 0:2 * H2:2, 0:2 * W2:2] = dout * ma
        dx[:, 0:2 * H2:2, 1:2 * W2:2] = dout * mb
        dx[:, 1:2 * H2:2, 0:2 * W2:2] = dout * mc
        dx[:, 1:2 * H2:2, 1:2 * W2:2] = dout * md
        self.cache = None
        return dx


class Flatten:
    params: dict = {}

    def forward(self, x, train=False):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, n_in, n_out, rng=None, gain=2.0):
        rng = rng or np.random.default_rng(0)
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(gain / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, np.float32)
        self.params = {"W": self.W, "b": self.b}

    def forward(self, x, train=False):
        if train:
            self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.grads = {"W": self.x.T @ dout, "b": dout.sum(axis=0)}
        self.x = None
        return dout @ self.W.T


class Dropout:
    params: dict = {}

    def __init__(self, p, rng=None):
        self.p = p
        self.rng = rng or np.random.default_rng(0)

    def forward(self, x, train=False):
        if not train or self.p <= 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape, dtype=np.float32) >= self.p).astype(np.float32) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


class CNN:
    def __init__(self, n_classes, arch=None, seed=0):
        self.arch = json.loads(json.dumps(arch or DEFAULT_ARCH))
        for spec in self.arch:
            if spec[0] == "dense" and spec[2] is None:
                spec[2] = n_classes
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        self.layers = []
        first = True
        last_dense = max(i for i, spec in enumerate(self.arch) if spec[0] == "dense")
        for i, spec in enumerate(self.arch):
            kind = spec[0]
            if kind == "conv":
                self.layers.append(Conv2D(spec[1], spec[2], rng=rng, first=first))
                first = False
            elif kind == "relu":
                self.layers.append(ReLU())
            elif kind == "pool":
                self.layers.append(MaxPool2())
            elif kind == "flatten":
                self.layers.append(Flatten())
            elif kind == "dense":   # smaller start-up weights for the output layer
                self.layers.append(Dense(spec[1], spec[2], rng=rng, gain=0.25 if i == last_dense else 2.0))
            elif kind == "dropout":
                self.layers.append(Dropout(spec[1], rng=rng))
            else:
                raise ValueError(f"unknown layer {kind}")

    # -- inference ---------------------------------------------------------
    def forward(self, x, train=False):
        x = np.asarray(x, np.float32)
        if x.ndim == 3:                      # (N, H, W) -> (N, H, W, 1)
            x = x[..., None]
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def predict_proba(self, x, batch=512):
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x[None]
        outs = [softmax(self.forward(x[i:i + batch])) for i in range(0, len(x), batch)]
        return np.concatenate(outs)

    # -- training ----------------------------------------------------------
    def backward(self, dlogits):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)
            if d is None:
                break

    def parameters(self):
        for li, layer in enumerate(self.layers):
            for name, arr in layer.params.items():
                yield (li, name), layer, arr

    def n_params(self):
        return int(sum(a.size for _, _, a in self.parameters()))

    # -- persistence ---------------------------------------------------------
    def save(self, path, classes, extra=None):
        data = {f"L{li}_{name}": arr for (li, name), _, arr in self.parameters()}
        meta = {"arch": self.arch, "classes": list(classes), "n_classes": self.n_classes, "layout": "NHWC"}
        meta.update(extra or {})
        data["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            net = cls(meta["n_classes"], arch=meta["arch"])
            for (li, name), layer, arr in net.parameters():
                arr[...] = f[f"L{li}_{name}"]
        return net, meta


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y, smoothing=0.0):
    """Mean softmax cross-entropy (optionally label-smoothed) and its gradient w.r.t. the logits."""
    n, k = logits.shape
    z = logits - logits.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    target = np.full((n, k), smoothing / k, np.float32)
    target[np.arange(n), y] += 1.0 - smoothing
    loss = -(target * log_p).sum(axis=1).mean()
    return float(loss), ((np.exp(log_p) - target) / n).astype(np.float32)


class AdamW:
    def __init__(self, net, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.net, self.lr, self.b1, self.b2, self.eps, self.wd = net, lr, betas[0], betas[1], eps, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self):
        self.t += 1
        b1t, b2t = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for key, layer, arr in self.net.parameters():
            g = layer.grads[key[1]]
            m = self.m.setdefault(key, np.zeros_like(arr))
            v = self.v.setdefault(key, np.zeros_like(arr))
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.wd and key[1] == "W":
                arr -= self.lr * self.wd * arr
            arr -= self.lr * (m / b1t) / (np.sqrt(v / b2t) + self.eps)
