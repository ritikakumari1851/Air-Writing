#!/usr/bin/env python3
"""Train the air-writing character recogniser (digits 0-9 + letters A-Z).

Runs anywhere NumPy and OpenCV run - no TensorFlow or PyTorch needed.

    python train_model.py                       # synthetic strokes + fonts only
    python train_model.py --mnist data/mnist.npz --emnist data/emnist
    python train_model.py --user-data user_data --init models/char_cnn.npz --epochs 4   # fine-tune on your own writing

Real handwriting is optional but improves accuracy:
  * MNIST digits   - mnist.npz (Keras format) or a folder with the 4 IDX .gz files
  * EMNIST letters - folder with emnist-letters-train-images-idx3-ubyte.gz + labels
  * your own samples recorded with `python air_writing.py --collect`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airwriting import datagen  # noqa: E402
from airwriting.cnn import ARCHS, CNN, AdamW, cross_entropy  # noqa: E402
from airwriting.preprocess import DEFAULT_THICKNESS, render_strokes  # noqa: E402
from airwriting.templates import CLASSES, DIGITS, TEMPLATES  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mnist", help="mnist.npz or folder with MNIST IDX files")
    ap.add_argument("--emnist", help="folder with EMNIST letters (or balanced/byclass) IDX files")
    ap.add_argument("--user-data", help="samples.jsonl file or folder recorded with --collect")
    ap.add_argument("--fonts", default="auto", help="'auto' (installed fonts), 'none', a folder, or comma-separated font files")
    ap.add_argument("--test-fonts", default="", help="comma-separated font files used only for testing")
    ap.add_argument("--per-class", type=int, default=4000, help="training images per character")
    ap.add_argument("--real-per-class", type=int, default=3000, help="max real handwriting samples per character")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--regen-every", type=int, default=3, help="draw a fresh augmented training set every N epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--smoothing", type=float, default=0.05, help="label smoothing")
    ap.add_argument("--model-size", choices=sorted(ARCHS), default="large", help="network size (small trains fastest)")
    ap.add_argument("--init", help="start from an existing model (fine-tuning)")
    ap.add_argument("--out", default="models/char_cnn.npz")
    ap.add_argument("--report-dir", default="docs", help="where to write training plots/report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    return ap.parse_args()


# ---------------------------------------------------------------------------

def resolve_fonts(spec: str):
    if spec == "none":
        return []
    if spec == "auto":
        return datagen.find_system_fonts()
    if os.path.isdir(spec):
        import glob
        return sorted(glob.glob(os.path.join(spec, "**", "*.[ot]tf"), recursive=True))
    return [p for p in spec.split(",") if p]


def clean_template_set():
    X, y = [], []
    for ch in CLASSES:
        for _, strokes in TEMPLATES[ch]:
            X.append(render_strokes(strokes=strokes, thickness=DEFAULT_THICKNESS))
            y.append(CLASSES.index(ch))
    return (np.stack(X) * 255).astype(np.uint8), np.array(y)


def clouds_to_images(clouds_by_char):
    X, y = [], []
    for ch, clouds in clouds_by_char.items():
        for c in clouds:
            X.append(render_strokes(clouds=[c], thickness=DEFAULT_THICKNESS))
            y.append(CLASSES.index(ch))
    if not X:
        return None
    return (np.stack(X) * 255).astype(np.uint8), np.array(y)


def evaluate(net, X, y, restrict=None, batch=512):
    """Accuracy; `restrict` = list of allowed class indices (e.g. digits only)."""
    correct, preds = 0, []
    for i in range(0, len(y), batch):
        p = net.predict_proba(X[i:i + batch].astype(np.float32) / 255.0)
        if restrict is not None:
            mask = np.full(p.shape[1], -1.0)
            mask[restrict] = 0.0
            p = p + mask
        preds.append(p.argmax(axis=1))
    preds = np.concatenate(preds)
    return float((preds == y).mean()), preds


def lr_at(step, total, base, warmup=300):
    warmup = max(1, min(warmup, total // 10))
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, t))))


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not installed - skipping plots)")
        return None


def plot_curves(report_dir, history):
    """Loss and accuracy per epoch -> training_curves.png (also usable on a saved *_report.json)."""
    plt = _pyplot()
    if plt is None:
        return
    os.makedirs(report_dir, exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
    for key, label in (("train_acc", "training batches"), ("val_acc", "synthetic validation"),
                       ("font_acc", "unseen fonts"), ("mnist_acc_digits_mode", "MNIST test digits, 0-9 mode"),
                       ("mnist_acc", "MNIST test digits, A-Z 0-9 mode")):
        vals = [h.get(key) for h in history]
        if any(v is not None for v in vals):
            ax[1].plot(ep, [v * 100 if v is not None else np.nan for v in vals], marker="o", label=label)
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
    lo = min(min(line.get_ydata()) for line in ax[1].get_lines())
    ax[1].set_ylim(max(0.0, lo - (100 - lo) * 0.6), 100)      # leave room for the legend underneath
    ax[1].legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(report_dir, "training_curves.png"), dpi=130); plt.close(fig)


def save_report(report_dir, history, conf, sample_X, sample_y):
    plot_curves(report_dir, history)
    plt = _pyplot()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, fontsize=7)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (synthetic validation)")
    fig.tight_layout(); fig.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(6, 12, figsize=(12, 6.4))
    for a in axes.ravel():
        a.axis("off")
    shown = {}
    for img, lab in zip(sample_X, sample_y):
        if shown.get(lab, 0) < 2:
            k = sum(shown.values())
            if k >= 72:
                break
            a = axes.ravel()[k]
            a.imshow(img, cmap="gray"); a.set_title(CLASSES[lab], fontsize=8)
            shown[lab] = shown.get(lab, 0) + 1
    fig.suptitle("Examples of generated training images"); fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "training_samples.png"), dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.quick:
        args.per_class, args.epochs, args.real_per_class = 60, 1, 60
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("Collecting glyph sources...")
    fonts = datagen.hershey_clouds()
    font_files = resolve_fonts(args.fonts)
    test_fonts = [p for p in args.test_fonts.split(",") if p]
    font_files = [f for f in font_files if os.path.abspath(f) not in {os.path.abspath(t) for t in test_fonts}]
    if font_files:
        ttf = datagen.ttf_clouds(font_files)
        for ch in CLASSES:
            fonts[ch].extend(ttf[ch])
    print(f"  fonts: 12 Hershey styles + {len(font_files)} installed font files "
          f"({sum(len(v) for v in fonts.values())} glyphs)")

    mnist_pool = mnist_test = None
    if args.mnist:
        xtr, ytr, xte, yte = datagen.load_mnist(args.mnist)
        chars = np.array([str(int(v)) for v in ytr])
        mnist_pool = datagen.clouds_from_images(xtr, chars, args.real_per_class, rng)
        print(f"  MNIST: {sum(len(v) for v in mnist_pool.values())} training digits")
        n_test = 1000 if args.quick else len(yte)
        test_clouds = {c: [] for c in CLASSES}
        for img, lab in zip(xte[:n_test], yte[:n_test]):
            cl = datagen.raster_to_cloud(img)
            if cl is not None:
                test_clouds[str(int(lab))].append(cl)
        mnist_test = clouds_to_images(test_clouds)

    emnist_pool = None
    if args.emnist:
        ex, ec = datagen.load_emnist(args.emnist)
        emnist_pool = datagen.clouds_from_images(ex, ec, args.real_per_class, rng)
        print(f"  EMNIST: {sum(len(v) for v in emnist_pool.values())} training characters")

    user_pool = None
    if args.user_data:
        user_pool = datagen.load_user_samples(args.user_data)
        n_user = sum(len(v) for v in user_pool.values())
        print(f"  your recorded samples: {n_user}")
        if n_user == 0:
            user_pool = None

    font_test = None
    if test_fonts:
        font_test = clouds_to_images(datagen.ttf_clouds(test_fonts))

    def make_train(seed):
        r = np.random.default_rng(seed)
        return datagen.build_dataset(args.per_class, r, fonts=fonts, mnist=mnist_pool, emnist=emnist_pool,
                                     user=user_pool, verbose=True)

    print(f"Generating {args.per_class * len(CLASSES):,} training images...")
    Xtr, ytr_, src = make_train(args.seed + 1)
    print("Generating validation images...")
    Xva, yva, _ = datagen.build_dataset(max(20, args.per_class // 20), np.random.default_rng(args.seed + 999),
                                        fonts=fonts, mnist=mnist_pool, emnist=emnist_pool, user=user_pool, verbose=False)
    Xcl, ycl = clean_template_set()
    print(f"  ready in {time.time() - t0:.0f}s")

    if args.init:
        net, meta = CNN.load(args.init)
        if meta["classes"] != CLASSES:
            sys.exit("--init model has different classes")
        print(f"Fine-tuning {args.init}")
    else:
        net = CNN(len(CLASSES), arch=ARCHS[args.model_size], seed=args.seed)
    print(f"Network: {net.n_params():,} parameters")
    opt = AdamW(net, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = int(np.ceil(len(ytr_) / args.batch_size))
    total = steps_per_epoch * args.epochs
    history, best, step = [], (-1.0, None), 0
    digit_idx = [CLASSES.index(d) for d in DIGITS]
    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and args.regen_every and (epoch - 1) % args.regen_every == 0:
            print("  drawing a fresh augmented training set...")
            Xtr, ytr_, src = make_train(args.seed + epoch)
        perm = rng.permutation(len(ytr_))
        te, losses, hits, seen = time.time(), [], 0, 0
        for bi in range(steps_per_epoch):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            xb = Xtr[idx].astype(np.float32) / 255.0
            yb = ytr_[idx]
            opt.lr = lr_at(step, total, args.lr) if not args.init else args.lr * 0.25 * (1 - step / total) + 1e-5
            logits = net.forward(xb, train=True)
            loss, d = cross_entropy(logits, yb, args.smoothing)
            net.backward(d)
            opt.step()
            step += 1
            losses.append(loss)
            hits += int((logits.argmax(axis=1) == yb).sum()); seen += len(yb)
            if bi % 50 == 0 or bi == steps_per_epoch - 1:
                print(f"\r  epoch {epoch}/{args.epochs}  batch {bi + 1}/{steps_per_epoch}  loss {np.mean(losses[-50:]):.3f}"
                      f"  lr {opt.lr:.5f}", end="", flush=True)
        val_acc, _ = evaluate(net, Xva, yva)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": hits / seen, "val_acc": val_acc,
               "clean_template_acc": evaluate(net, Xcl, ycl)[0], "seconds": round(time.time() - te, 1)}
        if mnist_test is not None:
            rec["mnist_acc"] = evaluate(net, *mnist_test)[0]
            rec["mnist_acc_digits_mode"] = evaluate(net, *mnist_test, restrict=digit_idx)[0]
        if font_test is not None:
            rec["font_acc"] = evaluate(net, *font_test)[0]
        history.append(rec)
        extra = "".join(f"  {k} {v * 100:.2f}%" for k, v in rec.items() if k.endswith("acc") and k not in ("train_acc", "val_acc"))
        print(f"\r  epoch {epoch}: loss {rec['train_loss']:.3f}  val {val_acc * 100:.2f}%{extra}  ({rec['seconds']:.0f}s)" + " " * 10)
        if val_acc > best[0]:
            best = (val_acc, [arr.copy() for _, _, arr in net.parameters()], epoch)

    # restore the best epoch
    for (_, _, arr), saved in zip(net.parameters(), best[1]):
        arr[...] = saved
    val_acc, val_pred = evaluate(net, Xva, yva)
    conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    np.add.at(conf, (yva, val_pred), 1)
    confusions = sorted(((int(conf[i, j]), CLASSES[i], CLASSES[j]) for i in range(len(CLASSES))
                         for j in range(len(CLASSES)) if i != j and conf[i, j]), reverse=True)[:12]
    summary = {
        "trained": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best[2],
        "val_acc": val_acc,
        "history": history,
        "top_confusions": [f"{a} -> {b}: {n}" for n, a, b in confusions],
        "data": {
            "per_class": args.per_class, "font_files": [os.path.basename(f) for f in font_files],
            "mnist": bool(args.mnist), "emnist": bool(args.emnist), "user_data": bool(user_pool),
        },
    }
    best_rec = history[best[2] - 1]
    for k in ("mnist_acc", "mnist_acc_digits_mode", "font_acc", "clean_template_acc"):
        if k in best_rec:
            summary[k] = best_rec[k]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    net.save(args.out, CLASSES, extra={"thickness": DEFAULT_THICKNESS, "training": {k: v for k, v in summary.items() if k != "history"}})
    with open(os.path.splitext(args.out)[0] + "_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.quick:
        save_report(args.report_dir, history, conf, Xtr[:3000], ytr_[:3000])

    print(f"\nSaved {args.out}  (best epoch {best[2]}, validation accuracy {val_acc * 100:.2f}%)")
    for k in ("mnist_acc", "mnist_acc_digits_mode", "font_acc"):
        if k in summary:
            print(f"  {k}: {summary[k] * 100:.2f}%")
    print("  most common confusions:", ", ".join(summary["top_confusions"][:6]))
    print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
