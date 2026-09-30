"""Train the stroke classifier (src/analysis/stroke_tcn.py).

Stage 1, pretrain on E2E-Spot's broadcast strokes (scripts/build_stroke_data_e2e.py
output), holding one whole match out to validate on:

    python scripts/train_stroke_tcn.py --train "data/strokes/e2e/*.npz" \\
        --val-match wimbledon_2019_womens_final_halep_williams \\
        --test data/strokes/ours/dingles_serve_volley.npz --out weights/stroke_tcn.pt

Stage 2, fine-tune on our own labels (once there are enough) - start from
stage 1 and train a few epochs at a lower rate:

    python scripts/train_stroke_tcn.py --train "data/strokes/ours/*.npz" --init weights/stroke_tcn.pt \\
        --epochs 15 --lr 1e-4 --test <a session held out> --out weights/stroke_tcn_ours.pt

Score a trained model without training (--epochs 0):

    python scripts/train_stroke_tcn.py --init weights/stroke_tcn.pt --epochs 0 \\
        --test data/strokes/ours/dingles_serve_volley.npz

Every run writes <out>.json beside the weights: per-class accuracy and the
confusion matrix on validation and test, so results can be compared later.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.stroke_tcn import CLASSES, SEQ_LEN, build_model, features, mirror_hand  # noqa: E402


def load(paths):
    """Features, end flags, labels and meta from npz files."""
    X, far, y, meta = [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        for i in range(len(d["label"])):
            X.append(features(d["kp"][i], d["conf"][i], d["box"][i], bool(d["far"][i])))
            far.append(bool(d["far"][i]))
            y.append(int(d["label"][i]))
            meta.append(str(d["meta"][i]))
    if not X:
        return np.zeros((0, SEQ_LEN, 1), np.float32), np.zeros(0, bool), np.zeros(0, int), []
    return np.stack(X), np.array(far), np.array(y), meta


def augment(x: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.5:
        x = mirror_hand(x)  # the same stroke by an opposite-handed player
    shift = int(rng.integers(-4, 5))  # the swing detector's timing wanders a few frames
    if shift:
        x = np.roll(x, shift, axis=0)
        if shift > 0:
            x[:shift] = 0
        else:
            x[shift:] = 0
    x = x.copy()
    x[:, :68] *= rng.uniform(0.9, 1.1)
    x[:, :34] += rng.normal(0, 0.01, size=x[:, :34].shape) * (x[:, 68:].repeat(2, axis=1) > 0)
    drop = rng.random(17) < 0.1  # a joint the pose model lost
    for j in np.flatnonzero(drop):
        x[:, [2 * j, 2 * j + 1, 34 + 2 * j, 35 + 2 * j, 68 + j]] = 0
    return x


def evaluate(model, X, far, y, device, batch=512):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(torch.from_numpy(X[i:i + batch]).to(device), torch.from_numpy(far[i:i + batch]).to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, int)
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    per = {c: (round(conf[i, i] / conf[i].sum(), 3) if conf[i].sum() else None, int(conf[i].sum()))
           for i, c in enumerate(CLASSES)}
    recalls = [v[0] for v in per.values() if v[0] is not None]
    return {"accuracy": round(float((pred == y).mean()), 3) if len(y) else None,
            "balanced_accuracy": round(float(np.mean(recalls)), 3) if recalls else None,
            "per_class_recall_and_count": per, "confusion": conf.tolist(), "classes": list(CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=None, help="Glob of npz files to train on")
    parser.add_argument("--val-match", default="wimbledon_2019_womens_final_halep_williams",
                        help="Windows whose meta starts with this are held out for validation")
    parser.add_argument("--test", nargs="*", default=[], help="npz files to score at the end (never trained on)")
    parser.add_argument("--init", help="Start from these weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "weights" / "stroke_tcn.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = build_model().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    report = {"device": device, "args": vars(args)}

    if args.epochs > 0:
        paths = sorted(glob.glob(args.train))
        if not paths:
            raise SystemExit(f"No training files match {args.train}")
        X, far, y, meta = load(paths)
        val = np.array([m.startswith(args.val_match) for m in meta]) if args.val_match else np.zeros(len(y), bool)
        Xt, ft, yt = X[~val], far[~val], y[~val]
        Xv, fv, yv = X[val], far[val], y[val]
        counts = np.bincount(yt, minlength=len(CLASSES))
        print(f"train {len(yt)} windows, validate {len(yv)} ({args.val_match}) on {device}")
        print("  train per class: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts)))
        # Rare classes (volleys, overheads) weighted up, by the square root so
        # they do not drown out the common ones.
        weights = torch.tensor(np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0), dtype=torch.float32)
        weights = (weights / weights[counts > 0].mean()).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        steps = args.epochs * max(1, int(np.ceil(len(yt) / args.batch)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
        best, best_state = -1.0, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(yt))
            total, started = 0.0, time.time()
            for i in range(0, len(order), args.batch):
                idx = order[i:i + args.batch]
                xb = np.stack([augment(Xt[j], rng) for j in idx])
                logits = model(torch.from_numpy(xb).to(device), torch.from_numpy(ft[idx]).to(device))
                loss = loss_fn(logits, torch.from_numpy(yt[idx]).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss) * len(idx)
            score = evaluate(model, Xv, fv, yv, device) if len(yv) else evaluate(model, Xt, ft, yt, device)
            key = score["balanced_accuracy"] or 0.0
            if key > best:
                best, best_state = key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss {total / len(yt):.3f}  {'val' if len(yv) else 'train'} accuracy "
                  f"{score['accuracy']}  balanced {score['balanced_accuracy']}  ({time.time() - started:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        report["validation"] = evaluate(model, Xv, fv, yv, device) if len(yv) else None
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"Saved the best epoch (balanced accuracy {best}) to {args.out}")

    for p in args.test:
        X, far, y, _ = load([p])
        report.setdefault("test", {})[p] = res = evaluate(model, X, far, y, device)
        print(f"test {p}: accuracy {res['accuracy']}, balanced {res['balanced_accuracy']}")
        for c, (rec, n) in res["per_class_recall_and_count"].items():
            if n:
                print(f"    {c:16s} {n:4d}  recall {rec}")
    out_json = Path(args.out).with_suffix(".json") if args.epochs > 0 else Path(args.init or args.out).with_suffix(".eval.json")
    out_json.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
