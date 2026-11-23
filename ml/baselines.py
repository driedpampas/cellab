"""Gradient-boosting baseline (scikit-learn HistGradientBoosting) for counts and whole-run questions.

    python -m ml.baselines data/elementary_w16.npz --out results/gbdt_elementary

Announced on the Deliverable 1 metrics slide as a tabular non-neural baseline next to AnswerNet.
Uses the exact same train/test/unseen splits and metrics as `ml.evaluate`:
    count after N    average absolute error, in cells (per horizon N)
    stable?          accuracy, F1, and step MAE (where stable)
    cycle            start MAE, length MAE, and exact-length share
    does C occur?    accuracy and F1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import eca
from .data import Data, known, load


def _grid_features(grids: np.ndarray, values: int, continuous: bool) -> np.ndarray:
    """Compact, shift-informative tabular features of a batch of grids [N, *shape]."""
    flat = grids.reshape(len(grids), -1).astype(np.float32)
    feats = [flat.mean(axis=1, keepdims=True), flat.std(axis=1, keepdims=True)]
    if not continuous:
        counts = np.stack([(flat == v).mean(axis=1) for v in range(values)], axis=1).astype(np.float32)
        feats.append(counts)
        if grids.ndim == 2:
            # 1D ring: include raw cells + neighbour-pair transitions
            transitions = (grids != np.roll(grids, 1, axis=1)).mean(axis=1, keepdims=True).astype(np.float32)
            feats.extend([flat, transitions])
        else:
            # 2D grid: row/col density profiles + edge-difference activity
            h_diff = (grids != np.roll(grids, 1, axis=1)).mean(axis=(1, 2), keepdims=False)[:, None].astype(np.float32)
            v_diff = (grids != np.roll(grids, 1, axis=2)).mean(axis=(1, 2), keepdims=False)[:, None].astype(np.float32)
            feats.extend([grids.mean(axis=1).astype(np.float32), grids.mean(axis=2).astype(np.float32), h_diff, v_diff])
    else:
        feats.extend([flat.max(axis=1, keepdims=True), flat.min(axis=1, keepdims=True)])
    return np.concatenate(feats, axis=1)


def _base_features(data: Data, idx: np.ndarray) -> np.ndarray:
    g = data.initial[idx].cpu().numpy()
    x = _grid_features(g, data.values, data.continuous)
    cond = data.cond[idx].cpu().numpy()
    return np.concatenate([x, cond], axis=1) if cond.shape[1] else x


def _query_features(data: Data, idx: np.ndarray, q: int) -> np.ndarray:
    g = data.initial[idx].cpu().numpy()
    c = data.queries[idx, q].cpu().numpy()
    base = _base_features(data, idx)
    qf = _grid_features(c, data.values, data.continuous)
    same = (g == c).reshape(len(idx), -1).mean(axis=1, keepdims=True).astype(np.float32)
    diff = np.abs(g.astype(np.float32) - c.astype(np.float32)).reshape(len(idx), -1).mean(axis=1, keepdims=True)
    return np.concatenate([base, qf, same, diff], axis=1)


def _yes_no(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    hits = int((pred & truth).sum())
    prec = hits / max(int(pred.sum()), 1)
    rec = hits / max(int(truth.sum()), 1)
    return {
        "accuracy": float((pred == truth).mean()),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "share_yes": float(truth.mean()),
    }


def _number(pred_log: np.ndarray, truth: np.ndarray, baseline: float) -> dict[str, float]:
    steps = np.clip(np.round(np.expm1(pred_log)), 0, None)
    return {
        "error": float(np.abs(steps - truth).mean()),
        "exactly_right": float((steps == truth).mean()),
        "common_error": float(np.abs(baseline - truth).mean()),
        "samples": int(len(truth)),
    }


def run_gbdt_baseline(dataset: str | Path | Data, seed: int = 0, max_iter: int = 200) -> dict:
    """Fit HistGradientBoosting models on train split and score on test/unseen splits."""
    t0 = time.time()
    data = dataset if isinstance(dataset, Data) else load(dataset, "cpu")
    train_idx = data.splits["train"].cpu().numpy()
    X_train = _base_features(data, train_idx)
    horizons = data.horizons.cpu().numpy().tolist()
    cells = data.cells

    # 1. Count models (one regressor per state value for discrete; 1 for continuous)
    # Build stacked (sample, horizon) features so one model covers all N like AnswerNet.
    n_train = len(train_idx)
    log_n = np.log1p(np.array(horizons, dtype=np.float32))
    X_count_train = np.concatenate([
        np.repeat(X_train, len(horizons), axis=0),
        np.tile(log_n, n_train)[:, None],
    ], axis=1)
    states_train = data.states[train_idx].cpu().numpy()  # [S, H, *shape]
    count_models = []
    if data.continuous:
        y_mean = states_train.reshape(n_train * len(horizons), -1).mean(axis=1)
        m = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
        m.fit(X_count_train, y_mean)
        count_models.append(m)
    else:
        flat_st = states_train.reshape(n_train * len(horizons), -1)
        # For binary (values == 2), fitting class 1 share alone defines both shares exactly.
        fit_vals = [1] if data.values == 2 else list(range(data.values))
        for v in fit_vals:
            y_share = (flat_st == v).mean(axis=1)
            m = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
            m.fit(X_count_train, y_share)
            count_models.append(m)

    # 2. Stable? + stable_at
    stable_from_tr = data.stable_from[train_idx].cpu().numpy()
    cycle_len_tr = data.cycle_length[train_idx].cpu().numpy()
    cycle_start_tr = data.cycle_start[train_idx].cpu().numpy()
    is_stable_tr = stable_from_tr != -1
    stable_known_tr = (is_stable_tr | (cycle_len_tr != -1)) if data.deterministic else np.ones(n_train, dtype=bool)

    stable_clf = None
    stable_const = None
    if stable_known_tr.any():
        y_st = is_stable_tr[stable_known_tr].astype(int)
        if len(np.unique(y_st)) > 1:
            stable_clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
            stable_clf.fit(X_train[stable_known_tr], y_st)
        else:
            stable_const = int(y_st[0])

    stable_at_reg = None
    if is_stable_tr.any():
        stable_at_reg = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
        stable_at_reg.fit(X_train[is_stable_tr], np.log1p(stable_from_tr[is_stable_tr].astype(np.float32)))

    # 3. Cycle start & length
    cycle_known_tr = cycle_len_tr != -1
    start_reg = length_reg = None
    if cycle_known_tr.any():
        start_reg = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
        start_reg.fit(X_train[cycle_known_tr], np.log1p(cycle_start_tr[cycle_known_tr].astype(np.float32)))
        length_reg = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
        length_reg.fit(X_train[cycle_known_tr], np.log1p(cycle_len_tr[cycle_known_tr].astype(np.float32)))

    # 4. Occurs?
    n_queries = data.queries.shape[1]
    X_occ_parts, y_occ_parts = [], []
    for q in range(n_queries):
        occ_q = data.occurs[train_idx, q].cpu().numpy()
        mask = occ_q != -1
        if mask.any():
            X_occ_parts.append(_query_features(data, train_idx[mask], q))
            y_occ_parts.append((occ_q[mask] == 1).astype(int))
    occurs_clf = None
    occurs_const = None
    if X_occ_parts:
        X_occ_tr = np.concatenate(X_occ_parts, axis=0)
        y_occ_tr = np.concatenate(y_occ_parts, axis=0)
        if len(np.unique(y_occ_tr)) > 1:
            occurs_clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
            occurs_clf.fit(X_occ_tr, y_occ_tr)
        else:
            occurs_const = int(y_occ_tr[0])

    # Precompute training medians for baselines
    med_stable_at = float(np.median(stable_from_tr[is_stable_tr])) if is_stable_tr.any() else 0.0
    med_start = float(np.median(cycle_start_tr[cycle_known_tr])) if cycle_known_tr.any() else 0.0
    med_length = float(np.median(cycle_len_tr[cycle_known_tr])) if cycle_known_tr.any() else 0.0

    def predict_split(idx: np.ndarray) -> dict:
        X = _base_features(data, idx)
        n_s = len(idx)
        X_c = np.concatenate([np.repeat(X, len(horizons), axis=0), np.tile(log_n, n_s)[:, None]], axis=1)
        if data.continuous:
            pred_count = count_models[0].predict(X_c).reshape(n_s, len(horizons), 1)
        elif data.values == 2:
            p1 = np.clip(count_models[0].predict(X_c), 0.0, 1.0).reshape(n_s, len(horizons))
            pred_count = np.stack([1.0 - p1, p1], axis=-1)
        else:
            raw = np.stack([np.clip(m.predict(X_c), 0.0, 1.0) for m in count_models], axis=-1).reshape(n_s, len(horizons), data.values)
            pred_count = raw / np.clip(raw.sum(axis=-1, keepdims=True), 1e-6, None)

        pred_stable = stable_clf.predict(X) if stable_clf is not None else np.full(n_s, stable_const or 0)
        pred_stable_at = stable_at_reg.predict(X) if stable_at_reg is not None else np.zeros(n_s)
        pred_start = start_reg.predict(X) if start_reg is not None else np.zeros(n_s)
        pred_length = length_reg.predict(X) if length_reg is not None else np.zeros(n_s)

        occ_preds = []
        for q in range(n_queries):
            Xq = _query_features(data, idx, q)
            occ_preds.append(occurs_clf.predict(Xq) if occurs_clf is not None else np.full(n_s, occurs_const or 0))
        pred_occurs = np.stack(occ_preds, axis=1)
        return {"count": pred_count, "stable": pred_stable, "stable_at": pred_stable_at,
                "start": pred_start, "length": pred_length, "occurs": pred_occurs}

    def score_subset(idx: np.ndarray, preds: dict) -> dict:
        res: dict = {"samples": int(len(idx))}
        if not len(idx):
            return res
        st_truth = data.states[idx].cpu().numpy()
        init_arr = data.initial[idx].cpu().numpy()
        by_h = []
        for h, n in enumerate(horizons):
            truth_h = st_truth[:, h].reshape(len(idx), -1)
            init_f = init_arr.reshape(len(idx), -1)
            if data.continuous:
                t_mean = truth_h.mean(axis=1)
                c_err = float(np.abs(preds["count"][:, h, 0] - t_mean).mean() * data.scale)
                copy_err = float(np.abs(init_f.mean(axis=1) - t_mean).mean() * data.scale)
            else:
                t_share = np.stack([(truth_h == v).mean(axis=1) for v in range(data.values)], axis=1)
                i_share = np.stack([(init_f == v).mean(axis=1) for v in range(data.values)], axis=1)
                c_err = float(np.abs(preds["count"][:, h] - t_share).sum(axis=1).mean() * cells / 2)
                copy_err = float(np.abs(i_share - t_share).sum(axis=1).mean() * cells / 2)
            by_h.append({"n": n, "model": {"count_error": c_err}, "copy": {"count_error": copy_err}})
        res["by_horizon"] = by_h

        sf = data.stable_from[idx].cpu().numpy()
        cl = data.cycle_length[idx].cpu().numpy()
        cs = data.cycle_start[idx].cpu().numpy()
        is_st = sf != -1
        st_known = (is_st | (cl != -1)) if data.deterministic else np.ones(len(idx), dtype=bool)
        if st_known.any():
            res["stable"] = _yes_no(preds["stable"][st_known], is_st[st_known])
            if is_st.any():
                res["stable"]["step"] = _number(preds["stable_at"][is_st], sf[is_st].astype(np.float32), med_stable_at)
        ck = cl != -1
        if ck.any():
            res["cycle_start"] = _number(preds["start"][ck], cs[ck].astype(np.float32), med_start)
            res["cycle_length"] = _number(preds["length"][ck], cl[ck].astype(np.float32), med_length)
        occ = data.occurs[idx].cpu().numpy()
        ok = occ != -1
        if ok.any():
            res["occurs"] = _yes_no(preds["occurs"][ok], occ[ok] == 1)
        return res

    report = {
        "kind": "gbdt",
        "dataset": data.name,
        "max_iter": max_iter,
        "seconds": round(time.time() - t0, 2),
        "splits": {},
    }
    for split in ("test", "unseen"):
        if split not in data.splits:
            continue
        idx = data.splits[split].cpu().numpy()
        preds = predict_split(idx)
        part = {"all": score_subset(idx, preds)}
        groups = data.group[idx].cpu().numpy()
        for g in sorted(set(groups.tolist()) - {0}):
            m = groups == g
            part[eca.CLASS_NAMES[g]] = score_subset(idx[m], {k: v[m] for k, v in preds.items()})
        report["splits"][split] = part
    report["seconds"] = round(time.time() - t0, 2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--compare", help="optional AnswerNet metrics.json to print side-by-side comparison")
    args = parser.parse_args()

    report = run_gbdt_baseline(args.dataset, args.seed, args.max_iter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"GBDT baseline on {report['dataset']} finished in {report['seconds']:.1f}s")
    nn = json.loads(Path(args.compare).read_text(encoding="utf-8")) if args.compare else None
    for split, groups in report["splits"].items():
        g = groups["all"]
        n_last = g["by_horizon"][-1]["n"]
        c_err = g["by_horizon"][-1]["model"]["count_error"]
        st_f1 = g["stable"]["f1"] if "stable" in g else float("nan")
        cyc = g["cycle_length"]["error"] if "cycle_length" in g else float("nan")
        occ_f1 = g["occurs"]["f1"] if "occurs" in g else float("nan")
        print(f"  {split:6s} GBDT | count(N={n_last})={c_err:.2f}  stable_F1={st_f1:.3f}  cycle_MAE={cyc:.2f}  occurs_F1={occ_f1:.3f}")
        if nn and split in nn["splits"]:
            ng = nn["splits"][split]["all"]
            nc = ng["by_horizon"][-1]["model"]["count_error"]
            ns = ng["stable"]["f1"] if "stable" in ng else float("nan")
            ny = ng["cycle_length"]["error"] if "cycle_length" in ng else float("nan")
            no = ng["occurs"]["f1"] if "occurs" in ng else float("nan")
            print(f"  {split:6s} NN   | count(N={n_last})={nc:.2f}  stable_F1={ns:.3f}  cycle_MAE={ny:.2f}  occurs_F1={no:.3f}")


if __name__ == "__main__":
    main()
