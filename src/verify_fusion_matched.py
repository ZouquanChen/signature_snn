"""Audited local-feature evaluation gated by exact legacy embedding reproduction.

Does not modify old verifiers, checkpoints, exports or results. CPU matching;
all three branches share view/query/reference IDs, original min/mean scoring,
fixed original threshold grid and the original compute_eer implementation.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import verifier_stylus_snn_tvs as legacy
import verifier_stylus_embedding as original_embedding
import real_forgery_data as real
from numba import njit


VIEWS = [("mcyt", None), ("bsid", None), ("bsds2", None), ("ebio2", None)] + [
    ("ebio1", device) for device in range(5)]


def view_indices(ids, subset, device):
    if device is not None:
        p = np.array([0, 1, 10, 11, 20, 21, 30, 31]) + device * 2
        n = np.array([40, 41, 42, 55, 56, 57]) + device * 3
        return ids[np.concatenate((p, n))]
    if subset == "bsds2":
        return ids[np.r_[0:4, 15:50]]
    return ids


def episodes(labels, inds, subset, device, references):
    if references not in (1, 4):
        raise ValueError("references must be 1 or 4")
    for loop in range(4 if references == 1 else 1):
        for writer in np.unique(labels):
            ids = view_indices(np.flatnonzero(labels == writer), subset, device)
            candidate = ids[inds[ids] == 1][:4]
            if len(candidate) != 4:
                raise ValueError("four genuine candidate references required")
            refs = candidate[loop:loop + 1] if references == 1 else candidate
            # Even 1vs1 excludes ALL four candidate references from queries.
            queries = ids[~np.isin(ids, candidate)]
            yield loop, int(writer), refs, queries


def score_matrix(dist, template_dist):
    # Reuse, rather than reimplement, the legacy float32 scoring operations.
    parts = legacy.compute_modality_score(dist, template_dist.copy())
    return np.sum(parts * [1, 1], axis=1)


def curves(positive, negative):
    th = legacy.get_thresholds(np.asarray(positive)[:, None], np.asarray(negative)[:, None], "none").ravel()
    # Exactly equivalent inequalities to the original subtract/compare arrays,
    # without allocating a signatures x 5000 matrix.
    p = np.sort(np.asarray(positive))
    n = np.sort(np.asarray(negative))
    frr = 1.0 - np.searchsorted(p, th, side="right") / len(p)
    far = 1.0 - (len(n) - np.searchsorted(n, th, side="left")) / len(n)
    return far, frr


def pooled(rows, field, references):
    far_sum = np.zeros(5000, dtype=np.float64)
    frr_sum = np.zeros(5000, dtype=np.float64)
    count_p = count_n = 0.0
    loops = 4 if references == 1 else 1
    for view in range(len(VIEWS)):
        for loop in range(loops):
            selected = (rows["view"] == view) & (rows["loop"] == loop)
            p = rows[field][selected & (rows["genuine"] == 1)]
            n = rows[field][selected & (rows["genuine"] == 0)]
            far, frr = curves(p, n)
            far_sum += far * len(n) / loops
            frr_sum += frr * len(p) / loops
            count_p += len(p) / loops
            count_n += len(n) / loops
    eer = legacy.getEER(far_sum / (count_n + 1e-8), frr_sum / (count_p + 1e-8))[0] * 100
    return dict(eer_percent=float(eer), genuine_trials=int(count_p * loops),
                forgery_trials=int(count_n * loops), pooled_genuine_count=count_p,
                pooled_forgery_count=count_n)


def load_exports(args):
    exports = {}
    manifest = real.load_manifest(args.data_root)
    manifest_hash = real.sha256_file(Path(args.data_root) / "manifest.json")
    for subset in dict(VIEWS):
        base = Path("exps") / args.baseline_export / ("seed%d" % args.seed) / subset
        local = Path("exps") / args.local_export / ("seed%d" % args.seed) / subset
        labels = np.load(base / ("labels_epoch%s.npy" % args.epoch))
        inds = np.load(base / ("inds_epoch%s.npy" % args.epoch))
        emb = np.load(base / ("feats_epoch%s.npy" % args.epoch))
        for directory in (base, local):
            provenance = json.loads((directory / "real_data_provenance.json").read_text())
            if (provenance["manifest_sha256"] != manifest_hash or provenance["split"] != "evaluation"):
                raise ValueError("export provenance mismatch")
        for prefix, expected in (("labels", labels), ("inds", inds)):
            if not np.array_equal(np.load(local / ("%s_epoch%s.npy" % (prefix, args.epoch))), expected):
                raise ValueError("local export sample ordering differs from baseline")
        other = np.load(local / ("feats_epoch%s.npy" % args.epoch))
        if not np.allclose(emb, other, rtol=1e-5, atol=1e-6):
            raise ValueError("local export embedding differs from frozen baseline")
        record = manifest["files"]["evaluation/" + subset]
        if (len(np.unique(labels)) != record["writers"] or len(emb) != record["genuine"] + record["forgery"]):
            raise ValueError("export counts differ from recorded data")
        if not np.isfinite(emb).all():
            raise ValueError("nonfinite embeddings")
        exports[subset] = (legacy.l2_norm(emb), labels, inds, local)
    return exports


def audit_embedding(exports, references):
    blocks = []
    for view, (subset, device) in enumerate(VIEWS):
        emb, labels, inds, _ = exports[subset]
        captured = []
        original = legacy.compute_combined_score
        def capture(*a, **kw):
            result = original(*a, **kw)
            captured.append(result[:, 0].copy())
            return result
        legacy.compute_combined_score = capture
        try:
            zeros = np.zeros((len(emb), 1), dtype=np.float32)
            legacy.verify_dataset_sf(
                dict(embedding=emb, spike_rate=zeros, spike_time=zeros), labels, inds,
                len(np.unique(labels)), references, 4 if references == 1 else 1,
                (1., 0., 0.), "none",
                np.r_[0:4, 15:50] if subset == "bsds2" else None, device)
        finally:
            legacy.compute_combined_score = original
        schedule = list(episodes(labels, inds, subset, device, references))
        if len(schedule) != len(captured):
            raise AssertionError("episode count differs from original verifier")
        for (loop, writer, refs, queries), expected in zip(schedule, captured):
            value = score_matrix(legacy.compute_dist_mat(emb[queries], emb[refs]),
                                 legacy.compute_dist_mat(emb[refs], emb[refs]))
            np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-12)
            ref_ids = np.full((len(queries), 4), -1, dtype=np.int64)
            ref_ids[:, :len(refs)] = refs
            blocks.append(dict(view=np.full(len(queries), view), loop=np.full(len(queries), loop),
                               writer=np.full(len(queries), writer), query=queries,
                               reference=ref_ids, genuine=inds[queries], embedding=value))
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def prepare_local(local, epoch, length):
    seq = np.load(local / ("seq_feat_epoch%s.npy" % epoch), mmap_mode="r")
    mask = np.load(local / ("seq_mask_epoch%s.npy" % epoch), mmap_mode="r")
    result = []
    for x, valid in zip(seq, mask):
        n = int(np.count_nonzero(valid))
        if n < 1 or not np.array_equal(valid.astype(bool), np.arange(len(valid)) < n):
            raise ValueError("invalid sequence padding mask")
        x = np.asarray(x[:, :n].T, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite local features")
        # Linear sampling shared by pointwise/DTW. Normalize AFTER interpolation
        # so 1-dot is actually cosine distance and self-distance is zero.
        grid = np.linspace(0, n - 1, length)
        lo = np.floor(grid).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        weight = (grid - lo)[:, None]
        x = x[lo] * (1 - weight) + x[hi] * weight
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        if (norm < 1e-12).any():
            raise ValueError("zero local vector prevents cosine distance")
        result.append(np.ascontiguousarray(x / norm))
    return result


@njit(cache=False)
def dtw(a, b, radius):
    n = len(a)
    prev = np.full(n + 1, np.inf)
    prev[0] = 0.
    lengths = np.zeros(n + 1, np.int64)
    for i in range(1, n + 1):
        cur = np.full(n + 1, np.inf)
        cur_l = np.zeros(n + 1, np.int64)
        for j in range(max(1, i - radius), min(n, i + radius) + 1):
            # Diagonal first for deterministic shortest tie path.
            best, size = prev[j - 1], lengths[j - 1]
            if prev[j] < best:
                best, size = prev[j], lengths[j]
            if cur[j - 1] < best:
                best, size = cur[j - 1], cur_l[j - 1]
            cost = max(0., 1. - np.dot(a[i - 1], b[j - 1]))
            cur[j], cur_l[j] = best + cost, size + 1
        prev, lengths = cur, cur_l
    return prev[n] / lengths[n]


def local_scores(rows, exports, args):
    point = np.empty(len(rows["query"]))
    warped = np.empty_like(point)
    timing = dict(preparation_seconds=0., pointwise_seconds=0., dtw_seconds=0., unique_pairs=0)
    radius = int(np.ceil(args.band * args.length))
    dtw(np.eye(4), np.eye(4), 2)  # Exclude compilation from reported matching time.
    for subset in dict(VIEWS):
        start = time.perf_counter()
        features = prepare_local(exports[subset][3], args.epoch, args.length)
        timing["preparation_seconds"] += time.perf_counter() - start
        cache = {}
        def pair(a, b):
            if a == b:
                return 0., 0.
            key = (int(a), int(b))
            if key not in cache:
                start = time.perf_counter()
                p = float(np.maximum(0., 1. - np.sum(features[a] * features[b], axis=1)).mean())
                timing["pointwise_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                d = float(dtw(features[a], features[b], radius))
                timing["dtw_seconds"] += time.perf_counter() - start
                cache[key] = p, d
            return cache[key]
        for view, (name, device) in enumerate(VIEWS):
            if name != subset:
                continue
            selected = np.flatnonzero(rows["view"] == view)
            # Group by loop/writer; batch scoring matches the original verifier.
            for loop in np.unique(rows["loop"][selected]):
                for writer in np.unique(rows["writer"][selected]):
                    group = selected[(rows["loop"][selected] == loop) & (rows["writer"][selected] == writer)]
                    refs = rows["reference"][group[0]]
                    refs = refs[refs >= 0]
                    temp = np.asarray([[pair(a, b) for b in refs] for a in refs])
                    dist = np.asarray([[pair(q, ref) for ref in refs] for q in rows["query"][group]])
                    point[group] = score_matrix(dist[:, :, 0], temp[:, :, 0])
                    warped[group] = score_matrix(dist[:, :, 1], temp[:, :, 1])
        timing["unique_pairs"] += len(cache)
        print("LOCAL DONE", subset, "unique pairs", len(cache), flush=True)
    rows.update(pointwise=point, dtw=warped, fusion=0.5 * rows["embedding"] + 0.5 * warped)
    for name in ("pointwise", "dtw"):
        timing[name + "_ms_per_pair"] = 1000 * timing[name + "_seconds"] / timing["unique_pairs"]
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-export", required=True)
    parser.add_argument("--local-export", required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data_real_forgery/deepsigndb_stylus_v1")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epoch", default="BestTrainLoss")
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--band", type=float, default=.25)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.length < 2 or not 0 <= args.band <= 1:
        parser.error("invalid length/band")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    exports = load_exports(args)
    expected = json.loads((args.baseline_results / "summary.json").read_text())
    report = dict(alpha=.5, protocol="legacy-verifier-exact-query-score-curve-v1", audit={}, results={},
                  local_preprocessing="96-grid linear interpolation then per-vector L2; cosine cost",
                  length=args.length, band=args.band, comparison_files_verified=False,
                  source_sha256={Path(__file__).name: real.sha256_file(__file__)})
    all_rows = {}
    # BOTH embedding branches must pass before any local matching is reported.
    for refs in (1, 4):
        oracle = original_embedding.main([
            "--run-tag", args.baseline_export, "--seeds", str(args.seed), "--epoch", args.epoch,
            "--datasets", "mcyt,bsid,bsds2,ebio1,ebio2", "--mode", "sf",
            "--references", str(refs), "--embedding-normalization", "l2"])
        rows = audit_embedding(exports, refs)
        actual = pooled(rows, "embedding", refs)
        recorded = expected[str(refs) + "vs1"]["mean_eer_percent"]
        if max(abs(oracle - recorded), abs(actual["eer_percent"] - oracle)) > 1e-6:
            raise AssertionError("embedding baseline reproduction failed: %s" % (recorded, oracle, actual))
        report["audit"][str(refs)] = dict(recorded=recorded, original=oracle, reproduced=actual,
                                          per_query_scores_match=True, passed=True)
        all_rows[refs] = rows
        np.savez_compressed(args.output_dir / ("embedding_scores_%dvs1.npz" % refs), **rows)
    (args.output_dir / "embedding_audit.json").write_text(json.dumps(report, indent=2))
    print("BOTH EMBEDDING AUDITS PASSED", flush=True)
    if args.audit_only:
        return
    for refs, rows in all_rows.items():
        timing = local_scores(rows, exports, args)
        report["results"][str(refs)] = {name: pooled(rows, name, refs) for name in ("embedding", "pointwise", "dtw", "fusion")}
        report["results"][str(refs)]["timing"] = timing
        np.savez_compressed(args.output_dir / ("all_scores_%dvs1.npz" % refs), **rows)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
