#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
import os
import argparse
try:
    from .evaluation_utils import compute_eer
except ImportError:
    from evaluation_utils import compute_eer
from matplotlib import pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(description="SNN-TVS Stylus verifier")
    parser.add_argument("--references", type=int, choices=[1, 4], default=1)
    parser.add_argument("--epoch", type=str, default="End", help="feature epoch suffix")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="optional run tag for exps subdirectory",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[111, 222, 333, 444, 555],
        help="seed list whose exported features will be aggregated by the verifier",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mcyt,bsid,bsds2,ebio1,ebio2",
        help="comma-separated exported evaluation datasets",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sf", "rf"],
        default="sf",
        help="verification mode: sf (skilled forgery) or rf (random forgery)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="weights for embedding, spike_rate, spike_time (default: 0.7 0.2 0.1)",
    )
    parser.add_argument(
        "--score-norm",
        type=str,
        choices=["none", "zscore"],
        default="none",
        help="optional score normalization before modality fusion",
    )
    parser.add_argument(
        "--layer-subset",
        type=str,
        choices=["all", "last2", "last3", "midhigh"],
        default="all",
        help="optional subset of exported spike layers to use",
    )
    parser.add_argument(
        "--time-view",
        type=str,
        choices=["global", "earlylate"],
        default="global",
        help="label the expected spike-time descriptor variant",
    )
    parser.add_argument(
        "--report-breakdown",
        action="store_true",
        help="print embedding-only and partial-fusion ablations alongside the main run",
    )
    return parser


def getEER(FAR, FRR):
    return compute_eer(FAR, FRR)


def selectTemplate(distMatrix):
    refNum = distMatrix.shape[0]
    if refNum == 1:
        return None, 1, 1, 1, 1, 1
    idx = numpy.argmin(numpy.sum(distMatrix, axis=1) / (refNum - 1))
    dvar = numpy.sqrt(
        (
            numpy.sum(distMatrix**2) / refNum / (refNum - 1)
            - (numpy.sum(distMatrix) / refNum / (refNum - 1)) ** 2
        )
    )
    dmean = numpy.sum(distMatrix) / refNum / (refNum - 1)
    dtmp = numpy.sum(distMatrix[:, idx]) / (refNum - 1)
    dmax = numpy.mean(numpy.max(distMatrix, axis=1))
    distMatrix[range(refNum), range(refNum)] = float("inf")
    distMatrix[distMatrix == 0] = float("inf")
    dmin = numpy.mean(numpy.min(distMatrix, axis=1))
    return idx, dtmp**0.5, dmax**0.5, dmin**0.5, dmean**0.5, dvar


def l2_norm(x):
    return x / (numpy.sum(x**2, axis=-1, keepdims=True) ** 0.5 + 1e-8)


def load_and_norm(feats):
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True) ** 0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    return feats / (numpy.sum(feats**2, axis=1, keepdims=True) ** 0.5 + 1e-8)


def compute_dist_mat(test, temp):
    return numpy.sum((test[:, None, :] - temp[None, :, :]) ** 2, axis=2)


def compute_modality_score(dist_test, dist_temp):
    _, dtmp, dmax, dmin, dmean, dvar = selectTemplate(dist_temp)
    distMax = numpy.max(dist_test, axis=1)[:, None] / (dmax + 1e-8)
    distMin = numpy.min(dist_test, axis=1)[:, None] / (dmin + 1e-8)
    distMean = numpy.mean(dist_test, axis=1)[:, None] / (dmean + 1e-8)
    score3 = numpy.concatenate((distMax, distMin, distMean), axis=1)
    return score3[:, 1:] * 0.1


def normalize_score(score, mode):
    if mode == "none":
        return score
    mean = numpy.mean(score, axis=0, keepdims=True)
    std = numpy.std(score, axis=0, keepdims=True)
    return (score - mean) / (std + 1e-8)


def build_thresholds(positive_scores, negative_scores, num_steps=5000):
    all_scores = numpy.concatenate(
        (positive_scores[:, 0], negative_scores[:, 0]), axis=0
    )
    score_min = float(numpy.min(all_scores))
    score_max = float(numpy.max(all_scores))
    span = max(score_max - score_min, 1e-3)
    margin = max(span * 0.05, 1e-3)
    return numpy.linspace(score_min - margin, score_max + margin, num_steps)[None, :]


def get_thresholds(positive_scores, negative_scores, score_norm):
    if score_norm == "none":
        # Preserve the published SynSig2Vec/DeepSignDB operating grid.
        return numpy.arange(0, 5, 0.001)[None, :]
    return build_thresholds(positive_scores, negative_scores)


def select_feature_subset(feats, dims, subset):
    if subset == "all" or not dims:
        return feats
    if subset == "last2":
        start_layer = max(len(dims) - 2, 0)
    elif subset == "last3":
        start_layer = max(len(dims) - 3, 0)
    elif subset == "midhigh":
        start_layer = min(2, len(dims))
    else:
        raise ValueError("Unsupported layer subset: %s" % subset)
    start_idx = sum(dims[:start_layer])
    return feats[:, start_idx:]


def resolve_run_tag_path(path, run_tag):
    if run_tag and path.startswith("./exps/seed"):
        return path.replace("./exps/", "./exps/%s/" % run_tag, 1)
    return path


def get_breakdown_weight_sets(weights):
    return [
        ("embedding_only", (1.0, 0.0, 0.0)),
        ("embedding_rate", (weights[0], weights[1], 0.0)),
        ("embedding_rate_time", tuple(weights)),
    ]


def maybe_select_spike_layers(spike_feat, meta_path, subset, run_tag=None):
    meta_path = resolve_run_tag_path(meta_path, run_tag)
    if subset == "all" or not os.path.exists(meta_path):
        return spike_feat
    meta = numpy.load(meta_path)
    if spike_feat.shape[1] == int(numpy.sum(meta["rate_dims"])):
        dims = meta["rate_dims"].tolist()
    elif spike_feat.shape[1] == int(numpy.sum(meta["time_dims"])):
        dims = meta["time_dims"].tolist()
    else:
        return spike_feat
    return select_feature_subset(spike_feat, dims, subset)


def compute_combined_score(
    emb_3d_score, rate_3d_score, time_3d_score, weights, score_norm="none"
):
    emb_score = numpy.sum(emb_3d_score * [1, 1], axis=1, keepdims=True)
    rate_score = numpy.sum(rate_3d_score * [1, 1], axis=1, keepdims=True)
    time_score = numpy.sum(time_3d_score * [1, 1], axis=1, keepdims=True)
    emb_score = normalize_score(emb_score, score_norm)
    rate_score = normalize_score(rate_score, score_norm)
    time_score = normalize_score(time_score, score_norm)
    return weights[0] * emb_score + weights[1] * rate_score + weights[2] * time_score


def verify_dataset_sf(
    feats,
    labels,
    inds,
    N_user,
    N_train,
    N_loops,
    weights,
    score_norm="none",
    ds_subset=None,
    ds_ebio1_device=None,
):
    EER_G = []
    EER_L = []
    ROC_FAR = 0
    ROC_FRR = 0
    TOTAL_P = 0
    TOTAL_N = 0

    emb = feats["embedding"]
    rate = feats["spike_rate"]
    time = feats["spike_time"]

    for rand in range(N_loops):
        EERs = []
        datum_p = []
        datum_n = []
        for k in numpy.arange(0, N_user):
            idxUser = numpy.where(labels == k)[0]
            embU = emb[idxUser]
            rateU = rate[idxUser]
            timeU = time[idxUser]
            indU = inds[idxUser]

            if ds_ebio1_device is not None:
                deviceIndP = (
                    numpy.array([0, 1, 10, 11, 20, 21, 30, 31]) + ds_ebio1_device * 2
                )
                deviceIndN = numpy.array([40, 41, 42, 55, 56, 57]) + ds_ebio1_device * 3
                embU = numpy.concatenate((embU[deviceIndP], embU[deviceIndN]), axis=0)
                rateU = numpy.concatenate(
                    (rateU[deviceIndP], rateU[deviceIndN]), axis=0
                )
                timeU = numpy.concatenate(
                    (timeU[deviceIndP], timeU[deviceIndN]), axis=0
                )
                indU = numpy.concatenate((indU[deviceIndP], indU[deviceIndN]), axis=0)

            if ds_subset is not None:
                embU = embU[ds_subset]
                rateU = rateU[ds_subset]
                timeU = timeU[ds_subset]
                indU = indU[ds_subset]

            idxGen = numpy.where(indU)[0]
            idxTemp = idxGen[0:4]
            if N_train == 1:
                tempEmb = embU[idxTemp[rand : rand + 1]]
                tempRate = rateU[idxTemp[rand : rand + 1]]
                tempTime = timeU[idxTemp[rand : rand + 1]]
            else:
                tempEmb = embU[idxTemp]
                tempRate = rateU[idxTemp]
                tempTime = timeU[idxTemp]

            testIdx = numpy.array(sorted(set(range(len(indU))) - set(idxTemp)))
            testEmb = embU[testIdx]
            testRate = rateU[testIdx]
            testTime = timeU[testIdx]
            testInd = indU[testIdx]

            distTempEmb = compute_dist_mat(tempEmb, tempEmb)
            distTempRate = compute_dist_mat(tempRate, tempRate)
            distTempTime = compute_dist_mat(tempTime, tempTime)

            distEmb = compute_dist_mat(testEmb, tempEmb)
            distRate = compute_dist_mat(testRate, tempRate)
            distTime = compute_dist_mat(testTime, tempTime)
            embScore = compute_modality_score(distEmb, distTempEmb)
            rateScore = compute_modality_score(distRate, distTempRate)
            timeScore = compute_modality_score(distTime, distTempTime)

            score = compute_combined_score(
                embScore, rateScore, timeScore, weights, score_norm
            )

            datum_p.append(score[numpy.where(testInd)[0]])
            datum_n.append(score[numpy.where(1 - testInd)[0]])

            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = get_thresholds(user_p, user_n, score_norm)
            FRR = 1.0 - numpy.sum(user_p[:, 0][:, None] - th <= 0, axis=0) / float(
                user_p.shape[0]
            )
            FAR = 1.0 - numpy.sum(user_n[:, 0][:, None] - th >= 0, axis=0) / float(
                user_n.shape[0]
            )
            EERs.append(getEER(FAR, FRR)[0] * 100)

        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        th = get_thresholds(datum_p, datum_n, score_norm)
        FRR = 1.0 - numpy.sum(datum_p[:, 0][:, None] - th <= 0, axis=0) / float(
            datum_p.shape[0]
        )
        FAR = 1.0 - numpy.sum(datum_n[:, 0][:, None] - th >= 0, axis=0) / float(
            datum_n.shape[0]
        )
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 1.0 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 1.0 / N_loops * datum_p.shape[0]

    TOTAL_P += datum_p.shape[0]
    TOTAL_N += datum_n.shape[0]

    return EER_G, EER_L, ROC_FAR, ROC_FRR, TOTAL_P, TOTAL_N


def verify_dataset_rf(
    feats,
    labels,
    inds,
    N_user,
    N_train,
    N_loops,
    weights,
    rf_index,
    score_norm="none",
    ds_subset=None,
    ds_ebio1_device=None,
):
    EER_G = []
    EER_L = []
    ROC_FAR = 0
    ROC_FRR = 0
    TOTAL_P = 0
    TOTAL_N = 0

    emb = feats["embedding"]
    rate = feats["spike_rate"]
    time = feats["spike_time"]

    N_samples = emb.shape[0] // N_user
    rfEmb = emb[rf_index::N_samples]
    rfRate = rate[rf_index::N_samples]
    rfTime = time[rf_index::N_samples]

    for rand in range(N_loops):
        EERs = []
        datum_p = []
        datum_n = []
        for k in numpy.arange(0, N_user):
            idxUser = numpy.where(labels == k)[0]
            embU = emb[idxUser]
            rateU = rate[idxUser]
            timeU = time[idxUser]
            indU = inds[idxUser]

            if ds_ebio1_device is not None:
                deviceIndP = (
                    numpy.array([0, 1, 10, 11, 20, 21, 30, 31]) + ds_ebio1_device * 2
                )
                deviceIndN = numpy.array([40, 41, 42, 55, 56, 57]) + ds_ebio1_device * 3
                embU = numpy.concatenate((embU[deviceIndP], embU[deviceIndN]), axis=0)
                rateU = numpy.concatenate(
                    (rateU[deviceIndP], rateU[deviceIndN]), axis=0
                )
                timeU = numpy.concatenate(
                    (timeU[deviceIndP], timeU[deviceIndN]), axis=0
                )
                indU = numpy.concatenate((indU[deviceIndP], indU[deviceIndN]), axis=0)

            if ds_subset is not None:
                embU = embU[ds_subset]
                rateU = rateU[ds_subset]
                timeU = timeU[ds_subset]
                indU = indU[ds_subset]

            idxGen = numpy.where(indU)[0]
            idxTemp = idxGen[0:4]
            if N_train == 1:
                tempEmb = embU[idxTemp[rand : rand + 1]]
                tempRate = rateU[idxTemp[rand : rand + 1]]
                tempTime = timeU[idxTemp[rand : rand + 1]]
            else:
                tempEmb = embU[idxTemp]
                tempRate = rateU[idxTemp]
                tempTime = timeU[idxTemp]

            testGIdx = numpy.array(sorted(set(idxGen) - set(idxTemp)))
            testGEmb = embU[testGIdx]
            testGRate = rateU[testGIdx]
            testGTime = timeU[testGIdx]
            testRFEmb = numpy.delete(rfEmb, k, axis=0)
            testRFRate = numpy.delete(rfRate, k, axis=0)
            testRFTime = numpy.delete(rfTime, k, axis=0)
            testEmb = numpy.concatenate((testGEmb, testRFEmb), axis=0)
            testRate = numpy.concatenate((testGRate, testRFRate), axis=0)
            testTime = numpy.concatenate((testGTime, testRFTime), axis=0)
            testInd = numpy.zeros(testEmb.shape[0])
            testInd[: testGEmb.shape[0]] = 1

            distTempEmb = compute_dist_mat(tempEmb, tempEmb)
            distTempRate = compute_dist_mat(tempRate, tempRate)
            distTempTime = compute_dist_mat(tempTime, tempTime)
            distEmb = compute_dist_mat(testEmb, tempEmb)
            distRate = compute_dist_mat(testRate, tempRate)
            distTime = compute_dist_mat(testTime, tempTime)
            embScore = compute_modality_score(distEmb, distTempEmb)
            rateScore = compute_modality_score(distRate, distTempRate)
            timeScore = compute_modality_score(distTime, distTempTime)

            score = compute_combined_score(
                embScore, rateScore, timeScore, weights, score_norm
            )

            datum_p.append(score[numpy.where(testInd)[0]])
            datum_n.append(score[numpy.where(1 - testInd)[0]])

            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = get_thresholds(user_p, user_n, score_norm)
            FRR = 1.0 - numpy.sum(user_p[:, 0][:, None] - th <= 0, axis=0) / float(
                user_p.shape[0]
            )
            FAR = 1.0 - numpy.sum(user_n[:, 0][:, None] - th >= 0, axis=0) / float(
                user_n.shape[0]
            )
            EERs.append(getEER(FAR, FRR)[0] * 100)

        EER_L.append(numpy.mean(EERs))
        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        th = get_thresholds(datum_p, datum_n, score_norm)
        FRR = 1.0 - numpy.sum(datum_p[:, 0][:, None] - th <= 0, axis=0) / float(
            datum_p.shape[0]
        )
        FAR = 1.0 - numpy.sum(datum_n[:, 0][:, None] - th >= 0, axis=0) / float(
            datum_n.shape[0]
        )
        EER_G.append(getEER(FAR, FRR)[0] * 100)
        ROC_FAR += FAR * 1.0 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 1.0 / N_loops * datum_p.shape[0]

    TOTAL_P += datum_p.shape[0]
    TOTAL_N += datum_n.shape[0]
    return EER_G, EER_L, ROC_FAR, ROC_FRR, TOTAL_P, TOTAL_N


def load_spike_file(path, TARGET_N):
    data = numpy.load(path)
    if data.shape != TARGET_N:
        if data.ndim == 2 and data.shape[0] == TARGET_N[0]:
            data = data.reshape(TARGET_N)
        else:
            data = numpy.ones(TARGET_N, dtype=numpy.float32) * 1e-6
    return data


def run_verification(args, weights, label=None):
    is_rf = args.mode == "rf"
    score_norm = args.score_norm
    layer_subset = args.layer_subset

    N_train = getattr(args, "references", 1)
    if N_train not in (1, 4):
        raise ValueError("references must be 1 or 4")
    N_loops = 1 if N_train == 4 else 4

    ROC_FAR_total = 0
    ROC_FRR_total = 0
    TOTAL_P_total = 0
    TOTAL_N_total = 0

    bsds2_subset = numpy.concatenate([numpy.array([0, 1, 2, 3]), numpy.arange(15, 50)])

    datasets = [
        ("mcyt", 100, 6, None, None),
        ("bsid", 132, 6, None, None),
        ("bsds2", 140, 5, bsds2_subset, None),
        ("ebio2", 35, 6, None, None),
        ("ebio1_w1", 35, 6, None, 0),
        ("ebio1_w2", 35, 6, None, 1),
        ("ebio1_w3", 35, 6, None, 2),
        ("ebio1_w4", 35, 6, None, 3),
        ("ebio1_w5", 35, 6, None, 4),
        ("msds", 121, 4, None, None),
        ("gpds", 300, 4, None, None),
    ]
    requested = {value.strip().lower() for value in args.datasets.split(",") if value.strip()}
    supported = {name.split("_")[0] for name, _, _, _, _ in datasets}
    unknown = requested.difference(supported)
    if not requested or unknown:
        raise ValueError("invalid --datasets values: %s" % sorted(unknown))
    datasets = [item for item in datasets if item[0].split("_")[0] in requested]
    if is_rf:
        datasets = [(n, u, r, s, e) for n, u, r, s, e in datasets if r is not None]

    if label is not None:
        print("Breakdown:", label, "weights=", weights)

    for dset_name, n_user, rf_index, ds_subset, ebio1_device in datasets:
        ds_key = dset_name.split("_")[0]
        EER_G_all = []
        EER_L_all = []
        for seed in args.seeds:
            epoch = str(args.epoch)
            featsEmb = numpy.load(
                "./exps/seed%d/%s/feats_epoch%s.npy" % (seed, ds_key, epoch)
            )
            embedding_normalization = getattr(args, "embedding_normalization", "legacy-head32")
            if embedding_normalization == "l2":
                featsEmb = l2_norm(featsEmb)
            elif embedding_normalization == "legacy-head32":
                featsEmb = load_and_norm(featsEmb)
            else:
                raise ValueError("unsupported embedding normalization")

            spikeRate = numpy.load(
                "./exps/seed%d/%s/spike_rate_epoch%s.npy" % (seed, ds_key, epoch)
            )
            meta_path = "./exps/seed%d/%s/spike_meta_epoch%s.npz" % (
                seed,
                ds_key,
                epoch,
            )
            spikeRate = maybe_select_spike_layers(
                spikeRate, meta_path, layer_subset, args.run_tag
            )
            spikeRate = l2_norm(spikeRate)
            spikeTime = numpy.load(
                "./exps/seed%d/%s/spike_time_epoch%s.npy" % (seed, ds_key, epoch)
            )
            spikeTime = maybe_select_spike_layers(
                spikeTime, meta_path, layer_subset, args.run_tag
            )
            spikeTime = l2_norm(spikeTime)

            labels = numpy.load(
                "./exps/seed%d/%s/labels_epoch%s.npy" % (seed, ds_key, epoch)
            )
            inds = numpy.load(
                "./exps/seed%d/%s/inds_epoch%s.npy" % (seed, ds_key, epoch)
            )
            actual_n_user = int(numpy.unique(labels).size)
            if actual_n_user <= 0:
                raise ValueError("dataset %s contains no user labels" % ds_key)

            feats = {
                "embedding": featsEmb,
                "spike_rate": spikeRate,
                "spike_time": spikeTime,
            }

            if is_rf:
                EER_G, EER_L, ROC_FAR, ROC_FRR, TOTAL_P, TOTAL_N = verify_dataset_rf(
                    feats,
                    labels,
                    inds,
                    actual_n_user,
                    N_train,
                    N_loops,
                    weights,
                    rf_index,
                    score_norm,
                    ds_subset,
                    ebio1_device,
                )
            else:
                EER_G, EER_L, ROC_FAR, ROC_FRR, TOTAL_P, TOTAL_N = verify_dataset_sf(
                    feats,
                    labels,
                    inds,
                    actual_n_user,
                    N_train,
                    N_loops,
                    weights,
                    score_norm,
                    ds_subset,
                    ebio1_device,
                )
            EER_G_all.extend(EER_G)
            EER_L_all.extend(EER_L)
            ROC_FAR_total += ROC_FAR
            ROC_FRR_total += ROC_FRR
            TOTAL_P_total += TOTAL_P
            TOTAL_N_total += TOTAL_N

        print(epoch, "~")
        print(
            "%s" % dset_name,
            "Global threshold:",
            numpy.mean(EER_G_all),
            "Local threshold:",
            numpy.mean(EER_L_all),
        )

    total_eer = (
        getEER(
            ROC_FAR_total * 1.0 / (TOTAL_N_total + 1e-8),
            ROC_FRR_total * 1.0 / (TOTAL_P_total + 1e-8),
        )[0]
        * 100
    )
    print("Total EER:", total_eer)
    return total_eer


def main(argv=None):
    args = build_parser().parse_args(argv)

    numpy.set_printoptions(threshold=1e6)
    numpy.random.seed(13)
    weights = tuple(args.weights)

    if args.run_tag:
        _orig_numpy_load = numpy.load

        def _load_with_run_tag(file, *load_args, **load_kwargs):
            if isinstance(file, str) and file.startswith("./exps/seed"):
                file = file.replace("./exps/", "./exps/%s/" % args.run_tag, 1)
            return _orig_numpy_load(file, *load_args, **load_kwargs)

        numpy.load = _load_with_run_tag

    if args.report_breakdown:
        for label, breakdown_weights in get_breakdown_weight_sets(weights):
            run_verification(args, breakdown_weights, label=label)
    else:
        run_verification(args, weights)


if __name__ == "__main__":
    main()
