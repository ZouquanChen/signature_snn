import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import verify_fusion_matched as matched


@pytest.fixture
def exports():
    rng = np.random.default_rng(10)
    data = {}
    for subset, genuine, forgery in [("mcyt",25,25), ("bsid",16,12), ("bsds2",30,20),
                                     ("ebio1",40,30), ("ebio2",8,6)]:
        labels = np.repeat(np.arange(2), genuine + forgery)
        inds = np.tile(np.r_[np.ones(genuine), np.zeros(forgery)], 2)
        emb = matched.legacy.l2_norm(rng.normal(size=(len(labels), 8)).astype(np.float32))
        data[subset] = (emb, labels, inds, None)
    return data


@pytest.mark.parametrize("references", [1,4])
def test_queries_and_every_score_and_pooled_match_original(exports, references):
    rows = matched.audit_embedding(exports, references)
    value = matched.pooled(rows, "embedding", references)
    loops = 4 if references == 1 else 1
    ps, ns = 0., 0.
    fars, frrs = np.zeros(5000), np.zeros(5000)
    for view, (subset, device) in enumerate(matched.VIEWS):
        emb, labels, inds, _ = exports[subset]
        zero = np.zeros((len(emb),1), dtype=np.float32)
        result = matched.legacy.verify_dataset_sf(
            dict(embedding=emb, spike_rate=zero, spike_time=zero), labels, inds,
            2, references, loops, (1.,0.,0.), "none",
            np.r_[0:4,15:50] if subset == "bsds2" else None, device)
        _, _, far, frr, p, n = result
        fars += far; frrs += frr; ps += p; ns += n
        for loop, writer, refs, queries in matched.episodes(labels, inds, subset, device, references):
            ids = matched.view_indices(np.flatnonzero(labels == writer), subset, device)
            assert not set(queries) & set(ids[inds[ids] == 1][:4])
            if subset == "bsds2":
                assert not set(queries) & set(np.flatnonzero(labels == writer)[4:15])
            if device is not None:
                assert len(queries) == 10
    oracle = matched.legacy.getEER(fars / (ns + 1e-8), frrs / (ps + 1e-8))[0] * 100
    assert value["eer_percent"] == pytest.approx(oracle, abs=1e-12)
    assert value["genuine_trials"] == ps * loops
    assert value["forgery_trials"] == ns * loops


def test_curves_exact_ties_and_original_grid():
    p = np.array([0., .001, .002, 2., 5.])
    n = np.array([0., .001, .002, 3., 6.])
    far, frr = matched.curves(p,n)
    th = matched.legacy.get_thresholds(p[:,None], n[:,None], "none")
    np.testing.assert_array_equal(far, 1-np.sum(n[:,None]-th >= 0,axis=0)/len(n))
    np.testing.assert_array_equal(frr, 1-np.sum(p[:,None]-th <= 0,axis=0)/len(p))


def test_dtw_self_diagonal_and_reference():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(8,4)); a /= np.linalg.norm(a,axis=1,keepdims=True)
    b = rng.normal(size=(8,4)); b /= np.linalg.norm(b,axis=1,keepdims=True)
    assert matched.dtw(a,a,3) == pytest.approx(0., abs=1e-14)
    expected = np.maximum(0.,1-np.sum(a*b,axis=1)).mean()
    assert matched.dtw(a,b,0) == pytest.approx(expected)
    assert matched.dtw(a,b,3) == pytest.approx(matched.dtw(b,a,3))
    # Reference cumulative-cost DP, including the same explicit tie convention.
    cost = np.full((9,9),np.inf); cost[0,0]=0.
    length = np.zeros((9,9),int)
    for i in range(1,9):
        for j in range(max(1,i-3), min(8,i+3)+1):
            prev = min([(i-1,j-1),(i-1,j),(i,j-1)], key=lambda x: cost[x])
            cost[i,j] = cost[prev]+max(0.,1-np.dot(a[i-1],b[j-1]))
            length[i,j] = length[prev]+1
    assert matched.dtw(a,b,3) == pytest.approx(cost[8,8]/length[8,8])


def test_local_normalization_after_interpolation(tmp_path):
    raw = np.array([[[1.,0.],[0.,1.]]],np.float32)
    np.save(tmp_path/'seq_feat_epochX.npy',raw)
    np.save(tmp_path/'seq_mask_epochX.npy',np.ones((1,2),bool))
    features = matched.prepare_local(tmp_path, 'X', 5)[0]
    np.testing.assert_allclose(np.linalg.norm(features,axis=1),1.)
    assert matched.dtw(features,features,2) == pytest.approx(0.,abs=1e-14)
