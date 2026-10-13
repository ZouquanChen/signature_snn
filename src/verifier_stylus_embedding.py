"""Embedding-only verifier using the same protocol implementation as SNN-TVS."""

import argparse
import os

import numpy

import verifier_stylus_snn_tvs as tvs


def build_parser():
    parser = argparse.ArgumentParser(description="Generic embedding-only verifier")
    parser.add_argument("--epoch", default="End")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=["sf", "rf"], default="sf")
    parser.add_argument("--references", type=int, choices=[1, 4], default=1)
    parser.add_argument("--embedding-normalization", choices=["legacy-head32", "l2"],
                        default="legacy-head32",
                        help="integer-tcn uses whole-vector l2, not Sig2Vec 32-dimensional heads")
    parser.add_argument(
        "--datasets", default="mcyt,bsid,bsds2,ebio1,ebio2"
    )
    parser.add_argument("--score-norm", choices=["none", "zscore"], default="none")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # The TVS verifier only uses these additional attributes for optional views.
    args.layer_subset = "all"
    args.time_view = "global"
    args.report_breakdown = False

    original_load = numpy.load

    def load_with_embedding_placeholders(file, *load_args, **kwargs):
        if isinstance(file, str) and file.startswith("./exps/seed"):
            file = file.replace("./exps/", "./exps/%s/" % args.run_tag, 1)
        basename = os.path.basename(str(file))
        if basename.startswith(("spike_rate_", "spike_time_")):
            feature_path = os.path.join(
                os.path.dirname(str(file)), basename.replace("spike_rate_", "feats_").replace("spike_time_", "feats_")
            )
            features = original_load(feature_path)
            return numpy.zeros((features.shape[0], 1), dtype=numpy.float32)
        return original_load(file, *load_args, **kwargs)

    numpy.load = load_with_embedding_placeholders
    try:
        return tvs.run_verification(args, (1.0, 0.0, 0.0))
    finally:
        numpy.load = original_load


if __name__ == "__main__":
    main()
