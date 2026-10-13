#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pathlib
import sys


def _ensure_sig2vec_module_path():
    sig2vec_dir = pathlib.Path(__file__).resolve().parent
    sig2vec_dir_str = str(sig2vec_dir)

    if sig2vec_dir_str not in sys.path:
        sys.path.insert(0, sig2vec_dir_str)


def _import_sig2vec_symbol(package_module_name, script_module_name, symbol_name):
    try:
        module = __import__(package_module_name, fromlist=[symbol_name])
    except ImportError:
        _ensure_sig2vec_module_path()
        module = __import__(script_module_name, fromlist=[symbol_name])
    return getattr(module, symbol_name)


def build_model(model_name, *args, **kwargs):
    kwargs.pop("model_name", None)

    if model_name == "integer-tcn-mamba2-interval":
        IntervalTCNMamba2 = _import_sig2vec_symbol(
            "fedburst_sig.src.network_interval_mamba", "network_interval_mamba", "IntervalTCNMamba2"
        )
        return IntervalTCNMamba2(*args, **kwargs)

    if model_name == "integer-tcn-mamba2":
        IntegerTCNMamba2 = _import_sig2vec_symbol(
            "fedburst_sig.src.network_integer_tcn_mamba", "network_integer_tcn_mamba", "IntegerTCNMamba2"
        )
        return IntegerTCNMamba2(*args, **kwargs)

    if model_name == "integer-tcn":
        IntegerTCN = _import_sig2vec_symbol(
            "fedburst_sig.src.network_integer_tcn", "network_integer_tcn", "IntegerTCN"
        )
        return IntegerTCN(*args, **kwargs)

    if model_name == "cnn":
        Sig2Vec = _import_sig2vec_symbol("SynSig2Vec.Sig2Vec.network", "network", "Sig2Vec")

        return Sig2Vec(*args, **kwargs)

    if model_name == "snn":
        Sig2VecSNN = _import_sig2vec_symbol("SynSig2Vec.Sig2Vec.network_snn", "network_snn", "Sig2VecSNN")

        return Sig2VecSNN(*args, **kwargs)

    raise ValueError("Unsupported model_name: %s" % model_name)
