"""Learnable-interval signed counts at selected Mamba projection boundaries.

The SSM and its control coefficients stay continuous. This is an activation
adaptation experiment, not full integer arithmetic or full spiking Mamba.
"""

from functools import partial

try:
    from .learnable_interval_spike import LearnableIntervalSpike, parse_thresholds
    from .mamba2_torch import TorchMamba2, Mamba2ResidualBlock
    from .network_integer_tcn_mamba import IntegerTCNMamba2
except ImportError:
    from learnable_interval_spike import LearnableIntervalSpike, parse_thresholds
    from mamba2_torch import TorchMamba2, Mamba2ResidualBlock
    from network_integer_tcn_mamba import IntegerTCNMamba2


class IntervalMamba2(TorchMamba2):
    def __init__(self, *args, interval_options, levels, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval_sites = interval_options["interval_sites"]
        neuron_options = dict(
            levels=levels, initial_thresholds=interval_options["interval_init_thresholds"],
            gap_min=interval_options["interval_gap_min"], gap_max=interval_options["interval_gap_max"],
            surrogate_width=interval_options["interval_surrogate_width"],
            channelwise=interval_options["interval_channelwise"], signed=True,
            fixed=interval_options["interval_fixed"],
            input_gradient=interval_options["interval_input_gradient"],
        )
        self.input_spike = (LearnableIntervalSpike(self.in_projection.in_features, **neuron_options)
                            if self.interval_sites in ("input", "both") else None)
        self.output_spike = (LearnableIntervalSpike(self.inner, **neuron_options)
                             if self.interval_sites in ("output", "both") else None)

    def _encode_input(self, sequence, mask):
        if self.input_spike is None:
            return sequence
        return self.input_spike(sequence.transpose(1, 2), mask).transpose(1, 2)

    def _encode_output(self, sequence, mask):
        if self.output_spike is None:
            return sequence
        return self.output_spike(sequence.transpose(1, 2), mask).transpose(1, 2)


class IntervalTCNMamba2(IntegerTCNMamba2):
    def __init__(self, n_in, n_classes, interval_sites="both", interval_init_thresholds=None,
                 interval_gap_min=.05, interval_gap_max=2., interval_surrogate_width=1.,
                 interval_channelwise=False, interval_fixed=False, interval_tcn=False,
                 interval_input_gradient="normalized-triangle", **kwargs):
        if interval_sites not in ("input", "output", "both"):
            raise ValueError("interval_sites must be input, output or both")
        levels = kwargs.get("integer_max", 4)
        self.interval_config = dict(
            interval_sites=interval_sites,
            interval_init_thresholds=list(parse_thresholds(interval_init_thresholds, levels)),
            interval_gap_min=float(interval_gap_min), interval_gap_max=float(interval_gap_max),
            interval_surrogate_width=float(interval_surrogate_width),
            interval_channelwise=bool(interval_channelwise), interval_fixed=bool(interval_fixed),
            interval_tcn=bool(interval_tcn),
            interval_input_gradient=str(interval_input_gradient),
        )
        super().__init__(n_in, n_classes, **kwargs)
        if self.interval_config["interval_tcn"]:
            # Replace the eight existing TCN quantizers after base construction;
            # all convolution/normalization/residual weights keep their exact
            # initialization and the same forward mask contract.
            for block in self.blocks:
                tcn_channels = int(block.residual_scale.shape[1])
                block.spike1 = LearnableIntervalSpike(tcn_channels, **{
                    "levels": levels, "initial_thresholds": self.interval_config["interval_init_thresholds"],
                    "gap_min": self.interval_config["interval_gap_min"], "gap_max": self.interval_config["interval_gap_max"],
                    "surrogate_width": self.interval_config["interval_surrogate_width"],
                    "channelwise": self.interval_config["interval_channelwise"], "signed": True,
                    "fixed": self.interval_config["interval_fixed"],
                    "input_gradient": self.interval_config["interval_input_gradient"],
                })
                block.spike2 = LearnableIntervalSpike(tcn_channels, **{
                    "levels": levels, "initial_thresholds": self.interval_config["interval_init_thresholds"],
                    "gap_min": self.interval_config["interval_gap_min"], "gap_max": self.interval_config["interval_gap_max"],
                    "surrogate_width": self.interval_config["interval_surrogate_width"],
                    "channelwise": self.interval_config["interval_channelwise"], "signed": True,
                    "fixed": self.interval_config["interval_fixed"],
                    "input_gradient": self.interval_config["interval_input_gradient"],
                })

    def _make_mamba_block(self, **options):
        factory = partial(IntervalMamba2, interval_options=self.interval_config, levels=self.integer_max)
        return Mamba2ResidualBlock(**options, mixer_factory=factory)

    @property
    def spike_neurons(self):
        # All hooked nodes expose [B,C,L], including the transposed Mamba sites.
        interval_nodes = tuple(node for block in self.mamba_blocks
                               for node in (block.mixer.input_spike, block.mixer.output_spike)
                               if node is not None)
        return super().spike_neurons + interval_nodes

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(version=3, model_name="integer-tcn-mamba2-interval",
                     interval_backend="signed-ordered-counts-v1",
                     interval_gradient=("clipped-input-ste-plus-normalized-threshold-triangle-v1"
                                        if self.interval_config["interval_input_gradient"] == "clipped-ste"
                                        else "normalized-triangle-v1"), **self.interval_config)
        return state

    def set_extra_state(self, state):
        """Load legacy v3 checkpoints created before interval gradient metadata.

        Older checkpoints omit ``interval_tcn`` and ``interval_input_gradient``;
        those mean the original Mamba-only, normalized-triangle behavior.
        Other architecture fields remain strictly checked.
        """
        saved = dict(state)
        saved.setdefault("interval_tcn", False)
        saved.setdefault("interval_input_gradient", "normalized-triangle")
        saved.pop("interval_gradient", None)
        current = dict(self.get_extra_state())
        current.pop("interval_gradient", None)
        if saved != current:
            raise ValueError("interval-Mamba checkpoint architecture/quantizer mismatch")

    @classmethod
    def from_state_dict_config(cls, state_dict, n_in):
        config = dict(state_dict.get("_extra_state", {}))
        expected = dict(version=3, model_name="integer-tcn-mamba2-interval",
                        mamba_backend="torch-ssd-prefix-v1", sim_steps=1,
                        spike_gradient="clipped-ste", interval_backend="signed-ordered-counts-v1")
        for name, value in expected.items():
            if config.pop(name, None) != value:
                raise ValueError("missing or unsupported interval-Mamba metadata: " + name)
        gradient = config.pop("interval_gradient", "normalized-triangle-v1")
        if gradient not in ("normalized-triangle-v1", "clipped-input-ste-plus-normalized-threshold-triangle-v1"):
            raise ValueError("unsupported interval gradient metadata")
        if config.get("n_in") != n_in:
            raise ValueError("evaluation feature dimension differs from training")
        return cls(**config)

    def interval_diagnostics(self):
        return {name: dict(intervals=node.intervals().detach().cpu().tolist(),
                           thresholds=node.thresholds().detach().cpu().tolist(),
                           fixed=node.fixed, channelwise=node.channelwise)
                for name, node in self.named_modules() if isinstance(node, LearnableIntervalSpike)}
