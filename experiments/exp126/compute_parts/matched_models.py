"""Matched dimensionless current-based LIF through two independent libraries."""

import snntorch as snn
import torch
from snnlab import extensions, lang
from snnlab.sim.execution import GraphExecutor, plan_graph
from snntorch import surrogate
from torch import nn

NEURON = "pinglab.matched_synaptic/v1"
SURROGATE = "pinglab.snntorch_fast_sigmoid/v1"
PARAMETERS = ("input_weight", "input_bias", "output_weight", "output_bias")


def initialize(context):
    zero = torch.zeros(context.shape, device=context.device, dtype=context.dtype)
    return {
        "voltage": zero,
        "synaptic": zero.clone(),
        "refractory": torch.zeros(
            context.shape, device=context.device, dtype=torch.long
        ),
    }


def neuron_step(context):
    cfg, state = context.config, context.state
    # snnTorch determines a detached reset from the previous membrane state.
    reset = (state["voltage"] > cfg["threshold"]).to(state["voltage"].dtype).detach()
    synaptic = (
        cfg["alpha"] * state["synaptic"] + context.excitatory - context.inhibitory
    )
    voltage = cfg["beta"] * state["voltage"] + synaptic - reset * cfg["threshold"]
    spikes = context.spike(voltage - cfg["threshold"])
    return dict(
        voltage=voltage, synaptic=synaptic, refractory=state["refractory"]
    ), spikes


def derivative(value, config):
    return 1 / (1 + config["slope"] * value.abs()).square()


def register():
    extensions.register_neuron(
        NEURON,
        neuron_step,
        initialize=initialize,
        input_unit="nA",
        state_units={"synaptic": "nA"},
    )
    extensions.register_surrogate(SURROGATE, derivative)


def initial_parameters(cfg, seed):
    torch.manual_seed(seed)
    first = nn.Linear(cfg["inputs"], cfg["hidden"])
    second = nn.Linear(cfg["hidden"], cfg["outputs"])
    return dict(
        input_weight=first.weight.detach().T.contiguous(),
        input_bias=torch.full((1, cfg["hidden"]), 0.1),
        output_weight=second.weight.detach().T.contiguous(),
        output_bias=torch.full((1, cfg["outputs"]), 0.1),
    )


class TorchModel(nn.Module):
    def __init__(self, cfg, initial):
        super().__init__()
        self.cfg = cfg
        self.weights = nn.ParameterDict(
            {key: nn.Parameter(initial[key].clone()) for key in PARAMETERS}
        )
        options = dict(
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            spike_grad=surrogate.fast_sigmoid(slope=cfg["surrogate_slope"]),
            reset_mechanism="subtract",
            reset_delay=True,
        )
        self.hidden = snn.Synaptic(**options)
        self.output = snn.Synaptic(**options)

    def canonical(self):
        return self.weights

    def forward(self, events, record=False):
        cfg = self.cfg
        hs = events.new_zeros((events.shape[1], cfg["hidden"]))
        hm = torch.zeros_like(hs)
        os = events.new_zeros((events.shape[1], cfg["outputs"]))
        om = torch.zeros_like(os)
        counts = torch.zeros_like(os)
        ones = events.new_ones((events.shape[1], 1))
        traces = {
            key: []
            for key in (
                "hidden_synaptic",
                "hidden_voltage",
                "hidden_spikes",
                "output_synaptic",
                "output_voltage",
                "output_spikes",
            )
        }
        for inputs in events:
            h, hs, hm = self.hidden(
                inputs @ self.weights["input_weight"]
                + ones @ self.weights["input_bias"],
                hs,
                hm,
            )
            out, os, om = self.output(
                h @ self.weights["output_weight"] + ones @ self.weights["output_bias"],
                os,
                om,
            )
            counts = counts + out
            if record:
                for key, value in zip(traces, (hs, hm, h, os, om, out), strict=True):
                    traces[key].append(value)
        return counts, {
            key: torch.stack(values) for key, values in traces.items()
        } if record else {}


class LabModel(nn.Module):
    def __init__(self, cfg, initial):
        super().__init__()
        register()
        net = lang.Network("matched_current_lif", dt=cfg["dt_ms"] * lang.ms)
        inputs = net.input(
            "pixels",
            shape=("time", "batch", cfg["inputs"]),
            signal_type="spikes",
            unit="spike",
        )
        bias = net.input(
            "bias", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
        )
        options = {key: cfg[key] for key in ("alpha", "beta", "threshold")}
        hidden = net.population(
            "hidden", size=cfg["hidden"], neuron=lang.CustomNeuron(NEURON, **options)
        )
        output = net.population(
            "output", size=cfg["outputs"], neuron=lang.CustomNeuron(NEURON, **options)
        )
        parameters = {}
        for key, value in initial.items():
            parameters[key] = net.parameter(
                key,
                shape=tuple(reversed(value.shape)),
                unit="nA",
                initializer=lang.Constant(0),
                initialization_scaling="direct",
            )
        # Zero projection decay: the custom neuron owns the single synaptic filter.
        # Positive tau is required; exp(-1000) underflows to exactly zero.
        for name, source, target, key in (
            ("hidden_weights", inputs, hidden, "input_weight"),
            ("hidden_bias", bias, hidden, "input_bias"),
            ("output_weights", hidden.spikes, output, "output_weight"),
            ("output_bias_projection", bias, output, "output_bias"),
        ):
            net.connect(
                source,
                target.excitatory,
                name=name,
                synapse=lang.ExponentialCurrent(tau=0.001 * lang.ms),
                weight=parameters[key],
                delay=0 * lang.ms,
            )
        net.output(
            "counts",
            lang.ops.reduce(output.spikes, operation="sum", over="time", name="count"),
        )
        for prefix, population in (("hidden", hidden), ("output", output)):
            net.expose(population.state("synaptic"), name=prefix + "_synaptic")
            net.expose(population.voltage, name=prefix + "_voltage")
            net.expose(population.spikes, name=prefix + "_spikes")
        graph = lang.compile(net, target="tools/snnsim").graph
        self.executor = GraphExecutor(
            plan_graph(graph),
            trainable_parameters=PARAMETERS,
            surrogate=lang.training.CustomSurrogate(
                SURROGATE, slope=cfg["surrogate_slope"]
            ).json(),
        )
        if self.executor.extension_validation != cfg.get(
            "snnlab_extension_validation", "structural"
        ):
            raise ValueError("snnlab default validation policy differs from the recipe")
        with torch.no_grad():
            for key, value in initial.items():
                self.executor.weights[key].copy_(value)

    def canonical(self):
        return self.executor.weights

    def forward(self, events, record=False):
        result = self.executor(
            {"pixels": events, "bias": events.new_ones((*events.shape[:2], 1))},
            diagnostics=record,
        )
        return result.outputs["counts"], result.diagnostics if record else {}


def make_model(implementation, cfg, initial, device):
    return (TorchModel if implementation == "snntorch" else LabModel)(cfg, initial).to(
        device
    )
