"""Sequential E/I perturbation draws on native emitted spikes.

The native stateless DropSpikes/AddPoissonSpikes streams are different from this
experiment's paired sequential device stream. Preview a step's raw spikes, then
replay its perturbed emissions from the same pre-step state. Only the replayed
step is committed; neuron reset follows the raw spike, while recurrent delays
and the readout receive transmitted spikes. No legacy simulator is constructed.
"""

from dataclasses import replace

import torch
from snnlab.sim.interventions import DenseSpikeReplay, ReplaySpikes


def perturb_forward(generator):
    def forward(model, drive, request):
        state = None
        totals = {
            key: {
                field: 0
                for field in (
                    "raw_spikes",
                    "transmitted_spikes",
                    "inserted_spikes",
                    "deleted_spikes",
                    "slots",
                )
            }
            for key in ("e1", "i1")
        }
        counts = {}
        score_sum = None
        recorded = {"spk_e": [], "spk_i": []}
        mode, level = request["perturb_mode"], request["perturb_level"]
        probability = (
            level
            if mode == "drop"
            else level * model.plan.graph["timebase"]["dt"]["value"] / 1000
        )
        if not 0 <= probability <= 1:
            raise ValueError("invalid perturbation probability")
        for step in range(len(drive)):
            inputs = {"drive": drive[step : step + 1]}
            raw = model(inputs, runtime_state=state, diagnostics=True)
            interventions = []
            for label in ("e", "i"):
                before = raw.diagnostics[f"spk_{label}"]
                draw = torch.rand(
                    before.shape[1:], device=before.device, generator=generator
                )
                after = (
                    before * (draw >= probability)
                    if mode == "drop"
                    else torch.maximum(before, (draw < probability).to(before.dtype))
                )
                interventions.append(
                    ReplaySpikes(
                        label.upper(),
                        DenseSpikeReplay.from_tensor(
                            after,
                            start_step=step,
                            dt_ms=model.plan.graph["timebase"]["dt"]["value"],
                        ),
                    )
                )
                row = totals[f"{label}1"]
                row["raw_spikes"] += int(before.sum())
                row["transmitted_spikes"] += int(after.sum())
                row["inserted_spikes"] += int((after > before).sum())
                row["deleted_spikes"] += int((after < before).sum())
                row["slots"] += before.numel()
            result = model(
                inputs,
                runtime_state=state,
                diagnostics=True,
                interventions=tuple(interventions),
            )
            state = result.runtime_state
            score = result.outputs["class_scores"]
            score_sum = score if score_sum is None else score_sum + score
            for label in ("e", "i"):
                key = f"spk_{label}_count"
                value = result.outputs[key]
                counts[key] = value if key not in counts else counts[key] + value
                if request["input"] == "snapshot":
                    recorded[f"spk_{label}"].append(result.diagnostics[f"spk_{label}"])
        diagnostics = (
            {key: torch.cat(value) for key, value in recorded.items()}
            if request["input"] == "snapshot"
            else {}
        )
        return replace(
            result,
            outputs={**counts, "class_scores": score_sum / len(drive)},
            diagnostics=diagnostics,
        ), totals

    return forward
