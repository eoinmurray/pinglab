# exp159 — One-way entrainment

Compare one matched uncoupled/control and one A→B mean-field trajectory, with
feedback absent. Show frequency capture, finite-time phase locking and the timing
of incoming excitation, receiver inhibition and output in two columns.

## CLI

1. `compute.py`: optional `--difference-hz` (2 Hz/channel), `--coupling-us`
   (0.06 µS), `--run-id`. Runs only the 1,500 ms pair with fresh seed 85 weights.
   No upstream scientific inputs. Optional `--preserve-exploration PATH` retains
   an original exploration archive as a provenance attachment; it is never used
   to define or execute the new simulation and is unnecessary for reproduction.
2. `analyse.py`: required `--source COMPUTE_ID`; optional `--run-id`. Validates
   matched starts and sender invariance, measures locking over 800–1,500 ms and
   describes volley timing over 1,000–1,450 ms.
3. `present.py`: required `--source ANALYSE_ID`; optional `--run-id`. Writes the
   two-column figure as SVG/PNG/PDF with explicitly pinned analysis and compute.
4. `test.py`: `uv run pytest experiments/exp159/test.py -q`.

The selected settings were explored before this experiment was created. The
exploration archive retains all four attempts, unsuccessful outcomes, original
settings, native evidence, code dependencies and the previous figure. New runs
are generated independently; historical evidence is not relabelled as new work.
No probes, training, spiking validation or competition task is included.

Migration verification: both new trajectories matched every saved state in the
selected exploration exactly. All original non-cache exploration files were
byte-verified inside `exploration.zip` in the initial compute run before the
source directory was removed. Ten focused tests and the local report passed
with the original directory absent.

Refresh only the network diagram while retaining an explicit presentation's
scientific figure and numbers byte-for-byte:
`uv run python experiments/exp159/present.py --redraw-from PRESENT_ID`.
