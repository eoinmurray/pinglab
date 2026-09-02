# snnsim tests

Tests are grouped by the subsystem they exercise, not by an ordered assurance
level. The directory names answer **what is under test**; pytest markers answer
**how the test runs**.

Keep a test module focused on one public API or coherent behaviour. Split it
when independent behaviours accumulate, normally before it grows beyond roughly
500 lines. Several closely related tests in one module are expected. Use test
classes only when the class gives those tests a meaningful behavioural name or
shared setup.

The registered cross-cutting markers are:

- `integration`: crosses module or process boundaries.
- `brian2`: compares against the independent Brian2 simulator.
- `accelerator`: requires or validates MPS or CUDA behaviour.
- `slow`: belongs outside the fast iteration lane.
- `regression`: checks against calibrated artifacts.

The normal fast lane is:

```sh
uv run pytest tools/snnsim/tests -m "not slow"
```

Markers compose, so the Brian2 comparisons can be selected with
`-m brian2`, while all currently marked cross-module tests can be selected with
`-m integration`.
