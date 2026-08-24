# Live Manim chat prototype

This proof of concept keeps an OpenGL Manim window open while `state.json`
changes. From the repository root:

```bash
uv venv temp/manim-live-venv --python 3.12
uv pip install --python temp/manim-live-venv/bin/python manim ipython
temp/manim-live-venv/bin/manim --renderer=opengl -p \
  tools/manim_live_demo/live_shape.py LiveShape
```

In another terminal—or through the chat agent—change the object:

```bash
python3 tools/manim_live_demo/set_state.py --shape star --color red --size 2
```

The current prototype supports `circle`, `square`, `triangle`, and `star`; the
colours are `blue`, `green`, and `red`; size must be between `0.1` and `5`.
Close the Manim window or enter `exit` in its IPython prompt to finish.
