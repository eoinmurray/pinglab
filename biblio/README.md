
## Getting Started

### Installation

First install [Task](https://taskfile.dev/docs/installation), then:

```bash
git clone https://github.com/eoinmurray/pinglab
task install
```

### Simulate

Then you can run experiments like:

```bash
task execute -- content/dynamical-regimes/experiment.py
# or
task execute:all
```

which will generate data in the appropriate folder.

### Present

Start the UI server with

```bash
task dev
# or
task bg # starts in background
```
