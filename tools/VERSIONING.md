# Tool versioning

`snnlang`, `snnsim`, and `snnviz` have independent [Semantic Versioning](https://semver.org/) identities because experiment code and persisted results depend on their behaviour separately from the containing `pinglab` release.

- **Major**: incompatible public API, persisted format, or behavioural-contract change after `1.0.0`.
- **Minor**: backward-compatible functionality; before `1.0.0`, also any incompatible contract change.
- **Patch**: backward-compatible correction with no intended contract change.

Change only the affected tool's `_version.py`. Experiment manifests record all three versions, while Git provenance continues to identify the exact implementation. A version is therefore a compatibility label, not a replacement for the commit and dirty-patch record.
