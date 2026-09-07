# Local publication

Production is built on a machine with operational v4 Pingstore data. GitHub Pages
serves `gh-pages` from `/`; it does not rebuild the writings. The retired production
and PR preview workflows are retained here with `.disabled` extensions.

```sh
task build-publication
task preview-publication # http://localhost:3001; Ctrl-C when finished
task publish
```

Requires the project environment (`uv sync`), Typst, Task, Git, and authenticated
Git push access to `origin`. Pages must already use `gh-pages` from `/`.

The build reads the current working tree, including uncommitted authored changes.
It freezes the most recently completed present run per required experiment, with
`writings/run-defaults.json` pins taking precedence. Completion time comes from
`run.json`; ties use run ID. Missing, invalid, or mismatched inputs fail the build.
Only selected runs and their full ancestry undergo payload validation. Discovery's
ordinary all-run contract is unchanged.

An isolated build workspace receives complete copies of selected present runs;
compute and analyse payloads stay in the original store. Copies are validated
again. A frozen presentation projection supplies article defaults and provenance.
Demolab compiles figures and data into articles and copies declared media. The
published output does not contain the store or build workspace.

The successful output is `.demolab/publication/site/`, with a local build receipt
at `.demolab/publication/build.json` recording selections, source status, and file
checksums. This is disposable publication scratch, not a Pingstore catalogue or
provenance authority. Failed builds preserve the previous output.

`task publish` checks every site file against that receipt and pushes that exact
build, without rebuilding. It preserves the remote `CNAME` and `pr-preview/` and
rejects concurrent branch updates through a normal Git push. No force push is
used. Existing preview directories remain; new PR previews are disabled.

Inspect the local publication before publishing. Subsequent source changes do not
change the frozen output; rebuild to include them. Publication does not commit or
push source changes. GitHub Pages completes deployment asynchronously after the
branch push.
