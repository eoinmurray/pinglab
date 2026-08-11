#let meta = (
  title: "Gold star run 2",
  date: "2026-08-11",
  description: "The working checklist for refreshing the gamma-gated sparsity collection from the new exp022 checkpoint bank.",
  collection: "ephemeral",
  status: "draft",
)

#let divider() = context {
  if target() == "html" {
    html.elem("hr", attrs: (style: "margin: 3rem 0;"))
  } else {
    v(1.25em)
    line(length: 100%, stroke: 0.7pt + luma(65%))
    v(1.25em)
  }
}

#let task(done, id, body) = {
  let mark = if done { [☑] } else { [☐] }
  let state = if done { [Complete] } else { [Incomplete] }
  [#mark *#state* #raw(id) #body]
}

#let body = [
  Gold Star Run 2 refreshes the gamma-gated sparsity collection after exp022 produces its new checkpoint bank. It reruns affected experiments in dependency order, replaces exp048 with exp082, rebuilds the collection, and records enough provenance to recover the expensive source data.

  #divider()

  == 1. Freeze the refresh

  - #task(false, "G2.1", [Complete the exp022 preparation and production campaign in #link("ar065.html")[ar065]. Record its campaign ID, commit, checkpoint location, and R2 snapshot.])

  - #task(false, "G2.2", [Correct #link("ar017.html")[ar017]: remove exp048, make exp042 depend on exp022 and exp041, mark exp046 as consuming both, and use `.html` links.])

  - #task(false, "G2.3", [Record the pinned commit, selected experiments, execution order, expected outputs, and any experiments deliberately reused rather than rerun.])

  *Gate.* The exp022 bank is complete and the refresh list has been reviewed.

  #divider()

  == 2. Validate exp022

  - #task(false, "G2.4", [Check every expected cell, load representative checkpoints from TR-01--TR-05, and load all three TR-06 checkpoints.])

  - #task(false, "G2.5", [Rebuild and inspect exp022, including one training curve and representative raster per TR ID.])

  - #task(false, "G2.6", [Confirm the local checkpoint bank matches its R2 archive.])

  *Gate.* The exact checkpoint bank used below is valid, published in exp022, and recoverable.

  #divider()

  == 3. Run direct consumers

  These experiments can run after exp022 is accepted:

  - #task(false, "G2.7", [Run and inspect exp024, exp025, exp037, and exp038 from the refreshed TR-02 cells.])

  - #task(false, "G2.8", [Run and inspect exp041 from TR-03. Its accepted measurements gate the next stage.])

  - #task(false, "G2.9", [Run and inspect exp044 from TR-04 and exp049 from TR-05.])

  - #task(false, "G2.10", [Run and inspect exp082 from all three TR-06 cells, including matched inference, variable-rate and variable-duration inference, the 200 ms psychometric, and both planned figures.])

  *Gate.* Every direct consumer has complete outputs and no unexplained result change.

  #divider()

  == 4. Run the exp041 branch

  - #task(false, "G2.11", [After accepting exp041, run and inspect exp033, exp042, and exp046.])

  - #task(false, "G2.12", [Confirm those outputs identify the refreshed exp022 bank and accepted exp041 measurements they consume.])

  *Gate.* The downstream timing and mean-field branch is complete.

  #divider()

  == 5. Decide whether independent roots need rerunning

  - #task(false, "G2.13", [Assess exp023, exp047, exp054, exp080, and exp081. Rerun only those affected by shared-code changes; otherwise record a reuse decision.])

  *Gate.* Every independent root has either a new accepted run or a stated reason for reuse.

  #divider()

  == 6. Refresh and publish the collection

  - #task(false, "G2.14", [Remove exp048 from the active sequence and update ar009 to use and interpret exp082. Remove stale exp048 and exp065 artifact reads.])

  - #task(false, "G2.15", [Update ar017 to the final dependency map and review changed claims against the new figures and numbers.])

  - #task(false, "G2.16", [Run the test suite, build the complete Demolab collection, and inspect every changed page in the web UI.])

  - #task(false, "G2.17", [Archive expensive new source data where needed, verify the archives, and record the final commit, exceptions, and major scientific changes.])

  *Complete.* Gold Star Run 2 is finished when the refreshed collection builds cleanly, exp082 has replaced exp048, all affected experiments are accepted, reused roots are justified, and the source checkpoint bank is recoverable.
]
