#let meta = (
  title: "Scientific-contract audit of the gamma-gated publication campaign",
  date: "2026-08-12",
  description: "A bounded audit of exp022 and its sixteen downstream experiments before production compute.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp083/numbers.json")
#let plan = json("/artifacts/data/exp083/production-plan.json")
#let issue(number) = link("https://github.com/eoinmurray/pinglab/issues/" + str(number))[\##number]
#let body = [
  == Abstract

  We audited the scientific contract of the gamma-gated sparsity campaign before
  production compute. The review followed exp022's #r.scope.exp022_cell_count
  registered training cells and all #r.scope.experiment_count collection
  experiments through planning, model construction, checkpoint selection,
  inference, aggregation, provenance, and publication. The dry production plan
  created no scheduler jobs. The audit found #r.finding_counts.blocker blockers,
  #r.finding_counts.important important findings, and
  #r.finding_counts.minor minor finding. The campaign is not ready for production:
  checkpoint epochs can differ from the epochs claimed in the writings, and cell
  validation can accept mismatches in scientifically central parameters. Focused
  issues record each required implementation change.

  == Methods

  #enum(
    [*Freeze the reviewed source.* We audited commit #raw(r.audited_commit). The
      checkout was clean when the campaign manifest was generated, and the
      manifest records the same commit and the frozen lockfile hash.],

    [*Generate the production plan without submission.* We initialized a
      production-profile campaign in an external disposable directory and ran
      the collection `submit` command without `--live` or `--test-only`. This
      resolved cell lists, commands, dependencies, resources, and expected
      outputs but did not call `sbatch`. Private paths and resource identities
      were replaced with publication placeholders before the plan was retained.],

    [*Trace exp022 parameters.* For every cell, we followed the registry through
      `build_train_args`, the SNN command line, model and training construction,
      `config.json`, `metrics.json`, checkpoint files, campaign validation, and
      exp022's writing. We compared argument values within and across the six
      training-run families and checked that architecture-specific differences
      were registered.],

    [*Trace downstream contracts.* For each collection node, we reviewed declared
      dependencies, runner arguments, training checkpoint identity, inference
      mutations, sample selection, random seeds, cache behavior, aggregation
      unit, uncertainty summary, required outputs, collection provenance, and the
      corresponding writing.],

    [*Classify and disposition concerns.* A blocker can make production
      scientifically wrong, irreproducible, or materially misreported. An
      important finding is a meaningful ambiguity or provenance gap that must be
      resolved or explicitly documented. A minor finding cannot plausibly change
      a conclusion. A false alarm is an investigated contract that agrees across
      source, artifacts, and prose. Material implementation defects were not
      fixed in this audit; each received a focused issue with an acceptance test.],
  )

  == Exact production plan

  The dry plan contains #r.scope.exp022_cell_count training cells and
  #r.scope.planned_job_count scheduler job definitions. It creates five exp022
  arrays, one aggregation job, sixteen downstream jobs, and one finalization job.
  The command produced #r.scope.paid_compute_jobs_created paid-compute or scheduler
  jobs.

  #table(
    columns: (1.2fr, 0.8fr, 2.5fr),
    table.header([*Training run*], [*Cells*], [*Role*]),
    [TR-01], [#r.production_plan.training_runs.at("TR-01")], [Canonical COBA and PING reference],
    [TR-02], [#r.production_plan.training_runs.at("TR-02")], [Spike-budget sweep],
    [TR-03], [#r.production_plan.training_runs.at("TR-03")], [Inhibitory-timescale sweep],
    [TR-04], [#r.production_plan.training_runs.at("TR-04")], [Integration-timestep sweep],
    [TR-05], [#r.production_plan.training_runs.at("TR-05")], [Recurrent-initialization sweep],
    [TR-06], [#r.production_plan.training_runs.at("TR-06")], [Variable-rate streaming bank],
  )

  #table(
    columns: (1.4fr, 0.7fr, 2.4fr),
    table.header([*Resource tier*], [*Cells*], [*Scientific shape*]),
    [Standard], [#r.production_plan.tiers.standard], [Sweep cells at the standard timestep],
    [Fine timestep], [#r.production_plan.tiers.fine_dt], [TR-04 cells at the smallest timestep],
    [Canonical COBA], [#r.production_plan.tiers.canonical_coba], [Full pooled-MNIST COBA cells],
    [Canonical PING], [#r.production_plan.tiers.canonical_ping], [Full pooled-MNIST PING cells],
    [Variable rate], [#r.production_plan.tiers.variable_rate], [TR-06 output-LIF cells],
  )

  The preserved `production-plan.json` contains the complete sanitized collection
  plan, all exp022 cell commands, and all dry submission commands. Its campaign
  source is #raw(r.production_plan.source_commit). The recorded exp022 manifest
  and resource-file SHA-256 prefixes are
  #raw(r.production_plan.manifest_sha256.slice(0, 12)) and
  #raw(r.production_plan.resource_file_sha256.slice(0, 12)); the preserved plan
  contains both complete hashes.

  == Reviewed execution graph

  #table(
    columns: (0.75fr, 1.2fr, 0.8fr, 2fr),
    table.header([*Experiment*], [*Dependencies*], [*Training run*], [*Scheduled arguments*]),
    ..r.inventory.map(row => (
      [#row.experiment],
      [#if row.dependencies.len() == 0 { [root] } else { row.dependencies.join(", ") }],
      [#if row.training_run == none { [none] } else { row.training_run }],
      [#if row.runner_arguments.len() == 0 { [none] } else { row.runner_arguments.join(" ") }],
    )).flatten(),
  )

  == Findings

  === Blockers

  #for finding in r.findings.filter(item => item.severity == "blocker") [
    ==== #finding.id: #finding.title

    *Observed behavior.* #finding.observed

    *Expected scientific meaning.* #finding.expected

    *Publication risk.* #finding.risk

    *Disposition.* #finding.resolution #if finding.issue != none { [Tracked in #issue(finding.issue).] }

    *Code evidence.* #finding.evidence.map(item => item.path + ":" + str(item.line) + " (`" + item.symbol + "`)").join("; ").
  ]

  === Important findings

  #for finding in r.findings.filter(item => item.severity == "important") [
    ==== #finding.id: #finding.title

    *Observed behavior.* #finding.observed

    *Expected scientific meaning.* #finding.expected

    *Publication risk.* #finding.risk

    *Disposition.* #finding.resolution #if finding.issue != none { [Tracked in #issue(finding.issue).] }

    *Code evidence.* #finding.evidence.map(item => item.path + ":" + str(item.line) + " (`" + item.symbol + "`)").join("; ").
  ]

  === Minor finding

  #for finding in r.findings.filter(item => item.severity == "minor") [
    ==== #finding.id: #finding.title

    #finding.observed #finding.risk #finding.resolution

    *Code evidence.* #finding.evidence.map(item => item.path + ":" + str(item.line) + " (`" + item.symbol + "`)").join("; ").
  ]

  == Checked contracts that held

  #for item in r.false_alarms [
    + *#item.contract.* #item.evidence
  ]

  == Accepted limitations

  #for item in r.accepted_limitations [
    + #item
  ]

  == Validation

  Focused campaign, argument, isolation, and smoke-cap tests reported
  #r.validation.focused_tests. The exact command is retained in `numbers.json`.
  The audit runner rechecked the captured commit, clean-manifest flag, production
  profile, #r.scope.exp022_cell_count cells, #r.scope.planned_job_count jobs in the
  dry plan, JSON validity, and a fail-closed sensitive-data scan. A complete
  Demolab build validates this article and its artifact references. No production
  training, Wilkes3 submission, RunPod job, Modal job, or other paid compute was
  run.

  == Conclusion

  The collection graph and most reproducibility boundaries are coherent, but the
  campaign must not be frozen for production while #issue(74) and #issue(75)
  remain unresolved. Issues #issue(76), #issue(77), #issue(78), and #issue(79)
  must also be resolved or explicitly accepted before publication. The internal
  pooled-MNIST split should be described as checkpoint-selection holdout data,
  with its optimizer-training and holdout counts stated separately. Once the two
  blockers and the important follow-ups have evidence-backed dispositions, this
  audit can be rerun against the new frozen commit and the production readiness
  decision reconsidered.
]

#body
