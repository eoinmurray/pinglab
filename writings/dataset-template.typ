// Article-owned dataset template: presentation inputs and the rendered Dataset section.
// URL renders and prepared builds read Pinglab's own validated projection.
// Legacy paths remain below until the separate retirement step is approved.
#let prepared = "demolab-url-render" in sys.inputs or "demolab-bundle-root" in sys.inputs
#let catalogue = if prepared { json("/.demolab/pinglab-inputs.json") } else {
  (articles: (:), defaults: (:), runs: ())
}
#let selected-run(article, key) = {
  let available = catalogue.runs.filter(run => run.experiment == key.split(".").last())
  let parameter = "source." + key
  if sys.inputs.at("demolab-url-article", default: "") == article and parameter in sys.inputs {
    let selected = available.filter(run => run.basepath == sys.inputs.at(parameter))
    assert(selected.len() == 1, message: "URL input is not a validated presentation: " + key)
    return selected.first()
  }
  let pinned = catalogue.defaults.at(article, default: (:)).at(key, default: none)
  if pinned != none {
    let selected = available.filter(run => run.id == pinned)
    assert(selected.len() == 1, message: "Unavailable default presentation: " + key)
    return selected.first()
  }
  if available.len() == 0 { none } else { available.first() }
}

#let media-extensions = ("mp4", "webm", "ogg", "ogv", "mov", "m4v")
#let media-url(run, filename) = "_pinglab-media/" + run.id + "/" + filename
#let input-assets(article, inputs) = {
  let assets = (:)
  if prepared {
    for key in inputs {
      let run = selected-run(article, key)
      if run != none {
        for file in run.files.filter(file => lower(file.split(".").last()) in media-extensions) {
          assets.insert(media-url(run, file), run.basepath + "/" + file)
        }
      }
    }
  }
  assets
}

#import "/.demolab/lib.typ" as engine
#let video(source, ..args) = {
  let url = source
  if prepared and source != none {
    let matches = catalogue.runs.filter(run => source.starts-with(run.basepath + "/"))
    assert(matches.len() == 1, message: "video requires a selected presentation file")
    url = media-url(matches.first(), source.split("/").last())
  }
  engine.video(url, ..args)
}
// Preview supplies article/key selections; publication supplies a fixed inventory.
// No input is a normal empty state. A selected input's missing files are errors.
#let preview = "demolab-preview-file" in sys.inputs
#let inventory = if not preview and "demolab-data-inputs" in sys.inputs {
  json(sys.inputs.at("demolab-data-inputs"))
} else { (:) }
#let selections = if preview {
  json(sys.inputs.at("demolab-preview-file"))
} else { inventory.at("sources", default: (:)) }

#let data-file(rel, article: none) = {
  assert(article != none, message: "run inputs require an article binding")
  let parts = rel.split("/")
  assert(parts.all(part => part not in ("", ".", "..")) and not rel.contains("\\"),
    message: "run inputs require a safe data key or key/filename")
  let key = parts.first()
  if prepared and not preview {
    assert(key in catalogue.articles.at(article, default: ()), message: "undeclared article input: " + key)
    let run = selected-run(article, key)
    if run == none { return none }
    if parts.len() == 1 { return run.basepath }
    let filename = parts.slice(1).join("/")
    assert(filename in run.files, message: "missing selected data file: " + rel)
    return run.basepath + "/" + filename
  }
  let selected = selections.at(article, default: (:))
  if not preview and article in selections {
    assert(key in selected,
      message: "build.sources." + article + " has no pin for data key '" + key + "'")
  }
  let directory = selected.at(key, default: none)
  if directory == none { return none }
  // A bare key checks availability without requiring an arbitrary numbers.json.
  if parts.len() == 1 { return directory }
  let path = directory + rel.slice(key.len())
  if not preview {
    assert(path in inventory.at("files", default: ()),
      message: "missing pinned data file: " + path)
  }
  path
}

#let inputs-ready(data-file, inputs) = inputs.all(
  key => data-file(key) != none,
)

// Defer the authored report's calculations until every required input exists.
// Keep the existing call signature; unavailable inputs show only a short notice.
// Do not substitute zeroes, historical values, or publication paths for empty inputs.
#let pending-report(data-file, inputs, question, figures, json-inputs: ()) = [
  A required run is unavailable, so there is no content to display yet.
]

// User-owned technical subview. No scientific report layout lives in Demolab.

#let bytes-label(value) = {
  let units = ((1000000000000, "TB"), (1000000000, "GB"), (1000000, "MB"), (1000, "KB"))
  for (scale, unit) in units {
    if value >= scale { return str(calc.round(value / scale, digits: 1)) + " " + unit }
  }
  str(value) + if value == 1 { " byte" } else { " bytes" }
}

#let date-label(value) = {
  // The validated projection normalizes every stage's creation time to UTC.
  let date = datetime(
    year: int(value.slice(0, 4)), month: int(value.slice(5, 7)), day: int(value.slice(8, 10)),
    hour: int(value.slice(11, 13)), minute: int(value.slice(14, 16)), second: 0,
  )
  date.display("[day padding:none] [month repr:short] [year], [hour repr:12 padding:none]:[minute] [period case:lower]")
}

#let duration-label(value) = {
  if value == none { return "—" }
  // Writers currently record whole-second timestamps; zero is below that resolution.
  if value < 1 { return "<1s" }
  let remaining = int(calc.round(value))
  let parts = ()
  for (scale, unit) in ((86400, "d"), (3600, "h"), (60, "m"), (1, "s")) {
    let count = calc.quo(remaining, scale)
    if count > 0 { parts.push(str(count) + unit) }
    remaining = calc.rem(remaining, scale)
  }
  parts.join(" ")
}

#let run-url(article, inputs, key, run) = {
  // Carry every current input into the new URL so comparisons are independent of later defaults.
  let selections = inputs.map(input => {
    let chosen = if input == key { run } else { selected-run(article, input) }
    if chosen == none { none } else {
      "source." + input + "=" + chosen.basepath.trim("/", at: start)
    }
  }).filter(value => value != none)
  "/" + article + "?" + selections.join("&")
}

#let run-view(article, inputs) = context {
  if target() == "html" {
    let interactive = sys.inputs.at("demolab-dev", default: "false") == "true"
    heading(level: 2)[Dataset]
    html.elem("style", ".run-view {margin:1rem 0 2rem;border:1px solid var(--rule-strong,#ddd);border-radius:.35rem;font-size:.85rem;overflow-x:auto;} .run-view .run-dependencies {display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.35rem 1rem;margin:0;padding:.7rem .8rem;border-bottom:1px solid var(--rule-strong,#ddd);} .run-view .run-dependencies dt {font-weight:600;color:var(--muted,#666);} .run-view .run-dependencies dd {margin:0;} .run-view .run-dependencies a {white-space:nowrap;} .run-view table {width:100%;margin:0;border:0;border-collapse:collapse;font-size:inherit;} .run-view th,.run-view td {padding:.6rem .8rem;text-align:left;vertical-align:baseline;border:0;border-bottom:1px solid var(--rule-strong,#ddd);} .run-view th {font-weight:600;color:var(--muted,#666);} .run-view tbody tr:last-child td {border-bottom:0;} .run-view .run-name {white-space:nowrap;} .run-view .run-stage-present {text-decoration:underline;text-underline-offset:.15em;} .run-view .run-date,.run-view .run-origin {color:var(--muted,#666);} .run-view .run-date {min-width:7.5em;white-space:nowrap;} .run-view .run-duration {white-space:nowrap;font-variant-numeric:tabular-nums;} .run-view .run-size {text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;} .run-view [aria-current=true] {font-weight:600;}")
    html.elem("aside", attrs: (class: "run-view", "aria-label": "Dataset"), {
      let dependencies = catalogue.at("experiment_dependencies", default: (:)).at(article, default: (:))
      html.elem("dl", attrs: (class: "run-dependencies"), {
        for (direction, label) in (("upstream", "Upstream"), ("downstream", "Downstream")) {
          html.elem("dt", label)
          html.elem("dd", attrs: (class: "experiment-" + direction), {
            let experiments = dependencies.at(direction, default: ())
            if experiments.len() == 0 { [—] }
            else {
              for (index, experiment) in experiments.enumerate() {
                if index > 0 { [, ] }
                html.elem("a", attrs: (href: "/" + experiment), experiment)
              }
            }
          })
        }
      })
      html.elem("table", attrs: ("aria-label": "Dataset"), {
        html.elem("thead", html.elem("tr", {
          for (label, class) in (("Run", "run-name"), ("Date", "run-date"), ("Duration", "run-duration"), ("Size", "run-size"), ("Ran on", "run-origin")) {
            html.elem("th", attrs: (scope: "col", class: class), label)
          }
        }))
        html.elem("tbody", {
        if inputs.len() == 0 {
          html.elem("tr", html.elem("td", attrs: (colspan: "5"), [No datasets declared.]))
        }
        for key in inputs {
          let selected = selected-run(article, key)
          let available = catalogue.at("display_runs", default: catalogue.runs).filter(run => {
            run.experiment == key.split(".").last() and (
              interactive or run.at("stage", default: "present") != "present"
              or (selected != none and selected.id == run.id)
            )
          })
          if available.len() == 0 {
            html.elem("tr", html.elem("td", attrs: (colspan: "5"), [#key — No presentation runs available.]))
          }
          for run in available {
            html.elem("tr", {
              let current = if selected != none and selected.id == run.id { "true" } else { "false" }
              let stage = run.at("stage", default: "present")
              let run-class = "run-name run-stage-" + stage
              html.elem("td", attrs: (class: "run-name"), {
              if interactive and stage == "present" {
                html.elem("a", attrs: (
                  class: run-class, href: run-url(article, inputs, key, run),
                  target: "_blank", rel: "noopener", "aria-current": current,
                ), run.id)
              } else {
                html.elem("span", attrs: (
                  class: run-class, "aria-current": current,
                ), run.id)
              }
              })
              html.elem("td", attrs: (class: "run-date"), {
                html.elem("time", attrs: (datetime: run.created_at, title: run.created_at), date-label(run.created_at))
              })
              let timing = run.at("display_timing", default: (
                duration_seconds: run.at("duration_seconds", default: none),
                basis: "recorded-operation",
                import_seconds: none,
              ))
              let duration = timing.duration_seconds
              html.elem("td", attrs: (
                class: "run-duration",
                title: if timing.basis == "scientific-execution" {
                  let total = if timing.at("job_seconds", default: none) != none {
                    " Sum of " + str(timing.jobs) + " recorded completed attempts: " + str(calc.round(timing.job_seconds / 3600, digits: 2)) + " job-hours; excludes unrecorded attempts."
                  } else { "" }
                  let operation = if timing.import_seconds != none {
                    " Import operation: " + str(timing.import_seconds) + " seconds (excluded)."
                  } else { "" }
                  "HPC wall-clock span: " + timing.started_at + " to " + timing.completed_at + "; includes gaps between jobs, not summed compute time." + total + operation
                } else if timing.basis == "historical-producer" {
                  let operation = if timing.import_seconds != none {
                    " Import operation: " + str(timing.import_seconds) + " seconds (excluded)."
                  } else { "" }
                  "Recorded HPC wall-clock: " + timing.started_at + " to " + timing.completed_at + "." + operation
                } else if timing.basis == "unrecorded-import-source" {
                  let operation = if timing.import_seconds != none {
                    " The local import took " + str(timing.import_seconds) + " seconds; that is excluded."
                  } else { "" }
                  "Original execution duration was not recorded." + operation
                } else if duration == none { "Execution timing not recorded" } else {
                  "Recorded elapsed time: " + str(duration) + " seconds (completed_at − started_at). This stage only; excludes upstream runs."
                },
              ), duration-label(duration))
              html.elem("td", attrs: (class: "run-size", title: "Export size: " + str(run.export_bytes) + " bytes"), bytes-label(run.export_bytes))
              html.elem("td", attrs: (class: "run-origin"), {
                let display = run.at("display_origin", default: (value: run.at("origin", default: "unknown"), basis: "recorded-operation"))
                let origin = display.value
                let label = if origin == "hpc" or origin.starts-with("slurm") { "HPC" }
                  else if origin == "local" { "Local" }
                  else if origin == "mixed" { "Mixed" }
                  else if origin == "modal" { "Modal" }
                  else if origin == "runpod" { "RunPod" }
                  else { "Unknown" }
                let title = if display.basis == "scientific-execution" {
                  "Recorded scientific execution origin: " + origin + ". The run record itself was created on " + run.origin + "."
                } else if display.basis == "historical-producer" {
                  "Recorded historical producer origin: " + origin + ". The import was performed on " + run.origin + "."
                } else if display.basis == "unrecorded-import-source" {
                  "Original execution origin was not recorded. The import was performed on " + run.origin + "."
                } else {
                  "Recorded stage execution origin: " + origin + "."
                }
                html.elem("span", attrs: (title: title), label)
              })
            })
          }
        }
        })
      })
    })
  }
}

// Flatten sequences only: retain styled content, figures and other containers intact.
#let dataset-parts(body) = {
  if body.func() == [].func() {
    body.children.map(dataset-parts).flatten()
  } else { (body,) }
}

#let dataset-heading-text(body) = {
  if body.has("text") { body.text }
  else if body.has("children") { body.children.map(dataset-heading-text).join() }
  else if body.has("body") { dataset-heading-text(body.body) }
  else if body == [ ] { " " }
  else { "" }
}

#let dataset-heading-level(item) = item.fields().at("level", default: item.fields().at("depth", default: 1))

#let with-datasets(article, inputs, body, placed: false) = {
  // Data-backed articles place their table explicitly inside the report's style scope.
  if placed { return body }
  let parts = dataset-parts(body)
  let headings = parts.enumerate().filter(((index, item)) => item.func() == heading)
  let end-matter = headings.filter(((index, item)) => {
    let title = lower(dataset-heading-text(item.body)).trim()
    dataset-heading-level(item) == 2 and (
      title == "references" or title.match(regex("^appendix(?:[.:]|\\s|$)")) != none
    )
  })
  let position = if end-matter.len() > 0 { end-matter.first().first() } else { parts.len() }
  parts.slice(0, position).join() + run-view(article, inputs) + parts.slice(position).join()
}
