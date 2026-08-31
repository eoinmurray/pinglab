// User-owned technical subview. No scientific report layout lives in Demolab.
#import "run-inputs.typ": catalogue, selected-run

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
          for (label, class) in (("Run", "run-name"), ("Date", "run-date"), ("Duration", "run-duration"), ("Size", "run-size"), ("Origin", "run-origin")) {
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
              let elapsed = run.at("duration_seconds", default: none)
              let imported = run.at("execution_operation", default: none) in ("import", "historical-import")
              let scientific = run.at("scientific_timing", default: none)
              let duration = if scientific != none { scientific.duration_seconds } else { elapsed }
              html.elem("td", attrs: (
                class: "run-duration",
                title: if scientific != none {
                  let total = if scientific.job_seconds != none {
                    " Sum of " + str(scientific.jobs) + " recorded completed attempts: " + str(calc.round(scientific.job_seconds / 3600, digits: 2)) + " job-hours; excludes unrecorded attempts."
                  } else { "" }
                  let operation = if imported and elapsed != none {
                    " Import operation: " + str(elapsed) + " seconds (excluded)."
                  } else { "" }
                  "Scientific execution span: " + scientific.started_at + " to " + scientific.completed_at + "; includes gaps between jobs, not summed compute time." + total + operation
                } else if elapsed == none { "Execution timing not recorded" } else {
                  let scope = if imported { "Import only; excludes original training or simulation." }
                    else { "This stage only; excludes upstream runs." }
                  "Recorded elapsed time: " + str(elapsed) + " seconds (completed_at − started_at). " + scope
                },
              ), duration-label(duration))
              html.elem("td", attrs: (class: "run-size", title: "Export size: " + str(run.export_bytes) + " bytes"), bytes-label(run.export_bytes))
              html.elem("td", attrs: (class: "run-origin"), {
                let origin = run.at("origin", default: "unknown")
                if origin in ("slurm", "modal", "runpod", "local", "mixed", "unknown") { origin } else { "unknown" }
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
