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
    heading(level: 2)[Datasets]
    html.elem("style", ".run-view {margin:1rem 0 2rem;border:1px solid var(--rule-strong,#ddd);border-radius:.35rem;font-size:.85rem;overflow-x:auto;} .run-view .run-dependencies {display:grid;grid-template-columns:max-content minmax(0,1fr);gap:.35rem 1rem;margin:0;padding:.7rem .8rem;border-bottom:1px solid var(--rule-strong,#ddd);} .run-view .run-dependencies dt {font-weight:600;color:var(--muted,#666);} .run-view .run-dependencies dd {margin:0;} .run-view .run-dependencies a {white-space:nowrap;} .run-view table {width:100%;margin:0;border:0;border-collapse:collapse;font-size:inherit;} .run-view th,.run-view td {padding:.6rem .8rem;text-align:left;vertical-align:baseline;border:0;border-bottom:1px solid var(--rule-strong,#ddd);} .run-view th {font-weight:600;color:var(--muted,#666);} .run-view tbody tr:last-child td {border-bottom:0;} .run-view .run-name {white-space:nowrap;} .run-view .run-date,.run-view .run-origin {color:var(--muted,#666);} .run-view .run-date {min-width:7.5em;} .run-view .run-size {text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;} .run-view [aria-current=true] {font-weight:600;}")
    html.elem("aside", attrs: (class: "run-view", "aria-label": "Datasets"), {
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
      html.elem("table", attrs: ("aria-label": "Datasets"), {
        html.elem("thead", html.elem("tr", {
          for (label, class) in (("Run", "run-name"), ("Date", "run-date"), ("Size", "run-size"), ("Origin", "run-origin")) {
            html.elem("th", attrs: (scope: "col", class: class), label)
          }
        }))
        html.elem("tbody", {
        if inputs.len() == 0 {
          html.elem("tr", html.elem("td", attrs: (colspan: "4"), [No datasets declared.]))
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
            html.elem("tr", html.elem("td", attrs: (colspan: "4"), [#key — No presentation runs available.]))
          }
          for run in available {
            html.elem("tr", {
              let current = if selected != none and selected.id == run.id { "true" } else { "false" }
              html.elem("td", attrs: (class: "run-name"), {
              if interactive and run.at("stage", default: "present") == "present" {
                html.elem("a", attrs: (
                  class: "run-name", href: run-url(article, inputs, key, run),
                  target: "_blank", rel: "noopener", "aria-current": current,
                ), run.id)
              } else {
                html.elem("span", attrs: (
                  class: "run-name",
                  style: if run.at("stage", default: "present") == "compute" {
                    "text-decoration:underline;text-underline-offset:.15em;"
                  } else { "text-decoration:none;" },
                ), run.id)
              }
              })
              html.elem("td", attrs: (class: "run-date"), {
                html.elem("time", attrs: (datetime: run.created_at, title: run.created_at), date-label(run.created_at))
              })
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
  let abstracts = headings.filter(((index, item)) => {
    lower(dataset-heading-text(item.body)).trim().match(regex("^([0-9]+[.]?\\s+)?abstract$")) != none
  })
  let position = parts.len()
  if abstracts.len() > 0 {
    let (index, abstract) = abstracts.first()
    let following = headings.filter(((next, item)) => next > index and dataset-heading-level(item) <= dataset-heading-level(abstract))
    if following.len() > 0 { position = following.first().first() }
  } else if headings.len() > 0 {
    // Reference pages keep their introduction, if any, before the datasets section.
    position = headings.first().first()
  }
  parts.slice(0, position).join() + run-view(article, inputs) + parts.slice(position).join()
}
