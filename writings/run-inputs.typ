// Article-owned presentation inputs, with no implicit publication directory.
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
