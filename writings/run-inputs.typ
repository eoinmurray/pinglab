// Article-owned presentation inputs, with no implicit publication directory.
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
