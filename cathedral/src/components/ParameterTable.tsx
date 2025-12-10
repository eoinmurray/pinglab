import { useFileContent } from "../../plugins/cathedral-plugin/src/client";
import { load } from "js-yaml"

export function ParameterTable({ path }: { path: string }) {
  const { content, loading, error } = useFileContent(path)

  let parsed = null;

  if (content && (path.endsWith(".yaml") || path.endsWith(".yml"))) {
    try {
      parsed = load(content);
    } catch (e) {
      console.error("Error parsing YAML:", e);
    }
  }

  if (content && path.endsWith(".json")) {
    try {
      parsed = JSON.parse(content);
    } catch (e) {
      console.error("Error parsing JSON:", e);
    }
  }

  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && <p>Error loading file: {error}</p>}
    </div>
  );
}