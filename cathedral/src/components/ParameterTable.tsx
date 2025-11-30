import { useDirectory, useFileContent } from "../../plugins/cathedral-plugin/src/client";
import yaml from "js-yaml";

export function ParameterTable({ path }: { path: string }) {
  const { file } = useDirectory(path)
  const { content, loading, error } = useFileContent(path);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (!file || !content) {
    return <div>No file found at the specified path.</div>;
  }

  let parsed: any = {}
  if (file.name.endsWith(".yaml") || file.name.endsWith(".yml")) {
    try {
      parsed = yaml.load(content);
      // Process the YAML data as needed
    } catch (e) {
      return <div>Error parsing YAML: {(e as Error).message}</div>;
    }
  } else if (file.name.endsWith(".json")) {
    try {
      parsed = JSON.parse(content);
      // Process the JSON data as needed
    } catch (e) {
      return <div>Error parsing JSON: {(e as Error).message}</div>;
    }
  } else {
    return <div>Unsupported file format. Please provide a YAML or JSON file.</div>;
  }

  return (
    <div>
      <div>
        {file.name}
      </div>

      <div>
        <pre>{JSON.stringify(parsed, null, 2)}</pre>
      </div>
    </div>
  )
}