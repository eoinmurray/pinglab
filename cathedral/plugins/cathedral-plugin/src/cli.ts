import { parseArgs } from "util";
import { buildAll } from "./lib";

async function cli(): Promise<void> {

  const { values } = parseArgs({
    args: Bun.argv,
    options: {
      dir: {
        type: "string",
      },
    },
    strict: true,
    allowPositionals: true,
  });

  if (!values.dir) {
    throw new Error("Content directory is required. Use --dir to specify.");
  }


  await buildAll([values.dir]);
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  cli()
}

export { cli };
