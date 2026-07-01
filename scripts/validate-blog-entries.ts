// Validates that markdown files are listed in _blog.yml.
// Usage: deno run --allow-read ./validate-blog-entries.ts file1.md file2.md ...
//
// File paths should be relative to the repository root.
// Use `deno fmt` to format this file.

import { parse } from "https://deno.land/std@0.194.0/yaml/mod.ts";

const SKIP = new Set(["README.md", "CONTRIBUTING.md"]);

const files = Deno.args;
if (files.length === 0) {
  console.log("No files to check.");
  Deno.exit(0);
}

const blogEntries = new Map<string, Set<string>>();

function getEntries(ymlPath: string): Set<string> {
  if (!blogEntries.has(ymlPath)) {
    const content = Deno.readTextFileSync(ymlPath);
    const entries = parse(content) as Array<{ local: string }>;
    blogEntries.set(ymlPath, new Set(entries.map((e) => e.local)));
  }
  return blogEntries.get(ymlPath)!;
}

let errors = 0;

for (const file of files) {
  const parts = file.split("/");
  const name = parts[parts.length - 1];

  if (!name.endsWith(".md")) {
    console.error(`ERROR: "${file}" is not a markdown file`);
    errors++;
    continue;
  }
  if (SKIP.has(name)) continue;

  // Only check root-level, zh/, and fr/ markdown files
  const isRoot = parts.length === 1;
  const subdir = parts.length === 2 && (parts[0] === "zh" || parts[0] === "fr")
    ? parts[0]
    : null;
  if (!isRoot && subdir === null) continue;
  const ymlPath = subdir ? `../${subdir}/_blog.yml` : "../_blog.yml";
  const slug = name.slice(0, -".md".length);
  const entries = getEntries(ymlPath);

  if (!entries.has(slug)) {
    console.error(
      `ERROR: "${file}" is not listed in ${ymlPath}`,
    );
    errors++;
  }
}

if (errors > 0) {
  console.error(`\n${errors} file(s) missing from _blog.yml`);
  Deno.exit(1);
} else {
  console.log("All checked files are listed in _blog.yml.");
}
