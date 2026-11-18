// Use `deno fmt` to format this file.

import { fromMarkdown } from "npm:mdast-util-from-markdown@2.0.2";
import { visit } from "npm:unist-util-visit@5.0.0";
import { parseAll } from "https://deno.land/std@0.194.0/yaml/mod.ts";

/** Validate YAML code examples without applying the blog frontmatter schema. */
export function validateYamlCodeBlocks(
  markdown: string,
  filename: string,
): string[] {
  const errors: string[] = [];
  visit(fromMarkdown(markdown), "code", (node) => {
    if (!["yaml", "yml"].includes(node.lang?.toLowerCase() ?? "")) {
      return;
    }

    try {
      parseAll(node.value);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${filename}:${node.position?.start.line}: ${message}`);
    }
  });
  return errors;
}
