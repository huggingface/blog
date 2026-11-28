import {
  assertEquals,
  assertStringIncludes,
} from "https://deno.land/std@0.194.0/testing/asserts.ts";
import { validateYamlCodeBlocks } from "./validate-markdown-yaml.ts";

Deno.test("accepts YAML and YML examples, including multiple documents", () => {
  const markdown = [
    "```yaml",
    "name: example",
    "---",
    "items: [one, two]",
    "```",
    "~~~YML title=example",
    "enabled: true",
    "~~~",
  ].join("\n");
  assertEquals(validateYamlCodeBlocks(markdown, "post.md"), []);
});

Deno.test("reports the Markdown filename and line of each invalid example", () => {
  const markdown =
    "# Title\n\n```yaml\nitems: [one\n```\n\n```yml\nitems: [two\n```";
  const errors = validateYamlCodeBlocks(markdown, "zh/post.md");
  assertEquals(errors.length, 2);
  assertStringIncludes(errors[0], "zh/post.md:3:");
  assertStringIncludes(errors[1], "zh/post.md:7:");
});

Deno.test("checks code blocks nested inside lists and blockquotes", () => {
  const markdown = [
    "- Example:",
    "",
    "  ```yaml",
    "  items: [one",
    "  ```",
    "",
    "> ~~~yml",
    "> items: [two",
    "> ~~~",
  ].join("\n");
  assertEquals(validateYamlCodeBlocks(markdown, "post.md").length, 2);
});

Deno.test("ignores other languages, unlabelled blocks and literal fence examples", () => {
  const markdown = [
    "```python",
    "items: [one",
    "```",
    "```",
    "items: [two",
    "```",
    "````markdown",
    "```yaml",
    "items: [three",
    "```",
    "````",
  ].join("\n");
  assertEquals(validateYamlCodeBlocks(markdown, "post.md"), []);
});

Deno.test("does not apply frontmatter field requirements to examples", () => {
  assertEquals(
    validateYamlCodeBlocks("```yaml\ncustom_key: custom_value\n```", "post.md"),
    [],
  );
});

Deno.test("checks every document in a multi-document YAML block", () => {
  assertEquals(
    validateYamlCodeBlocks(
      "```yaml\nvalid: true\n---\nitems: [one\n```",
      "post.md",
    ).length,
    1,
  );
});
