// Use `deno fmt` to format this file.

import { parse } from "https://deno.land/std@0.194.0/yaml/mod.ts";
import { z } from "https://deno.land/x/zod@v3.16.1/mod.ts";

for (const lang of ["", "zh/"]) {
  {
    const relativePath = `${lang}_events.yml`;
    console.log(`Validating ${relativePath}`);

    z.array(z.object({
      link: z.string().url(),
      name: z.string(),
      date: z.string(),
      date_formatted: z.string().optional(),
      description: z.string(),
    })).parse(parse(Deno.readTextFileSync("../" + relativePath)));
  }

  {
    const relativePath = `${lang}_blog.yml`;
    console.log(`Validating ${relativePath}`);

    const localMdFiles = [...Deno.readDirSync("../" + lang)].map((entry) =>
      entry.name
    ).filter((name) => name.endsWith(".md")).map((name) =>
      name.slice(0, -".md".length)
    );

    z.array(z.object({
      local: z.enum([...localMdFiles] as [string, ...string[]]),
      guest: z.boolean().optional(),
      date: z.string(),
      tags: z.array(z.string()),
    })).parse(parse(Deno.readTextFileSync("../" + relativePath)));
  }

  {
    const relativePath = `${lang}_tags.yml`;
    console.log(`Validating ${relativePath}`);

    z.array(z.object({
      value: z.string(),
      label: z.string(),
    })).parse(parse(Deno.readTextFileSync("../" + relativePath)));
  }

  {
    console.log(`Validating ${lang}blog post frontmatter`);

    const localMdFiles = [...Deno.readDirSync("../" + lang)].map((entry) =>
      entry.name
    ).filter((name) => name.endsWith(".md"));

    const frontmatterSchema = z.object({
      title: z.string(),
      thumbnail: z.string().regex(/\.(jpg|jpeg|gif|png|webp)$/i, "Thumbnail must end with .jpg, .jpeg, .gif, .png, or .webp"),
      authors: z.array(z.object({
        user: z.string(),
      })),
    });

    for (const mdFile of localMdFiles) {
      // Skip special files
      if (mdFile === "README.md" || mdFile === "CONTRIBUTING.md") {
        continue;
      }
      
      const content = Deno.readTextFileSync("../" + lang + mdFile);
      const frontmatterMatch = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
      
      if (!frontmatterMatch) {
        throw new Error(`No frontmatter found in ${lang}${mdFile}`);
      }

      try {
        frontmatterSchema.parse(parse(frontmatterMatch[1]));
      } catch (error) {
        console.error(`Error in ${lang}${mdFile}:`, error);
        throw error;
      }
    }
  }
}
