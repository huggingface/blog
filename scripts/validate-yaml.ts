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
      title: z.string(),
      thumbnail: z.string().optional(),
      author: z.string(),
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
}
