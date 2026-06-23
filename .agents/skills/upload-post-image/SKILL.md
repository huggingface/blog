---
name: upload-post-image
description: Use when adding or migrating non-thumbnail images for a Hugging Face Blog post. Uploads body images to the Hugging Face documentation-images dataset, updates the Markdown, and verifies the new links.
metadata:
  short-description: Upload blog post images
---

# Upload Post Image

Use this skill when a blog post needs an image in the article body.

Do not use this skill for thumbnails. Blog thumbnails stay in this repository under `assets/<post-slug>/` and are referenced from frontmatter with `/blog/assets/<post-slug>/<thumbnail-file>`.

Body images, diagrams, screenshots, GIFs, and videos should live in the Hugging Face Documentation Images dataset:

`huggingface/documentation-images`

Use a mirrored folder under:

`blog/<post-slug>/`

This keeps the blog repository small while still making images stable and public.

## Workflow

1. Identify the post slug.

Use the Markdown filename without `.md` unless the user asks for a different folder.

For `my-post.md`, use:

`blog/my-post/`

2. Identify the image role.

If the image is the post thumbnail, keep it in this repo under:

`assets/<post-slug>/`

If the image appears inside the article body, upload it to:

`huggingface/documentation-images`

3. Choose a descriptive filename.

Use a short filename that describes the image's role in the post.

Good examples:

- `architecture-diagram.svg`
- `dashboard-screenshot.png`
- `model-comparison-chart.jpg`
- `demo-output.gif`

Avoid temporary names, source-site names, hashes, timestamps, and names tied to a one-off migration.

4. Download or prepare the image locally.

For a remote image, download the direct image URL into `/tmp`:

```bash
mkdir -p /tmp/<post-slug>-images
curl -L --fail --silent --show-error "<direct-image-url>" -o /tmp/<post-slug>-images/<descriptive-file-name>
```

Use a direct image URL, not an album page or webpage.

5. Inspect the local file.

Confirm the file type, dimensions, and size:

```bash
file /tmp/<post-slug>-images/<descriptive-file-name>
du -h /tmp/<post-slug>-images/<descriptive-file-name>
```

Keep files reasonably small. Compress large PNG/JPEG files before upload when possible.

6. Confirm Hugging Face authentication.

```bash
hf auth whoami
```

If this fails, ask the user to authenticate before uploading.

7. Upload the image to the documentation-images dataset.

```bash
hf upload huggingface/documentation-images \
  /tmp/<post-slug>-images/<descriptive-file-name> \
  blog/<post-slug>/<descriptive-file-name> \
  --repo-type dataset
```

8. Use the resolver URL in the blog post.

The public URL has this shape:

```md
https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/<post-slug>/<descriptive-file-name>
```

Markdown image:

```md
![Clear alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/<post-slug>/<descriptive-file-name>)
```

HTML figure with caption:

```html
<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/<post-slug>/<descriptive-file-name>" alt="Clear alt text" width="70%">
  <figcaption>Short caption.</figcaption>
</figure>
```

9. Verify the new URL resolves.

```bash
curl -I --fail --silent --show-error "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/<post-slug>/<descriptive-file-name>"
```

A redirect to Hugging Face storage is expected.

10. Update the Markdown and check for stale links.

Search for the old host or path:

```bash
rg -n "<old-host-or-file-name>|documentation-images/resolve/main/blog/<post-slug>" <post-slug>.md
```

Make sure the article no longer depends on temporary hosts for body images.

11. Commit only the blog repo changes.

The dataset upload creates its own commit in `huggingface/documentation-images`.
In this repository, commit the Markdown change that points at the new image URL.

Use a clear Conventional Commit message, for example:

```bash
git commit -m "docs: move post images to documentation dataset"
```

## Notes

- Preserve meaningful alt text.
- Keep captions short and factual.
- Prefer lowercase filenames with hyphens.
- Avoid spaces in filenames.
- Do not upload private, sensitive, or uncleared copyrighted images.
- Do not move the post thumbnail to the dataset repo.
