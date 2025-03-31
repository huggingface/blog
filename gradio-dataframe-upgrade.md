---
title: "Introducing Gradio's new Dataframe!" 
thumbnail: /blog/assets/gradio-dataframe-upgrade/thumbnail.png
authors:
- user: hmb
- user: abidlabs
---

# Introducing Gradio's new Dataframe!

Gradio’s `gr.Dataframe` component is one of our most popular components, we've seen it used in a variety of awesome apps, like leaderboards, dashboards, and interactive visualisations. Although we hadn't made any changes to the dataframe in quite some time, our backlog of issues had been growing, and some improvements had been in demand for a while. 

Well — we’re now super excited to release a host of new updates to Gradio’s dataframe component. Over the last 6 weeks, we’ve closed over 70 dataframe issues - including bugs, improvements and enhancements. 

### **1. Multi-Cell Selection**

You can select multiple cells at once! Copy or delete values across your selection with ease.

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/multicell.mp4">
</video>

### 2. Row Numbers & **Column Pinning**

Add row number columns and keep critical columns in view while navigating wide datasets using the `pinned_columns` parameter. No more losing track of what you're looking at!

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/rownumbers.mp4">
</video>

### **3. Copy Button and Full Screen Button**

Easily copy cell values into a comma-separated format with our new copy button. Need a better view? The full screen button gives you interactivity without distractions and can be enabled with the `show_full_screen` parameter. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/buttons.mp4">
</video>

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/buttons2.mp4">
</video>


### 4. Scroll to Top Button

Look at all that data! Now we can just scroll to the top. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/scrolltop.mp4">
</video>

### **5. Accessibility Upgrade and Enhanced Styling**

Improved keyboard navigation makes gr.Dataframe more accessible than ever. You can also take control of your dataframe’s look with a dedicated styler parameter and enhance the user experience of your app. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/a11y.mp4">
</video>

### **6. Row and Column Selection**

Access entire row data in select events for more intuitive interactivity and data manipulation.

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/rowcol.mp4">
</video>

### 7. Static Columns

Customise the interactivity of your dataframe by specifying non-editable columns using the `static_columns` parameter. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/static.mp4">
</video>

### **8. Search functionality**

Quickly find the data you need with our powerful search feature by setting the `show_search` parameter to `"search"`. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/searchfun.mp4">
</video>

### **9. Filter functionality**

Narrow down your dataset to focus on exactly what you need with flexible filtering options with `show_search` set to `"filter"`. 

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/filter.mp4">
</video>

### **10. Improved cell selection**

Experience smoother, more intuitive cell selection that behaves the way you expect.

<video width="600" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-dataframe-upgrade/cellselect.mp4">
</video>

## What’s next?

With over 70 issues closed, we’ve made huge improvements, but there’s always more for us to work on. Looking ahead, we still have more ideas to implement and we’re excited to keep refining accessibility, performance, and integration. Look out for some cool demos on our socials using the dataframe on Gradio’s [X](https://x.com/gradio).

## Try it yourself!

The updated dataframe is live in the latest Gradio release. Update your installation with `pip install --upgrade gradio`. 

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/5.22.0/gradio.js"
></script>

<gradio-app theme_mode="light" space="hmb/basic-dataframe"></gradio-app>

```python
import gradio as gr

df_headers = ["Name", "Population", "Size (min cm)", "Size (max cm)", "Weight (min kg)", "Weight (max kg)", "Lifespan (min years)", "Lifespan (max years)"]
df_data = [
    ["Irish Red Fox", 185000, 48, 92, 4.2, 6.8, 3, 5],
    ["Irish Badger", 95000, 62, 88, 8.5, 13.5, 6, 8],
    ["Irish Otter", 13500, 58, 98, 5.5, 11.5, 9, 13]
]

with gr.Blocks() as demo:
    df = gr.Dataframe(
        label="Irish Wildlife",
        value=df_data,
        headers=df_headers,
        interactive=True,
        show_search="search",
        show_copy_button=True,
        show_fullscreen_button=True,
        show_row_numbers=True,
        pinned_columns=1,
        static_columns=[0],
        column_widths=["300px"]
    )

demo.launch()
```

Check out the [Gradio documentation](https://www.gradio.app/docs/gradio/dataframe) for examples and tutorials to get started with these new features. We’re eager to see what you create! Got thoughts or suggestions? Share them by raising an issue in our [GitHub](https://github.com/gradio-app/gradio) repo. 

Happy building!
