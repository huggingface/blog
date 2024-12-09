
---
title: "Open Preference Dataset for Text-to-Image Generation by the 🤗 Community"
thumbnail: /blog/assets/image_preferences/thumbnail.png
authors:
- user: davidberenstein1957
- user: burtenshaw
- user: dvilasuero
- user: davanstrien
- user: sayakpaul
- user: Ameeeee
- user: linoyts
---
# Open Preference Dataset for Text-to-Image Generation by the 🤗 Community

The Data is Better Together community releases yet another important dataset for open source development. Due to the lack of open preference datasets for text-to-image generation, we set out to release an Apache 2.0 licensed dataset for text-to-image generation. This dataset is focused on text-to-image preference pairs across common image generation categories, while mixing different model families and varying prompt complexities.

TL;DR? All results can be found in [this collection on the Hugging Face Hub](https://huggingface.co/collections/data-is-better-together/image-preferences-675135cc9c31de7f912ce278) and code for pre- and post-processing can be found in [this GitHub repository](https://github.com/huggingface/data-is-better-together). Most importantly, there is a [ready-to-go preference dataset](https://huggingface.co/datasets/data-is-better-together/image-preferences-results-binarized) and a [flux-dev-lora-finetune](https://huggingface.co/data-is-better-together/image-preferences-flux-dev-lora). If you want to show your support already, don’t forget to like, subscribe and follow us before you continue reading further.

<details>
  <summary>Unfamiliar with the Data is Better Together community?</summary>
  <p>
    [Data is Better Together](https://huggingface.co/data-is-better-together) is a collaboration between 🤗 Hugging Face and the Open-Source AI community. We aim to empower the open-source community to build impactful datasets collectively. You can follow the organization to stay up to date with the latest datasets, models, and community sprints.
  </p>
</details>

<details>
  <summary>Similar efforts</summary>
  <p>
    There have been several efforts to create an open image preference dataset but our effort is unique due to the varying complexity and categories of the prompts, alongside the openness of the dataset and the code to create it. The following are some of the efforts:

    - [yuvalkirstain/pickapic_v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2)
    - [fal.ai/imgsys](https://imgsys.org/)
    - [TIGER-Lab/GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)
    - [artificialanalysis image arena](https://artificialanalysis.ai/text-to-image/arena)
  </p>
</details>



## The input dataset

To get a proper input dataset for this sprint, we started with some base prompts, which we cleaned, filtered for toxicity and injected with categories and complexities using synthetic data generation with [distilabel](https://github.com/argilla-io/distilabel). Lastly, we used Flux and Stable Diffusion models to generate the images. This resulted in the following dataset:

<iframe
src="https://huggingface.co/datasets/data-is-better-together/image-preferences/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### Input prompts

[Imgsys](https://imgsys.org/) is a generative image model arena hosted by [fal.ai](http://fal.ai), where people provide prompts and get to choose between two model generations to provide a preference. Sadly, the generated images are not published publicly, however, [the associated prompts are hosted on Hugging Face](https://huggingface.co/datasets/fal/imgsys-results). These prompts represent real-life usage of image generation containing good examples focused on day-to-day generation, but this real-life usage also meant it contained duplicate and toxic prompts, hence we had to look at the data and do some filtering.

### Reducing toxicity

We aimed to remove all NSFW prompts and images from the dataset before starting the community. We settled on a multi-model approach where we used two text-based and two image-based classifiers as filters. Post-filtering, we decided to do a manual check of each one of the images to make sure no toxic content was left, luckily we found our approach had worked.

We used the following pipeline:
- Classify images as NSFW
- Remove all positive samples
- Argilla team manually reviews the dataset
- Repeat based on review

### Synthetic prompt enhancement

Data diversity is important for data quality, which is why we decided to enhance our dataset by synthetically rewriting prompts based on various categories and complexities. This was done using a [distilabel pipeline](https://github.com/huggingface/data-is-better-together/blob/main/community-efforts/image_preferences/01_synthetic_data_generation_total.py).

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Prompt</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Default</td>
      <td>a harp without any strings</td>
      <td><img src="/blog/assets/image_preferences/basic.jpeg" alt="Default Harp Image" width="100"></td>
    </tr>
    <tr>
      <td>Stylized</td>
      <td>a harp without strings, in an anime style, with intricate details and flowing lines, set against a dreamy, pastel background</td>
      <td><img src="/blog/assets/image_preferences/stylized.jpeg" alt="Stylized Harp Image" width="100"></td>
    </tr>
    <tr>
      <td>Quality</td>
      <td>a harp without strings, in an anime style, with intricate details and flowing lines, set against a dreamy, pastel background, bathed in soft golden hour light, with a serene mood and rich textures, high resolution, photorealistic</td>
      <td><img src="/blog/assets/image_preferences/quality.jpeg" alt="Quality Harp Image" width="100"></td>
    </tr>
  </tbody>
</table>

#### Prompt categories

[InstructGPT](https://arxiv.org/pdf/2203.02155) describes foundational task categories for text-to-text generation but there is no clear equivalent of this for text-to-image generation. To alleviate this, we used two main sources as input for our categories: [google/sdxl](https://huggingface.co/spaces/google/sdxl/blob/main/app.py) and [Microsoft](https://www.microsoft.com/en-us/bing/do-more-with-ai/ai-art-prompting-guide/ai-genres-and-styles?form=MA13KP). This led to the following main categories:  ["Cinematic", "Photographic", "Anime", "Manga", "Digital art", "Pixel art", "Fantasy art", "Neonpunk", "3D Model", “Painting”, “Animation” “Illustration”]. On top of that we also chose some mutually exclusive, sub-categories to allow us to further diversify the prompts. These categories and sub-categories have been randomly sampled and are therefore roughly equally distributed across the dataset.

#### Prompt complexities

[The Deita paper](https://arxiv.org/pdf/2312.15685) proved that evolving complexity and diversity of prompts leads to better model generations and fine-tunes, however, humans don’t always take time to write extensive prompts. Therefore we decided to use the same prompt in a complex and simplified manner as two datapoints for different preference generations.

### Image generation

The [ArtificialAnalysis/Text-to-Image-Leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard) shows an overview of the best performing image models. We choose two of the best performing models based on their license and their availability on the Hub. Additionally, we made sure that the model would belong to different model families in order to not highlight generations across different categories. Therefore, we chose [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). Each of these models was then used to generate an image for both the simplified and complex prompt within the same stylistic categories.

![image-generation](/blog/assets/image_preferences/example_generation.png)

## The results

A raw export of all of the annotated data contains responses to a multiple choice, where each annotator chose whether either one of the models was better, both models performed good or both models performed bad. Based on this we got to look at the annotator alignment, the model performance across categories and even do a model-finetune, which you can already [play with on the Hub](https://huggingface.co/black-forest-labs/FLUX.1-dev)! The following shows the annotated dataset:

<iframe
src="https://huggingface.co/datasets/data-is-better-together/image-preferences/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### Annotator alignment

Annotator agreement is a way to check the validity of a task. Whenever a task is too hard, annotators might not be aligned, and whenever a task is too easy they might be aligned too much. Striking a balance is rare but we managed to get it spot on during this sprint. We did [this analysis using the Hugging Face datasets SQL console](https://huggingface.co/datasets/data-is-better-together/image-preferences-results/embed/sql-console/0KQAlsp). Overall, SD3.5-XL was a bit more likely to win within our test setup.

### Model performance

Given the annotator alignment, both models proved to perform better within their own right, so [we did an additional analysis](https://huggingface.co/datasets/data-is-better-together/image-preferences-results/viewer/default/train?views%5B%5D=train&sql_console=true&sql=WITH+parsed_responses+AS+%28%0A++++--+Break+down+responses+into+individual+rows%0A++++SELECT+%0A++++++++id%2C+%0A++++++++UNNEST%28train.%22preference.responses%22%29+AS+response%0A++++FROM+train%0A%29%2C%0Aresponse_model_mapping+AS+%28%0A++++--+Map+responses+to+their+corresponding+models+and+include+category%0A++++SELECT+%0A++++++++pr.id%2C%0A++++++++train.category%2C%0A++++++++pr.response%2C%0A++++++++CASE+%0A++++++++++++WHEN+pr.response+%3D+%27image_1%27+THEN+train.model_1%0A++++++++++++WHEN+pr.response+%3D+%27image_2%27+THEN+train.model_2%0A++++++++END+AS+winning_model%0A++++FROM+parsed_responses+pr%0A++++JOIN+train+ON+pr.id+%3D+train.id%0A%29%2C%0Acategory_totals+AS+%28%0A++++--+Calculate+total+responses+and+agreement+per+category%0A++++SELECT+%0A++++++++category%2C%0A++++++++COUNT%28*%29+AS+total_responses%2C%0A++++++++AVG%281.0+%2F+COUNT%28DISTINCT+response%29%29+OVER+%28PARTITION+BY+category%29+AS+average_agreement%0A++++FROM+response_model_mapping%0A++++GROUP+BY+category%0A%29%2C%0Acategory_model_wins+AS+%28%0A++++--+Count+wins+for+each+model+grouped+by+category%0A++++SELECT+%0A++++++++category%2C%0A++++++++winning_model+AS+model%2C%0A++++++++COUNT%28*%29+AS+wins%0A++++FROM+response_model_mapping%0A++++GROUP+BY+category%2C+winning_model%0A%29%2C%0Acategory_win_ratios+AS+%28%0A++++--+Calculate+win+ratios+per+model+grouped+by+category%0A++++SELECT+%0A++++++++cmw.category%2C%0A++++++++cmw.model%2C%0A++++++++cmw.wins%2C%0A++++++++cmw.wins+*+1.0+%2F+SUM%28cmw.wins%29+OVER+%28PARTITION+BY+cmw.category%29+AS+win_ratio%0A++++FROM+category_model_wins+cmw%0A%29%2C%0Afinal_category_results+AS+%28%0A++++--+Combine+all+metrics+into+a+single+table+grouped+by+category%0A++++SELECT+%0A++++++++ct.category%2C%0A++++++++ct.total_responses%2C%0A++++++++ct.average_agreement%2C%0A++++++++MAX%28CASE+WHEN+cwr.model+%3D+%27dev%27+THEN+cwr.win_ratio+END%29+AS+dev_win_ratio%2C%0A++++++++MAX%28CASE+WHEN+cwr.model+%3D+%27sd%27+THEN+cwr.win_ratio+END%29+AS+sd_win_ratio%0A++++FROM+category_totals+ct%0A++++LEFT+JOIN+category_win_ratios+cwr+ON+ct.category+%3D+cwr.category%0A++++GROUP+BY+ct.category%2C+ct.total_responses%2C+ct.average_agreement%0A%29%0ASELECT+%0A++++category+AS+%22Category%22%2C%0A++++total_responses+AS+%22Total+Responses%22%2C%0A++++--+average_agreement+AS+%22Average+Agreement%22%2C%0A++++dev_win_ratio+AS+%22Win+Ratio+%28FLUX+dev%29%22%2C%0A++++sd_win_ratio+AS+%22Win+Ratio+%28Stable+Diffusion%29%22%0AFROM+final_category_results%0AORDER+BY+category%3B) to see if there were differences across the categories. In short, FLUX-dev works better for anime, and SD3.5-XL works better for art and cinematic scenarios.

- Tie: Photographic, Animation
- FLUX-dev better: 3D Model, Anime, Manga
- SD3.5-XL better: Cinematic, Digital art, Fantasy art, Illustration, Neonpunk, Painting, Pixel art

### Model-finetune

To verify the quality of the dataset, while not spending too much time and resources we decided to do a LoRA fine-tune of the [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model based on [the diffusers example on GitHub](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py). During this process, we included the chosen sample as expected completions for the FLUX-dev model and left out the rejected samples. Interestingly, the chosen fine-tuned models perform much better in art and cinematic scenarios where it was initially lacking! You can [test the fine-tuned adapter here](https://huggingface.co/data-is-better-together/image-preferences-flux-dev-lora).

![model-finetune](/blog/assets/image_preferences/model_finetune.png)

## The community

In short, we annotated 10K preference pairs with an annotator overlap of 2 / 3, which resulted in over 30K responses in less than 2 weeks with over 250 community members! The image leaderboard shows some community members even giving more than 5K preferences. We want to thank everyone that participated in this sprint with a special thanks to the top 3 users, who will all get a month of Hugging Face Pro membership. Make sure to follow them on the Hub: [aashish1904](https://huggingface.co/aashish1904), [prithivMLmods](https://huggingface.co/prithivMLmods), [Malalatiana](https://huggingface.co/Malalatiana).

![leaderboard](/blog/assets/image_preferences/leaderboard.png)

## What is next?

After another successful community sprint, we will continue organising them on the Hugging Face Hub. Make sure to follow [the Data Is Better Together organisation](https://huggingface.co/data-is-better-together) to stay updated. We also encourage community members to take action themselves and are happy to guide and reshare on socials and within the organisation on the Hub. You can contribute in several ways:

- Join and participate in other sprints.
- Propose your own sprints or requests for high quality datasets.
- Fine-tune models on top of [the preference dataset](https://huggingface.co/datasets/data-is-better-together/image-preferences-results-binarized). One idea would be to do a full SFT fine-tune of SDXL or FLUX-schnell. Another idea would be to do a DPO/ORPO fine-tune.
- Evaluate the improved performance of [the LoRA adapter](https://huggingface.co/data-is-better-together/image-preferences-flux-dev-lora) compared to the original SD3.5-XL and FLUX-dev models.
