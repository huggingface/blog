---
title: "FineVideo: behind the scenes" 
thumbnail: /blog/assets/186_fine_video/thumbnail.png
authors:
- user: mfarre
- user: andito
- user: lewtun
- user: lvwerra
- user: pcuenq
- user: thomwolf
---
<center>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/logo.png" alt="FineVideo logo" style="width: 50%;"><br>
</center>

Open video datasets are scarce and therefore slowing down the development of open-source video AI. For this reason we built [FineVideo](https://huggingface.co/spaces/HuggingFaceFV/FineVideo-Explorer), a dataset with 43k videos that span 3.4k hours and are annotated with rich descriptions, narrative details, scene splits, and QA pairs. 

FineVideo contains a highly diverse collection of videos and metadata which makes it a good ingredient to train models to understand video content, train diffusion models to generate videos from a text description or train computer vision models using its structured data as input.

Wait, you haven’t seen FineVideo yet? take a look at it through the [dataset explorer page](https://huggingface.co/spaces/HuggingFaceFV/FineVideo-Explorer).

<center>
    <br>
    <a href="https://huggingface.co/spaces/HuggingFaceFV/FineVideo-Explorer">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/finevideo.gif" alt="FineVideo Explorer" style="width: 60%;">
    </a>
    <br><br>
</center>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [About this blog post](#about-this-blog-post)
- [Building the Raw dataset](#building-the-raw-dataset)
  - [Filtering YouTube-Commons](#filtering-youtube-commons)
  - [Downloading the videos](#downloading-the-videos)
- [Keeping dynamic content](#keeping-dynamic-content)
  - [Word density filtering](#word-density-filtering)
  - [Visual dynamism filtering](#visual-dynamism-filtering)
- [Video Categorization](#video-categorization)
  - [Custom built Taxonomy](#custom-built-taxonomy)
  - [Content annotation](#content-annotation)
  - [Feedback loop taxonomy - content annotation](#feedback-loop-taxonomy---content-annotation)
- [Contributing descriptive metadata](#contributing-descriptive-metadata)
  - [Long videos \& Gemini 1.5 Pro](#long-videos--gemini-15-pro)
  - [Content selection](#content-selection)
  - [Annotating with Gemini 1.5 Pro and Structured Output with GPT4o](#annotating-with-gemini-15-pro-and-structured-output-with-gpt4o)
- [Fine Alignment and anomaly filtering](#fine-alignment-and-anomaly-filtering)
- [Future Work](#future-work)

## About this blog post
In this blog post, we share the technical details and code involved in developing FineVideo: a journey that starts with 1.9M videos in [YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons) and ends with 44K videos with all details annotated.

A good way to start is taking a look at the different steps of our journey. Those steps involve content filtering, annotation and output structuring.

<center>
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/dataset-creation.png" alt="Dataset Creation" style="width: 70%;">
    <figcaption style="font-style: italic;">FineVideo video filtering and annotation pipeline</figcaption>
    <br><br>
</center>

In the following sections we discuss each of the steps and provide references to relevant parts of the code. If you prefer to navigate the code directly, take a look at our FineVideo repository on [Github](https://github.com/mfarre/fineVideo).

First, let’s have a look how we got an initial list of YouTube videos and how we apply some first filters.
<br>
## Building the Raw dataset

Our journey starts in [YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons): a collection of audio transcripts of videos shared on YouTube under a CC-By license.  Such project was created and is currently maintained by [PleIAs](https://pleias.fr/) as part of their corpus collection projects.
<br>
### Filtering YouTube-Commons

YouTube Commons contain videos and transcripts in a diverse set of languages, our initial task is about narrowing down the content of YouTube Commons to the same language.

We filter YouTube-Commons for videos in English and at the same time we gather relevant metadata. From this initial filtering, we collect 1.9M videos, their closed captions and metadata.

Below some details on the filters and metadata fields that we keep:

**Filters**

<div style="text-align: center;margin: auto; width: 80%;">

| **Field** | **Filter value** | **Description** |
| --- | --- | --- |
| original_language | en | videos in English|
| transcription_language | en | transcripts in English |
</div>
<br>

**Metadata fields**

<div style="text-align: center;margin: auto; width: 80%;">


  <details>
    <summary>Click to Expand Metadata Fields</summary>
    <table style="width: 100%; margin-top: 10px;">
      <tr>
        <th>Field</th>
        <th>Description</th>
      </tr>
      <tr>
        <td>acodec</td>
        <td>audio codec</td>
      </tr>
      <tr>
        <td>age_limit</td>
        <td>YouTube age restrictions for the video</td>
      </tr>
      <tr>
        <td>categories</td>
        <td>YouTube video category</td>
      </tr>
      <tr>
        <td>channel</td>
        <td>YouTube channel</td>
      </tr>
      <tr>
        <td>channel_follower_count</td>
        <td>Number of subscribed users to the channel</td>
      </tr>
      <tr>
        <td>channel_id</td>
        <td>YouTube channel identifier</td>
      </tr>
      <tr>
        <td>character_count</td>
        <td>Number of characters in the closed caption</td>
      </tr>
      <tr>
        <td>comment_count</td>
        <td>Number of comments in YouTube</td>
      </tr>
      <tr>
        <td>description</td>
        <td>YouTube video description</td>
      </tr>
      <tr>
        <td>duration_string</td>
        <td>Video duration in hh:mm:ss format</td>
      </tr>
      <tr>
        <td>license</td>
        <td>Video License</td>
      </tr>
      <tr>
        <td>like_count</td>
        <td>Number of video likes in YouTube</td>
      </tr>
      <tr>
        <td>resolution</td>
        <td>Pixel resolution of the video in the format Width x Height</td>
      </tr>
      <tr>
        <td>tags</td>
        <td>YouTube free text tags associated with the video</td>
      </tr>
      <tr>
        <td>text</td>
        <td>Closed Caption</td>
      </tr>
      <tr>
        <td>title</td>
        <td>YouTube video title</td>
      </tr>
      <tr>
        <td>upload_date</td>
        <td>YouTube upload date</td>
      </tr>
      <tr>
        <td>vcodec</td>
        <td>Video Codec</td>
      </tr>
      <tr>
        <td>video_id</td>
        <td>YouTube video identifier</td>
      </tr>
      <tr>
        <td>view_count</td>
        <td>Number of views in YouTube</td>
      </tr>
      <tr>
        <td>word_count</td>
        <td>Number of words in the closed caption</td>
      </tr>
    </table>
  </details>
</div>
<br>

Code for content filtering and metadata gathering available here [[link](https://github.com/mfarre/fineVideo/blob/main/rawdataset/filter-yt-commons.py)]

### Downloading the videos

Once we had a target video list with 1.9M videos, we managed to successfully download 1.8M videos (some of the videos where removed by the channel owners and some changed their permissions).

We explored two different approaches for distributed downloading.

<u><b>Option 1: Video2dataset</b></u>

video2dataset is an open-source project [[link](https://github.com/iejMac/video2dataset)] that focuses on distributed video download, transformation and packaging in different dataset formats. The project natively supports Slurm Workload Manager and therefore we could run it in our CPU cluster.

<center>
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/video2dataset_overview.png" alt="Dataset Creation" style="width: 60%;">
    <figcaption style="font-style: italic;">Source: Video2Dataset GitHub page</figcaption>
    <br><br>
</center>

As all our cluster instances face internet with the same public IP, we contributed to the project the possibility to specify a proxy to facilitate video downloads. While the feature is not yet merged, you can patch video2dataset with our PR [[link](https://github.com/iejMac/video2dataset/pull/350)] to use the proxy capabilities.

<br>
<u><b>Option 2: Cloud batch jobs</b></u>
<br><br>
Most cloud providers have the possibility to run jobs by simply defining the type of instance that will execute each job, defining a queue and providing a container with the code that will be executed.

We used Google Cloud and AWS to run a custom-made docker container that downloads videos and metadata with [ytdlp](https://github.com/yt-dlp/yt-dlp) and pushes the results to S3. 

The files to build the Docker container can be found here [[code](https://github.com/mfarre/fineVideo/tree/main/rawdataset/ytdlps3)].

<u><b>Our conclusion</b></u>

While Video2Dataset was functional with a proxy and allowed us to do additional processing steps, the requests / second we could do to the proxy became a bottleneck. This made us pivot towards cloud batch jobs.

## Keeping dynamic content

In our search for the best videos, we narrowed down our selection to content where there is both visual action and people speaking at a mid-fast pace. We achieve this with word density filtering and visual dynamism filtering.

### Word density filtering

We took the density of words in the video as a proxy of audio dynamism. The definition of word density being:

`Word density = Number of words in closed captions / Total video length in seconds`


  

By sampling and visually evaluating the quality of the content at different density thresholds, we decided to remove all videos with a word density lower than 0.5 words/second.

Examples:

<div style="text-align: center;margin: auto; width: 50%;">

| **Word density** | **Example** |
| --- | --- |
| 0.25 | <iframe width="200" height="113" src="https://www.youtube.com/embed/mqAeYCSP1wA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |
| 0.5  | <iframe width="200" height="113" src="https://www.youtube.com/embed/eLtOfmzdU_o" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |
| 0.75 | <iframe width="200" height="113" src="https://www.youtube.com/embed/nx9yfGgXK6s" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |
| 1.0  | <iframe width="200" height="113" src="https://www.youtube.com/embed/7xMDfivSrkg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |

</div>


The code to filter by word density and explore examples can be found here [[link](https://github.com/mfarre/fineVideo/blob/main/dynamicfilters/worddensityfiltering.py)]

### Visual dynamism filtering

We repurposed FFMPEG’s [Freezedetect filter](https://ffmpeg.org/ffmpeg-filters.html#freezedetect) to  judge the dynamism of the video. While this filter is designed to identify frozen sections of a video (multiple equal frames placed one after the other), we could also identify chunks with low movement by exaggerating the `noise` parameter to a very high value.

Rather than running freezedetect across the full video, we analyzed the video by temporal segments and we voted if the video was static based on the amount of segments categorized as static. Through manual evaluation we set a threshold to discard the video if 40% of the segments analyzed have low movement.

Some types of content discarded after this filtering:
<div style="text-align: center;margin: auto; width: 50%;">

| **Type** | **Example** |
| --- | --- |
| Static image with music | <iframe width="200" height="113" src="https://www.youtube.com/embed/-3PjwEGxu9w" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |
| Presentation screen cast | <iframe width="200" height="113" src="https://www.youtube.com/embed/-72DqMfjtF8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |
| Highly static people talking to camera | <iframe width="200" height="113" src="https://www.youtube.com/embed/0-KRYKbg_T8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> |

</div>

The DockerFile and code to classify video by its dynamism can be found here [[link](https://github.com/mfarre/fineVideo/tree/main/dynamicfilters/videodynamismfiltering)]

From the 1.8M videos analyzed, after this step we keep 600K dynamic videos. At this stage, we dig deeper into the content of the videos, which will be key to ensure diversity in the dataset.


## Video Categorization

In order to achieve the most diverse content selection, we categorized the 600K filtered assets using the closed captioning and YouTube metadata. As a way to gain control on the categorization, we created a taxonomy and guided the annotation process to adhere to the taxonomy.

### Custom built Taxonomy

We bootstrapped the custom built taxonomy using GPT4-o and an information scientist reviewed and adjusted it. The taxonomy contains 126 fine categories aggregated in multiple levels. This multi-level approach allow users of FineVideo to slice the dataset to fit their particular use-case.

![taxonomy](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/taxonomy.png)

The taxonomy is also available in JSON [[link](https://github.com/mfarre/fineVideo/blob/main/videocategorization/content_taxonomy.json)]

With an initial version of the taxonomy we started content annotation and by looking at the results of content annotation, with the help of an information scientist, we adjusted the taxonomy accordingly. 

### Content annotation

We categorized the videos using Llama 3.1 70B served through Text Generation Inference [TGI](https://github.com/huggingface/text-generation-inference) [[code](https://github.com/mfarre/fineVideo/tree/main/videocategorization)]. 

The prompt required multiple iterations to ensure the answer is strictly a category in our taxonomy. During our prompt evaluation we learned that by removing the existing YouTube tags and categories from the prompt, the quality of our results increased drastically: YouTube metadata was biasing the text generated by Llama 3.1 towards one of the categories provided by YouTube.

```python
prompt_template = """
Given those categories: {leaves}
Classify a youtube video given its closed captioning and some metadata details. RETURN ONLY the selected category and nothing else!
Title: {title}
Description: {description}
Channel: {channel}
Closed Caption: {closed_caption}
"""
```

### Feedback loop taxonomy - content annotation

<center>
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/categorization-feedback-loop.png" alt="Categorization feedback loop" style="width: 40%;">
    <figcaption style="font-style: italic;">Taxomy adjustments during content categorization</figcaption>
    <br><br>
</center>
One of the roles of information scientists is to curate taxonomies over-time to add new categories or add some extra degrees of differentiation when needed. 

Using LLMs to categorize content stresses the need to adjust taxonomies from months / years to hours. Furthermore, in some cases, we created categories specifically to discard sensitive videos such as the ones falling under `Firearms & Weapons` and `Substance Use & Drugs`.

## Contributing descriptive metadata

At this point of the process, we have three sources of video level metadata:
* video category (inferred with Llama 3.1)
* YouTube Metadata (title, description)
* Transcripts from YouTube-Commons


In order to contribute in the field of video understanding, we decided to go deeper into timecode-level metadata, for example activities, objects, narrative and editing aspects. 
While human annotation was something we considered as part of active learning setup where one or more models propose annotations and the human does a QA step, as we will discuss in the next sections, we found in Gemini a good solution especially when we constrained the input video length and the output format. 

### Long videos & Gemini 1.5 Pro

We dig deeper into Gemini 1.5 Pro iterating our prompt and testing it with different content length. 

Given its limitation to 1M tokens, which is approximately equivalent to ~1hour of video, we were forced to drop videos longer than 1 hour. 
An idea to overcome this situation was to accelerate videos longer than one hour and that way fit in Gemini’s context. 

<center>
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/gemini-context-cartoon.png" alt="Gemini context" style="width: 80%;">
    <figcaption style="font-style: italic;">Exploration: accelerating videos to fit more content in Gemini's context</figcaption>
    <br><br>
</center>

While it seemed to work at high level, when we started looking at the details we realized that only the first minutes of the video were accurately annotated. 

Finding that quality drops on long videos made us wonder: is this an issue impacting the rest of our videos? by sampling videos of different lengths and inspecting the video coverage of the annotations, we found a reduction in quality for videos longer than 10+ minutes. 

Aligned with our goal to bring high quality data back to the community, we dropped videos longer than 10+ minutes.

### Content selection

Given that each hour of video costs more than $5 to annotate with Gemini, we can’t annotate all the videos that we have after filtering. Therefore, we wanted to make sure that we have a good coverage over all topics and we search a good compromise of content diversity for late-pre-training / fine-tuning task and budget. We set this size constraint to 4K hours of video.

In order to go from 600K videos to 4K hours of content we prepared an algorithm that balances content categories, user engagement, and channel representation to achieve the targeted duration.

<div style="display: flex; align-items: flex-start;">

  <!-- Image on the left -->
  <div style="flex: 1; text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/oracle-flow.png" alt="Oracle Flow" style="max-width: 100%; height: auto;clip-path: inset(0px 0px 3px 0px);">
    <p><em>Algorithm flow diagram</em></p>
  </div>

  <!-- Text on the right -->
  <div style="flex: 1; padding-left: 20px;">
    <br><br><br>
    <h3>Some key parts of the content selection algorithm:</h3>
    <ul>
      <li><strong>Activity Score</strong>: We calculate an engagement metric for each video by combining comment, view, and like counts with weighted importance. This score helps prioritize videos that have resonated well with viewers.</li><br><br>
      <li><strong>Video Selection</strong>: This step iteratively selects videos to meet the target duration while ensuring diversity. It balances between high-engagement content and representation from various categories and channels, using a penalty system to avoid overrepresentation of any single channel.</li><br><br>
      <li><strong>Final Adjustment</strong>: We adjust the selection to match the target duration as closely as possible without exceeding it. It sorts the selected videos by duration and adds them to the final list until reaching the closest possible total duration to the target.</li>
    </ul>
  </div>
</div>

<!-- Additional text beneath the image and text -->
<div style="margin-top: 20px;">
  <p>The code can be found in the repository <a href="https://github.com/mfarre/fineVideo/blob/main/contentselection/oracle.py" target="_blank">[link]</a>.</p>
</div>


<br>

### Annotating with Gemini 1.5 Pro and Structured Output with GPT4o

<u><b>Why structured data?</b></u>

One of our goals with FineVideo is to provide structured data as a way to empower our community: if you are working on MultiModal LLMs, you can slice the data and decide which categories fit your pre-training or fine-tuning mix. If you are more into computer vision, you can directly use the dataset to train classifiers based on the numerical categories included in FineVideo such as the dynamism score, scene boundaries or audio/video correlation score.


<u><b>Structured data and Gemini 1.5</u></b>

Gemini 1.5 Pro allows JSON based outputs by providing a schema. We explored this feature and we quickly realized two issues:

- We could not fit our original schema into Gemini because our schema is highly complex
- When we tried with slightly simpler schemas -still quite complex- the quality of the Gemini results dropped substantially: most of the scene types of data (characters, activities, props) dropped. We tried splitting the prompt in multiple prompts and matching the different prompts to different parts of the schema without much success.

What we observed completely matched what other researchers experienced: adding concrete schema constraints can decrease performance. ([Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://huggingface.co/papers/2408.02442)).

Our solution relied on generating free text with Gemini 1.5 and add a second processing step to align the results of Gemini with our schema.

The Gemini prompt that we used is the following:

```
Study the video and provide the following details about the video and the semantic scenes that compose it:

- characterList: a list of characters that appear in the whole video and a visual description that should allow me to identify them just seeing an image of them.
- scenes: a list of the scenes with the following properties:
  - start/end timestamps of the scene
  - list of all the characters that appear in the scene
  - list of all activities and their timestamps
  - list of all props and their timestamps
  - list of all video editing details and their start/end timestamps. Details include transitions, effects, music as well as suggestions like segments of the scene that could be removed and why 
  - scene mood with notes on how the visuals, audio and context contribute to it. Use the following taxonomy returning only the name in your answer {"moods":{"Positive":[{"name":"Happy","description":"Feeling joyful, content, or delighted."},{"name":"Excited","description":"Feeling enthusiastic, energetic, or eager."},{"name":"Calm","description":"Feeling peaceful, relaxed, or serene."},{"name":"Grateful","description":"Feeling appreciative or thankful."},{"name":"Proud","description":"Feeling satisfied with one's achievements or the achievements of others."}],"Negative":[{"name":"Sad","description":"Feeling down, unhappy, or sorrowful."},{"name":"Angry","description":"Feeling irritated, frustrated, or furious."},{"name":"Anxious","description":"Feeling nervous, worried, or uneasy."},{"name":"Lonely","description":"Feeling isolated, disconnected, or abandoned."},{"name":"Bored","description":"Feeling uninterested, disengaged, or restless."}],"Neutral":[{"name":"Indifferent","description":"Feeling neither particularly positive nor negative."},{"name":"Content","description":"Feeling satisfied but not overly excited."},{"name":"Curious","description":"Feeling interested or inquisitive without strong emotion."},{"name":"Confused","description":"Feeling uncertain or unclear but without strong negative feelings."},{"name":"Pensive","description":"Feeling thoughtful or reflective without strong emotional engagement."}]}}
    - specific  mood changing moments inside the scene, report the timestamp and what we transition from/to in any of the dimensions (visual / auditive)
  - scene narrative progression and plot development
    - specific narrative moments inside the scene. Report the timestamp and what happened
  - character interaction and dynamics descriptions and their start/end timestamps
  - specific thematic elements and descriptions
  - specific relevant happenings to create deeper meanings and subtexts not explicitly stated that contribute to the richness and depth of the content, timestamp and descriptions
  - dynamism score of the scene. Score between 0 and 1. 1 is highly dynamic
  - audio - visual correlation score. Score between 0 and 1. 0 what we see is not correlated with the speech and 1 is highly correlated

- storylines: a list of the different storylines found and which scenes belong to it. 
  - Specify where is the climax (scene and timestamp) and if the content is being presented a narrative story, or is it more like a collection of facts or non-narrative information
  - if there are scenes not matching storylines, explain how those scenes contribute to the video
- looking at the overall video and the storylines, which segments of the video could be trimmed to make it more dynamic?
- q&a: a list of 5 questions/answers about the video that focus on fine details (objects and or activities), overall story reasoning and mood. Focus on Q&A aspects captured on the audio and the video whenever possible difficult to get only by looking at the transcription.
```
<br>
<u><b>Adding Instructor</u></b>

Once the result was processed by Gemini, we parsed it with [Instructor](https://github.com/jxnl/instructor/): a library built on top of Pydantic to achieve structured outputs given a schema. See table with example below.

Instructor allowed us to experiment with different models to convert the free text from Gemini into the schema that we defined in Pydantic. We tried Gemini and GPT4o and we sticked with GPT4o given the higher success rate. 

<div style="text-align: center;margin: auto; width: 100%; font-size: 12px;">
<table>
  <thead>
    <tr>
      <th><strong>Video</strong></th>
      <th><strong>Gemini Output</strong></th>
      <th><strong>Instructor Output</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <iframe width="200" height="113" src="https://www.youtube.com/embed/-E9Te2S1gXY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
      </td>
      <td>
        <pre style="line-height: 1.0;"><code>

CharacterList:
Man Slim build, brown eyes, shaved sides, black hoodie with colorful logo, black pants.
Scenes
Scene 1
Start 0:00 End 0:55
Characters: [Man]
Activities:
Introduces bus
Describes peaceful location with cows
Props: Bus, cows, deck.
Mood:Excited, adventure.
Narrative Progression:
Introduction to bus.
Tour begins outside, highlighting nature and relaxation.
Dynamism Score 0.7 
Audio-Visual Correlation 1
        </code></pre>
      </td>
      <td>
        <pre style="line-height: 1.0;"><code>
{
  "title": "Bertie the Bus Tour",
  "description": "Guided tour of converted bus.",
  "characterList": [
    {
      "name": "Narrator",
      "description": "Slim build, brown eyes, shaved sides, black hoodie with colorful logo, black pants."
    }
  ],
  "scenes": [
    {
      "sceneId": 1,
      "title": "Introduction to Bus",
      "timestamps": {
        "start": "0:00",
        "end": "0:55"
      },
      "cast": ["Narrator"],
      "activities": [
        "Narrator speaks in front of bus",
        "Shows outdoor deck with chairs, cows nearby."
      ],
      "props": ["Bus", "Deck", "Cows"],
      "mood": "Excited, adventure."
    }
  ],
  "dynamismScore": 0.7,
  "audioVisualCorrelation": 1
}
        </code></pre>
      </td>
    </tr>
  </tbody>
</table>

</div>




<br>
It is worth highlighting that the content filtering in Gemini dropped some videos as this is something that can happen to you if you use Gemini. In our case, given the amount of content that we were targetting, the total minutes of content that were dropped by Gemini’s filtering was negligible.

The full code to annotate video can be found here [[link](https://github.com/mfarre/fineVideo/blob/main/contentannotation/video2annotation.py)].
<br>
## Fine Alignment and anomaly filtering

With the videos annotated and the data properly aligned to our schema, we look at the temporal domain of the data and we ensure its alignment with the video: Gemini 1.5 reads video at 1 frame per second and quite frequently videos have 25 - 29 frames per second. In our Fine Alignment we make sure scene boundaries provided by Gemini 1.5 match the correct frames in the video.

We also use this temporal alignment to discard cases were Gemini stopped providing useful data and a part of the video is wrongly annotated. Notice that thanks to dropping all content longer than 10+ minutes earlier in the pipeline, the number of videos with bad quality data was negligible (lower than 0.5%).

<center>
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/finevideo/fine-alignment.png" alt="Fine Alignment" style="width: 60%;">
    <figcaption style="font-style: italic;">Fine metadata - video scene boundary to shot alignment as a mechanism to discard outliers</figcaption>
    <br><br>
</center>

Link to video alignment code here [[link](https://github.com/mfarre/fineVideo/blob/main/finealignment/video_alignment.py)]

## Future Work

We are currently preparing the training of a multi-modal LLM trained with FineVideo, we plan to share the model weights and training recipe with the community as soon as it is completed.

We are also open to other extensions of FineVideo, speak up and tell us what you would like to see!