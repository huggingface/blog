---
title: "Using video datasets for Robotics Application"
thumbnail: /blog/assets/video-encoding/thumbnail.png 
authors:
- user: aliberts
- user: cadene
---

# Video encoding: A case study for video datasets in robotics

Building a library for end-to-end machine learning applied to robotics such as [🤗 LeRobot](https://github.com/huggingface/lerobot) comes with many challenges. A big chunk of these is the dataset aspect of things. While we have been lucky with LLMs and image generation models to have a huge database of text and images in the form of the internet, we have not been so lucky with robotics. In their general form — at least the one we are interested in within an end-to-end learning framework — they come in 2 modalities: videos and state/action vectors (the robot's proprioception and the goal positions). Here's what this can look like in practice:

<center>
    <iframe 
        width="560" 
        height="315" 
        src="https://www.youtube.com/embed/69rEK7eSBAk?si=fE4Z2Ax6pP3vazUH"
        title="Aloha static battery video player"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen>
    </iframe>
</center>

These datasets are usually released in various formats from academic papers (hdf5, zarr, pickle...), and most of the time the "video" modality is actually a series of images which can themselves be compressed (e.g. in a png format) or uncompressed.

## Motivation

One of the first things we wanted to do with this project was to try and have a standard format for these datasets. We came up with a simple `LeRobotDataset` class and in doing so came the question of how to handle the video modality. Of course, one important aspect in building this interface was to have it easily integrate with the Hugging Face Hub ecosystem so that they can be shared easily. If the community and us are to upload hundreds of datasets to the platform, optimizing size is critical, both for disk space and download times.

These days, modern video codec can achieve impressive compression ratios — that is the size of the encoded video over the size of its set of original unencoded frames — while having decent quality. This means that with a ratio compression ratio of 1:20, or 5% for instance (which is easily achievable), you get from a 20GB dataset down to a single GB of data.

For this reason — at least initially — we decided to use video encoding for the video modalities of our datasets.

## But what is a codec? And what exactly is video encoding & decoding actually doing?

<center>
    <iframe 
        width="560" 
        height="315" 
        src="https://www.youtube.com/embed/7YQ1mikDhIo?si=YRx_Rlq0c3pjJYAm"
        title="Video Codecs 101 video player"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen>
    </iframe>
</center>

At its core, video encoding reduces the size of videos by using mainly 2 ideas:

- **Spatial Compression:** This is the same principle used in a compressed image like jpeg or png. Spatial compression uses the self-similarities of an image to reduce its size. For instance, a single frame of a video showing a blue sky will have large areas of similar color. Spatial compression takes advantage of this to compress these areas without losing much in quality.

- **Temporal Compression:** Rather than storing each frame *as is*, which takes up a lot of space, temporal compression calculates the differences between each frame and keeps only those differences (which are generally much smaller) in the encoded video stream. At decoding time, each frame is reconstructed by applying those differences back. Of course, this approach requires at least one frame of reference to start computing these differences with. In practice though, we use more than one placed at regular intervals. There are several reasons of why that is that are detailed in [this article](https://aws.amazon.com/blogs/media/part-1-back-to-basics-gops-explained/). These "reference frames" are called keyframes or I-frames (for Intra-coded frames).

Thanks to these 2 ideas, video encoding is able to reduce the size of videos down to something manageable. Knowing this, the encoding process roughly looks like this:
1. Keyframes are determined based on user's specifications and scenes changes.
2. Those keyframes are compressed spatially.
3. The frames in-between are then compressed temporally as "differences" (P-frames or B-frames).
4. These differences themselvses are then compressed spatially.
5. This compressed data from I-frames, P-frames and B-frames is encoded into a bitstream.
6. That video bitstream is then packaged into a container format (MP4, MKV, AVI...) along with potentially other bitstreams (audio, subtitles) and metadata.
7. At this point, some post processing may be applied to reduce compression artifacts and ensure quality.

Obviously, this is a high-level summary of what's happening and there are a lot of moving parts and configuration choices to make in this process. Logically, we wanted to evaluate the best way of doing it given our needs and constraints, so we built a benchmark to assess this according to a number of criteria.

## Criteria

While size was the initial reason we decided to go with video encoding, we soon realized that there are other aspects to consider as well. Of course, decoding time is an important one for machine learning applications as we want to maximize to amount of time spent training rather than loading data. Quality needs to remains above a certain level as well so as to not degrade our policies performance. Lastly, one less obvious but equally important aspect is the compatibility of our encoded videos in order to be easily decoded and played on the majority of media player, web browser, devices etc. Having the ability to easily and quickly visualize the content of any of our datasets was a must-have feature for us.

To summarize, these are the criteria we wanted to optimize:
- **Size:** Impacts storage disk space and download times.
- **Decoding time:** Impacts training time.
- **Quality:** Impacts training accuracy.
- **Compatibility:** Impacts the ability to easily decode the video and visualize it across devices and platforms.

Obviously, some of these criteria are in direct contradiction: you can hardly e.g. reduce the file size without degrading quality and vice versa. The goal was therefore to find the best compromise overall.

Note that because of our specific use case and our needs, some encoding settings traditionnally used for media consumption don't really apply to us. A good example of that is with [GOP](https://en.wikipedia.org/wiki/Group_of_pictures) (Group of Pictures) size. More on that in a bit.

## Metrics

Given those criteria, we chose metrics accordingly.

- **Size compression ratio (lower is better)**: as mentionned, this is the size of the encoded video over the size of its set of original, unencoded frames.

- **Load times ratio (lower is better)**: this is the time it take to decode a given frame from a video over the time it takes to load that frame from an individual image.

For quality, we looked at 3 commonly used metrics:

- **[Average Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error) (lower is better):** the average mean square error between each decoded frame and its corresponding original image over all requested timestamps, and also divided by the number of pixels in the image to be comparable across different image sizes.

- **[Average Peak Signal to Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (higher is better):** measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR indicates better quality.

- **[Average Structural Similarity Index Measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (higher is better):** evaluates the perceived quality of images by comparing luminance, contrast, and structure. SSIM values range from -1 to 1, where 1 indicates perfect similarity.

Additionally, we checked visually various levels of encoding quality to get a sense of what these metrics translate to visually. However, video encoding is designed to appeal to the human eye by taking advantage of several principles of how the human visual perception works, tricking our brains to maintain a level of perceived quality. This might have a different impact on a neural net. Therefore, besides these metrics and a visual check, it was important for us to also validate that the encoding did not degrade our policies performance by A/B testing it.

For compatibility, we don't have a metric *per se*, but it basically boils down to the video codec and the pixel format. For the video codec, the three that we chose (h264, h265 and AV1) are common and don't pose an issue. However, the pixel format is important as well and we found afterwards that on most browsers for instance, `yuv444p` is not supported and the video can't be decoded.

## Variables

#### Image content & size
We don't expect the same optimal settings for a dataset of images from a simulation, or from real-world in an appartment, or in a factory, or outdoor, or with lots of moving objects in the scene, etc. Similarly, loading times might not vary linearly with the image size (resolution).
For these reasons, we ran this benchmark on four representative datasets:
- `lerobot/pusht_image`: (96 x 96 pixels) simulation with simple geometric shapes, fixed camera.
- `aliberts/aloha_mobile_shrimp_image`: (480 x 640 pixels) real-world indoor, moving camera.
- `aliberts/paris_street`: (720 x 1280 pixels) real-world outdoor, moving camera.
- `aliberts/kitchen`: (1080 x 1920 pixels) real-world indoor, fixed camera.

### Encoding parameters

We used [FFmpeg](https://www.ffmpeg.org/) for encoding our videos. Here are the main parameters we played with:

#### Video Codec (`vcodec`)
The [codec](https://en.wikipedia.org/wiki/Video_codec) (**co**der-**dec**oder) is the algorithmic engine that's driving the video encoding. The codec defines a format used for encoding and decoding. Note that for a given codec, several implementations may exist. For example for AV1: `libaom` (official implementation), `libsvtav1` (faster, encoder only), `libdav1d` (decoder only).

Note that the rest of the encoding parameters are interpreted differently depending on the video codec used. In other words, the same `crf` value used with one codec doesn't necessarily translate into the same compression level with another codec. In fact, the default value (`None`) isn't the same amongst the different video codecs. Importantly, it is also the case for many other ffmpeg arguments like `g` which specifies the frequency of the key frames.

#### Pixel Format (`pix_fmt`)
Pixel format specifies both the [color space](https://en.wikipedia.org/wiki/Color_space) (YUV, RGB, Grayscale) and, for YUV color space, the [chroma subsampling](https://en.wikipedia.org/wiki/Chroma_subsampling) which determines the way chrominance (color information) and luminance (brightness information) are actually stored in the resulting encoded bitstream. For instance, `yuv420p` indicates YUV color space with 4:2:0 chroma subsampling. This is the most common format for web video and standard playback. For RGB color space, this parameter specifies the number of bits per pixel (e.g. `rbg24` means RGB color space with 24 bits per pixel).

#### Group of Pictures size (`g`)
[GOP](https://en.wikipedia.org/wiki/Group_of_pictures) (Group of Pictures) size determines how frequently keyframes are placed throughout the encoded bitstream. The lower that value is, the more frequently keyframes are placed. One key thing to understand is that when requesting a frame at a given timestamp, unless that frame happens to be a keyframe itself, the decoder will look for the last previous keyframe before that timestamp and will need to decode each subsequent frame up to the requested timestamp. This means that increasing GOP size will increase the average decoding time of a frame as fewer keyframes are available to start from. For a typical online content such as a video on Youtube or a movie on Netflix, a keyframe placed every 2 to 4 seconds of the video — 2s corresponding to a GOP size of 48 for a 24 fps video — will generally translate to a smooth viewer experience as this makes loading time acceptable for that use case (depending on hardware). For training a policy however, we need access to any frame as fast as possible meaning that we'll probably need a much lower value of GOP.

#### Constant Rate Factor (`crf`)
The constant rate factor represent the amount of lossy compression applied. A value of 0 means that no information is lost while a high value (around 50-60 depending on the codec used) is very lossy.
Using this parameter rather than specifying a target bitrate is [preferable](https://www.dr-lex.be/info-stuff/videotips.html#bitrate) since it allows to aim for a constant visual quality level with a potentially variable bitrate rather than the opposite.

<!-- #TODO: Merge PR https://huggingface.co/datasets/huggingface/documentation-images/discussions/349 and link images there  -->
| crf | libx264 | libx265 | libsvtav1 |
|-----|---------|---------|-----------|
| `10` | ![libx264_yuv420p_2_10.png](https://github.com/huggingface/lerobot/assets/75076266/f7e263f5-9c58-4987-8adb-e7ecc7909b80) | ![libx265_yuv420p_2_10.png](https://github.com/huggingface/lerobot/assets/75076266/98f94cbb-24b9-48aa-a4c4-dd1a3b149534) | ![libsvtav1_yuv420p_2_10.png](https://github.com/huggingface/lerobot/assets/75076266/bcd2db64-12a8-4024-b3d0-3fec2441adf7) |
| `30` | ![libx264_yuv420p_2_30.png](https://github.com/huggingface/lerobot/assets/75076266/f2bf0600-5e2e-45f3-830c-3b3913be0ca0) | ![libx265_yuv420p_2_30.png](https://github.com/huggingface/lerobot/assets/75076266/9a12dc40-49a6-4210-a061-6f2c7165ab44) | ![libsvtav1_yuv420p_2_30.png](https://github.com/huggingface/lerobot/assets/75076266/1fd3b0a1-3831-4be4-8ec5-ab5666984371) |
| `50` | ![libx264_yuv420p_2_50.png](https://github.com/huggingface/lerobot/assets/75076266/019fbe0e-b543-46b8-800e-61b597bf955a) | ![libx265_yuv420p_2_50.png](https://github.com/huggingface/lerobot/assets/75076266/fc5d8602-403c-4625-88dd-9a177e122cbb) | ![libsvtav1_yuv420p_2_50.png](https://github.com/huggingface/lerobot/assets/75076266/c395a505-004c-4a15-a4e2-fe735479a69d) |

This table summarizes the different values we tried for our study:
| parameter   | values                                                       |
|-------------|--------------------------------------------------------------|
| **vcodec**  | `libx264`, `libx265`, `libsvtav1`                            |
| **pix_fmt** | `yuv444p`, `yuv420p`                                         |
| **g**       | `1`, `2`, `3`, `4`, `5`, `6`, `10`, `15`, `20`, `40`, `None` |
| **crf**     | `0`, `5`, `10`, `15`, `20`, `25`, `30`, `40`, `50`, `None`   |

### Decoding parameters

<!-- TODO(aliberts): Add text here -->

#### Decoder
We tested two video decoding backends from torchvision:
- `pyav` (default)
- `video_reader`

#### Timestamps scenarios
Given the way video decoding works, once a keyframe has been loaded, the decoding of subsequent frames is fast.
This of course is affected by the `-g` parameter during encoding, which specifies the frequency of the keyframes. Given our typical use cases in robotics policies which might request a few timestamps in different random places, we want to replicate these use cases with the following scenarios:
- `1_frame`: 1 frame,
- `2_frames`: 2 consecutive frames (e.g. `[t, t + 1 / fps]`),
- `6_frames`: 6 consecutive frames (e.g. `[t + i / fps for i in range(6)]`)

Note that this differs significantly from a typical use case like watching a movie, in which every frame is loaded sequentially from the beginning to the end and it's acceptable to have big values for `-g`.

Additionally, because some policies might request single timestamps that are a few frames appart, we also have the following scenario:
- `2_frames_4_space`: 2 frames with 4 consecutive frames of spacing in between (e.g `[t, t + 5 / fps]`),

However, due to how video decoding is implemented with `pyav`, we don't have access to an accurate seek so in practice this scenario is essentially the same as `6_frames` since all 6 frames between `t` and `t + 5 / fps` will be decoded.


## Results

### Benchmark
The full results of our study are available in [this spreadsheet](https://docs.google.com/spreadsheets/d/1OYJB43Qu8fC26k_OyoMFgGBBKfQRCi4BIuYitQnq3sw/edit?usp=sharing). The tables below show a summary of the results for `g=2` and `crf=30`, using `timestamps-modes=6_frames` and `backend=pyav`

| video_images_size_ratio (lower is better) | vcodec     | pix_fmt |           |           |           |
| ----------------------------------------- | ---------- | ------- | --------- | --------- | --------- |
|                                           | libx264    |         | libx265   |           | libsvtav1 |
| repo_id                                   | yuv420p    | yuv444p | yuv420p   | yuv444p   | yuv420p   |
| lerobot/pusht_image                       | **16.97%** | 17.58%  | 18.57%    | 18.86%    | 22.06%    |
| aliberts/aloha_mobile_shrimp_image        | 2.14%      | 2.11%   | 1.38%     | **1.37%** | 5.59%     |
| aliberts/paris_street                     | 2.12%      | 2.13%   | **1.54%** | **1.54%** | 4.43%     |
| aliberts/kitchen                          | 1.40%      | 1.39%   | **1.00%** | **1.00%** | 2.52%     |


| video_images_load_time_ratio (lower is better) | vcodec  | pix_fmt |          |         |           |
| ---------------------------------------------- | ------- | ------- | -------- | ------- | --------- |
|                                                | libx264 |         | libx265  |         | libsvtav1 |
| repo_id                                        | yuv420p | yuv444p | yuv420p  | yuv444p | yuv420p   |
| lerobot/pusht_image                            | 6.45    | 5.19    | **1.90** | 2.12    | 2.47      |
| aliberts/aloha_mobile_shrimp_image             | 11.80   | 7.92    | 0.71     | 0.85    | **0.48**  |
| aliberts/paris_street                          | 2.21    | 2.05    | 0.36     | 0.49    | **0.30**  |
| aliberts/kitchen                               | 1.46    | 1.46    | 0.28     | 0.51    | **0.26**  |


|                                    |          | vcodec   | pix_fmt      |          |           |              |
|------------------------------------|----------|----------|--------------|----------|-----------|--------------|
|                                    |          | libx264  |              | libx265  |           | libsvtav1    |
| repo_id                            | metric   | yuv420p  | yuv444p      | yuv420p  | yuv444p   | yuv420p      |
| lerobot/pusht_image                | avg_mse  | 2.90E-04 | **2.03E-04** | 3.13E-04 | 2.29E-04  | 2.19E-04     |
|                                    | avg_psnr | 35.44    | 37.07        | 35.49    | **37.30** | 37.20        |
|                                    | avg_ssim | 98.28%   | **98.85%**   | 98.31%   | 98.84%    | 98.72%       |
| aliberts/aloha_mobile_shrimp_image | avg_mse  | 2.76E-04 | 2.59E-04     | 3.17E-04 | 3.06E-04  | **1.30E-04** |
|                                    | avg_psnr | 35.91    | 36.21        | 35.88    | 36.09     | **40.17**    |
|                                    | avg_ssim | 95.19%   | 95.18%       | 95.00%   | 95.05%    | **97.73%**   |
| aliberts/paris_street              | avg_mse  | 6.89E-04 | 6.70E-04     | 4.03E-03 | 4.02E-03  | **3.09E-04** |
|                                    | avg_psnr | 33.48    | 33.68        | 32.05    | 32.15     | **35.40**    |
|                                    | avg_ssim | 93.76%   | 93.75%       | 89.46%   | 89.46%    | **95.46%**   |
| aliberts/kitchen                   | avg_mse  | 2.50E-04 | 2.24E-04     | 4.28E-04 | 4.18E-04  | **1.53E-04** |
|                                    | avg_psnr | 36.73    | 37.33        | 36.56    | 36.75     | **39.12**    |
|                                    | avg_ssim | 95.47%   | 95.58%       | 95.52%   | 95.53%    | **96.82%**   |

### Policies
<!-- TODO(aliberts): Add training runs and results here -->

### Encoded datasets

After running this study, we decided to switch to a different encoding from v1.6 on.

| codebase version | `v1.5`         | `v1.6`      |
| ---------------- | -------------- | ----------- |
| vcodec           | `libx264`      | `libsvtav1` |
| pix-fmt          | `yuv444p`      | `yuv420p`   |
| g                | `2`            | `2`         |
| crf              | `None` (=`23`) | `30`        |


Considering this, we can compare the repo size (so the total size of the dataset, including non-video modalities) from the raw format vs. our format using video encoding.

| repo_id                                    | raw size | v1.5 size | v1.6 size | ratio (v1.6/raw) |
| ------------------------------------------ | -------- | --------- | --------- | ---------------- |
| `lerobot/pusht`                            | 29.6MB   | 12.9MB    | 7.5MB     | 25.3%            |
| `lerobot/unitreeh1_two_robot_greeting`     | 181.2MB  | N/A       | 79.0MB    | 43.6%            |
| `lerobot/unitreeh1_rearrange_objects`      | 283.3MB  | N/A       | 138.4MB   | 48.8%            |
| `lerobot/aloha_static_pingpong_test`       | 480.9MB  | 151.2MB   | 168.5MB   | 35.0%            |
| `lerobot/unitreeh1_warehouse`              | 666.7MB  | N/A       | 236.9MB   | 35.5%            |
| `lerobot/xarm_push_medium`                 | 808.5MB  | 13.6MB    | 15.9MB    | 2.0%             |
| `lerobot/xarm_push_medium_replay`          | 808.5MB  | 13.6MB    | 17.8MB    | 2.2%             |
| `lerobot/xarm_lift_medium`                 | 808.6MB  | 13.1MB    | 17.3MB    | 2.1%             |
| `lerobot/xarm_lift_medium_replay`          | 808.6MB  | 13.6MB    | 18.4MB    | 2.3%             |
| `lerobot/aloha_static_ziploc_slide`        | 1.3GB    | 418.5MB   | 498.4MB   | 37.2%            |
| `lerobot/aloha_static_screw_driver`        | 1.5GB    | 461.3MB   | 507.8MB   | 33.1%            |
| `lerobot/aloha_static_thread_velcro`       | 1.5GB    | 1023.0MB  | 1.1GB     | 73.2%            |
| `lerobot/aloha_static_cups_open`           | 1.6GB    | 459.9MB   | 486.3MB   | 30.4%            |
| `lerobot/aloha_static_towel`               | 1.6GB    | 534.1MB   | 565.3MB   | 34.0%            |
| `lerobot/unitreeh1_fold_clothes`           | 2.0GB    | N/A       | 922.0MB   | 44.5%            |
| `lerobot/aloha_static_battery`             | 2.3GB    | 712.6MB   | 770.5MB   | 33.0%            |
| `lerobot/aloha_static_tape`                | 2.5GB    | 769.8MB   | 829.6MB   | 32.5%            |
| `lerobot/aloha_static_candy`               | 2.6GB    | 799.2MB   | 833.4MB   | 31.5%            |
| `lerobot/aloha_static_vinh_cup`            | 3.1GB    | 979.6MB   | 1.0GB     | 32.3%            |
| `lerobot/aloha_static_vinh_cup_left`       | 3.5GB    | 1.1GB     | 1.1GB     | 32.1%            |
| `lerobot/aloha_mobile_elevator`            | 3.7GB    | 600.6MB   | 558.5MB   | 14.8%            |
| `lerobot/aloha_mobile_shrimp`              | 3.9GB    | 1.1GB     | 1.3GB     | 34.6%            |
| `lerobot/aloha_mobile_wash_pan`            | 4.0GB    | 939.8MB   | 1.1GB     | 26.5%            |
| `lerobot/aloha_mobile_wipe_wine`           | 4.3GB    | 1.1GB     | 1.2GB     | 28.0%            |
| `lerobot/aloha_static_fork_pick_up`        | 4.6GB    | 1.3GB     | 1.4GB     | 31.6%            |
| `lerobot/aloha_static_coffee`              | 4.7GB    | 1.4GB     | 1.5GB     | 31.3%            |
| `lerobot/aloha_static_coffee_new`          | 6.1GB    | 1.8GB     | 1.9GB     | 31.5%            |
| `lerobot/aloha_mobile_cabinet`             | 7.0GB    | 1.5GB     | 1.6GB     | 23.2%            |
| `lerobot/aloha_mobile_chair`               | 7.4GB    | 1.9GB     | 2.0GB     | 27.2%            |
| `lerobot/umi_cup_in_the_wild`              | 16.8GB   | 2.7GB     | 2.9GB     | 17.6%            |
| `lerobot/aloha_sim_transfer_cube_human`    | 17.9GB   | 49.9MB    | 66.7MB    | 0.4%             |
| `lerobot/aloha_sim_insertion_scripted`     | 17.9GB   | 50.0MB    | 67.6MB    | 0.4%             |
| `lerobot/aloha_sim_transfer_cube_scripted` | 17.9GB   | 50.3MB    | 68.5MB    | 0.4%             |
| `lerobot/aloha_static_pro_pencil`          | 21.1GB   | 397.9MB   | 504.0MB   | 2.3%             |
| `lerobot/aloha_sim_insertion_human`        | 21.5GB   | 64.0MB    | 87.3MB    | 0.4%             |
<!-- TODO(aliberts): Add open X repos here once they're pushed -->

We managed to gain more quality thanks to AV1 encoding while using the more compatible `yuv420p` pixel format.
Thankfully, the size remains similar with an average total compression ratio of about 25%.

## Caveats

Video encoding/decoding is a vast and complex subject, and we're only scratching the surface here. Here are the things we left over in this experiment:

For the encoding, additional encoding parameters exist that are not included in this benchmark. In particular:
- `-preset` which allows for selecting encoding presets. This represents a collection of options that will provide a certain encoding speed to compression ratio. By leaving this parameter unspecified, it is considered to be `medium` for libx264 and libx265 and `8` for libsvtav1.
- `-tune` which allows to optimize the encoding for certains aspects (e.g. film quality, live, etc.). In particular, a `fast decode` option is available to optimise the encoded bit stream for faster decoding.

The more detailed and comprehensive list of these parameters and others is available on the codecs documentations:
- h264: https://trac.ffmpeg.org/wiki/Encode/H.264
- h265: https://trac.ffmpeg.org/wiki/Encode/H.265
- AV1: https://trac.ffmpeg.org/wiki/Encode/AV1

Similarly on the decoding side, other decoders exist but are not implemented in our current benchmark. To name a few:
- `torchaudio`
- `ffmpegio`
- `decord`
- `nvc`

Note as well that since we are mostly interested in the performance at decoding time (also because encoding is done only once before uploading a dataset), we did not measure encoding times nor have any metrics regarding encoding.
However, besides the necessity to build ffmpeg from source, encoding did not pose any issue and it didn't take a significant amount of time during this benchmark.


## Conclusions

As we are looking to build the largest set of publicly available datasets for robotics thanks to the community, spending time on the basics of how those datasets are built and stored is important.
<!-- TODO -->