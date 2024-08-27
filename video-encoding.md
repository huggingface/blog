---
title: "Scaling robotics datasets with video encoding"
thumbnail: /blog/assets/video-encoding/thumbnail.png 
authors:
- user: aliberts
- user: cadene
- user: mfarre
---

# Scaling robotics datasets with video encoding

Over the past few years, text and image-based models have seen dramatic performance improvements, primarily due to scaling up model weights and dataset sizes. While the internet provides an extensive database of text and images for LLMs and image generation models, robotics lacks such a vast and diverse qualitative data source and efficient data formats. Despite efforts like [Open X](https://robotics-transformer-x.github.io/), we are still far from achieving the scale and diversity seen with Large Language Models. Additionally, we lack the necessary tools for this endeavor, such as dataset formats that are lightweight, fast to load from, easy to share and visualize online. This gap is what [🤗 LeRobot](https://github.com/huggingface/lerobot) aims to address.

## What's a dataset in robotics?

In their general form — at least the one we are interested in within an end-to-end learning framework — robotics datasets typically come in two modalities: the visual modality and the robot's proprioception / goal positions modality (state/action vectors). Here's what this can look like in practice:

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

Until now, the best way to store visual modality was png for individual frames. This is very redundant as there's a lot of repetition among the frames. People did not use videos because the loading times could be order of magnitude above. These datasets are usually released in various formats from academic papers (hdf5, zarr, pickle...).


## Motivation & contribution

These days, modern video codecs can achieve impressive compression ratios — meaning the size of the encoded video compared to the original uncompressed frames — while still preserving excellent quality. This means that with a compression ratio of 1:20, or 5% for instance (which is easily achievable), you get from a 20GB dataset down to a single GB of data. Because of this, we decided to use video encoding to store the visual modalities of our datasets.

We propose a `LeRobotDataset` format that is simple, lightweight, easy to share (with native integration to the hub) and easy to visualize.
Our datasets are on average 14% the size their original version (reaching up to 0.2% in the best case) while preserving full training capabilities on them by maintaining a very good level of quality. Additionally, we observed decoding times of video frames to follow this patern, depending on resolution:
- In the nominal case where we're decoding a single frame, our loading time is comparable to that of loading the frame from a compressed image (png).
- In the advantageous case where we're decoding multiple successive frames, our loading time is 25%-50% that of loading those frames from compressed images.


## But what is a codec? And what is video encoding & decoding actually doing?

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

- **Spatial Compression:** This is the same principle used in a compressed image like JPEG or PNG. Spatial compression uses the self-similarities of an image to reduce its size. For instance, a single frame of a video showing a blue sky will have large areas of similar color. Spatial compression takes advantage of this to compress these areas without losing much in quality.

- **Temporal Compression:** Rather than storing each frame *as is*, which takes up a lot of space, temporal compression calculates the differences between each frame and keeps only those differences (which are generally much smaller) in the encoded video stream. At decoding time, each frame is reconstructed by applying those differences back. Of course, this approach requires at least one frame of reference to start computing these differences with. In practice though, we use more than one placed at regular intervals. There are several reasons for this, which are detailed in [this article](https://aws.amazon.com/blogs/media/part-1-back-to-basics-gops-explained/). These "reference frames" are called keyframes or I-frames (for Intra-coded frames).

Thanks to these 2 ideas, video encoding is able to reduce the size of videos down to something manageable. Knowing this, the encoding process roughly looks like this:
1. Keyframes are determined based on user's specifications and scenes changes.
2. Those keyframes are compressed spatially.
3. The frames in-between are then compressed temporally as "differences" (also called P-frames or B-frames, more on these in the article linked above).
4. These differences themselvses are then compressed spatially.
5. This compressed data from I-frames, P-frames and B-frames is encoded into a bitstream.
6. That video bitstream is then packaged into a container format (MP4, MKV, AVI...) along with potentially other bitstreams (audio, subtitles) and metadata.
7. At this point, additional processing may be applied to reduce any visual distortions caused by compression and to ensure the overall video quality meets desired standards.

Obviously, this is a high-level summary of what's happening and there are a lot of moving parts and configuration choices to make in this process. Logically, we wanted to evaluate the best way of doing it given our needs and constraints, so we built a benchmark to assess this according to a number of criteria.

## Criteria

While size was the initial reason we decided to go with video encoding, we soon realized that there were other aspects to consider as well. Of course, decoding time is an important one for machine learning applications as we want to maximize to amount of time spent training rather than loading data. Quality needs to remains above a certain level as well so as to not degrade our policies performance. Lastly, one less obvious but equally important aspect is the compatibility of our encoded videos in order to be easily decoded and played on the majority of media player, web browser, devices etc. Having the ability to easily and quickly visualize the content of any of our datasets was a must-have feature for us.

To summarize, these are the criteria we wanted to optimize:
- **Size:** Impacts storage disk space and download times.
- **Decoding time:** Impacts training time.
- **Quality:** Impacts training accuracy.
- **Compatibility:** Impacts the ability to easily decode the video and visualize it across devices and platforms.

Obviously, some of these criteria are in direct contradiction: you can hardly e.g. reduce the file size without degrading quality and vice versa. The goal was therefore to find the best compromise overall.

Note that because of our specific use case and our needs, some encoding settings traditionally used for media consumption don't really apply to us. A good example of that is with [GOP](https://en.wikipedia.org/wiki/Group_of_pictures) (Group of Pictures) size. More on that in a bit.

## Metrics

Given those criteria, we chose metrics accordingly.

- **Size compression ratio (lower is better)**: as mentioned, this is the size of the encoded video over the size of its set of original, unencoded frames.

- **Load times ratio (lower is better)**: this is the time it take to decode a given frame from a video over the time it takes to load that frame from an individual image.

For quality, we looked at 3 commonly used metrics:

- **[Average Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error) (lower is better):** the average mean square error between each decoded frame and its corresponding original image over all requested timestamps, and also divided by the number of pixels in the image to be comparable across different image sizes.

- **[Average Peak Signal to Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (higher is better):** measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR indicates better quality.

- **[Average Structural Similarity Index Measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (higher is better):** evaluates the perceived quality of images by comparing luminance, contrast, and structure. SSIM values range from -1 to 1, where 1 indicates perfect similarity.

Additionally, we tried various levels of encoding quality to get a sense of what these metrics translate to visually. However, video encoding is designed to appeal to the human eye by taking advantage of several principles of how the human visual perception works, tricking our brains to maintain a level of perceived quality. This might have a different impact on a neural net. Therefore, besides these metrics and a visual check, it was important for us to also validate that the encoding did not degrade our policies performance by A/B testing it.

For compatibility, we don't have a metric *per se*, but it basically boils down to the video codec and the pixel format. For the video codec, the three that we chose (h264, h265 and AV1) are common and don't pose an issue. However, the pixel format is important as well and we found afterwards that on most browsers for instance, `yuv444p` is not supported and the video can't be decoded.

## Variables

#### Image content & size
We don't expect the same optimal settings for a dataset of images from a simulation, or from the real world in an apartment, or in a factory, or outdoor, or with lots of moving objects in the scene, etc. Similarly, loading times might not vary linearly with the image size (resolution).
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

After running this study, we switched to a different encoding from v1.6 on.

| codebase version | v1.5           | v1.6        |
| ---------------- | -------------- | ----------- |
| vcodec           | `libx264`      | `libsvtav1` |
| pix-fmt          | `yuv444p`      | `yuv420p`   |
| g                | `2`            | `2`         |
| crf              | `None` (=`23`) | `30`        |

We managed to gain more quality thanks to AV1 encoding while using the more compatible `yuv420p` pixel format.

### Sizes

We achieved an average compression ratio of about 14% across the total dataset sizes. Most of our datasets are reduced to under 40% of their original size, with some being less than 1%. These variations can be attributed to the diverse formats from which these datasets originate. Datasets with the highest size reductions often contain uncompressed images, allowing the encoder’s temporal and spatial compression to drastically reduce their sizes. On the other hand, datasets where images were already stored using a form of spatial compression (such as JPEG or PNG) have experienced less reduction in size. Other factors, such as image resolution, also affect the effectiveness of video compression.

<details>
  <summary><b> Table 1: Dataset sizes comparison </b></summary>
  <table>
    <thead>
        <tr>
            <th>repo_id</th>
            <th>raw</th>
            <th>ours (v1.6)</th>
            <th>ratio (ours/raw)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>lerobot/nyu_rot_dataset</td>
            <td>5.3MB</td>
            <td>318.2KB</td>
            <td>5.8%</td>
        </tr>
        <tr>
            <td>lerobot/pusht</td>
            <td>29.6MB</td>
            <td>7.5MB</td>
            <td>25.3%</td>
        </tr>
        <tr>
            <td>lerobot/utokyo_saytap</td>
            <td>55.4MB</td>
            <td>6.5MB</td>
            <td>11.8%</td>
        </tr>
        <tr>
            <td>lerobot/imperialcollege_sawyer_wrist_cam</td>
            <td>81.9MB</td>
            <td>3.8MB</td>
            <td>4.6%</td>
        </tr>
        <tr>
            <td>lerobot/utokyo_xarm_bimanual</td>
            <td>138.5MB</td>
            <td>8.1MB</td>
            <td>5.9%</td>
        </tr>
        <tr>
            <td>lerobot/unitreeh1_two_robot_greeting</td>
            <td>181.2MB</td>
            <td>79.0MB</td>
            <td>43.6%</td>
        </tr>
        <tr>
            <td>lerobot/usc_cloth_sim</td>
            <td>254.5MB</td>
            <td>23.7MB</td>
            <td>9.3%</td>
        </tr>
        <tr>
            <td>lerobot/unitreeh1_rearrange_objects</td>
            <td>283.3MB</td>
            <td>138.4MB</td>
            <td>48.8%</td>
        </tr>
        <tr>
            <td>lerobot/tokyo_u_lsmo</td>
            <td>335.7MB</td>
            <td>22.8MB</td>
            <td>6.8%</td>
        </tr>
        <tr>
            <td>lerobot/utokyo_pr2_opening_fridge</td>
            <td>360.6MB</td>
            <td>29.2MB</td>
            <td>8.1%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_pingpong_test</td>
            <td>480.9MB</td>
            <td>168.5MB</td>
            <td>35.0%</td>
        </tr>
        <tr>
            <td>lerobot/cmu_franka_exploration_dataset</td>
            <td>602.3MB</td>
            <td>18.2MB</td>
            <td>3.0%</td>
        </tr>
        <tr>
            <td>lerobot/unitreeh1_warehouse</td>
            <td>666.7MB</td>
            <td>236.9MB</td>
            <td>35.5%</td>
        </tr>
        <tr>
            <td>lerobot/cmu_stretch</td>
            <td>728.1MB</td>
            <td>38.7MB</td>
            <td>5.3%</td>
        </tr>
        <tr>
            <td>lerobot/asu_table_top</td>
            <td>737.6MB</td>
            <td>39.1MB</td>
            <td>5.3%</td>
        </tr>
        <tr>
            <td>lerobot/xarm_push_medium</td>
            <td>808.5MB</td>
            <td>15.9MB</td>
            <td>2.0%</td>
        </tr>
        <tr>
            <td>lerobot/xarm_push_medium_replay</td>
            <td>808.5MB</td>
            <td>17.8MB</td>
            <td>2.2%</td>
        </tr>
        <tr>
            <td>lerobot/xarm_lift_medium_replay</td>
            <td>808.6MB</td>
            <td>18.4MB</td>
            <td>2.3%</td>
        </tr>
        <tr>
            <td>lerobot/xarm_lift_medium</td>
            <td>808.6MB</td>
            <td>17.3MB</td>
            <td>2.1%</td>
        </tr>
        <tr>
            <td>lerobot/utokyo_pr2_tabletop_manipulation</td>
            <td>829.4MB</td>
            <td>40.6MB</td>
            <td>4.9%</td>
        </tr>
        <tr>
            <td>lerobot/utokyo_xarm_pick_and_place</td>
            <td>1.3GB</td>
            <td>54.6MB</td>
            <td>4.1%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_ziploc_slide</td>
            <td>1.3GB</td>
            <td>498.4MB</td>
            <td>37.2%</td>
        </tr>
        <tr>
            <td>lerobot/ucsd_kitchen_dataset</td>
            <td>1.3GB</td>
            <td>46.5MB</td>
            <td>3.4%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_gnm_cory_hall</td>
            <td>1.4GB</td>
            <td>85.6MB</td>
            <td>6.0%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_thread_velcro</td>
            <td>1.5GB</td>
            <td>1.1GB</td>
            <td>73.2%</td>
        </tr>
        <tr>
            <td>lerobot/austin_buds_dataset</td>
            <td>1.5GB</td>
            <td>87.8MB</td>
            <td>5.7%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_screw_driver</td>
            <td>1.5GB</td>
            <td>507.8MB</td>
            <td>33.1%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_cups_open</td>
            <td>1.6GB</td>
            <td>486.3MB</td>
            <td>30.4%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_towel</td>
            <td>1.6GB</td>
            <td>565.3MB</td>
            <td>34.0%</td>
        </tr>
        <tr>
            <td>lerobot/dlr_sara_grid_clamp</td>
            <td>1.7GB</td>
            <td>93.6MB</td>
            <td>5.5%</td>
        </tr>
        <tr>
            <td>lerobot/unitreeh1_fold_clothes</td>
            <td>2.0GB</td>
            <td>922.0MB</td>
            <td>44.5%</td>
        </tr>
        <tr>
            <td>lerobot/droid_100<sup>*</sup></td>
            <td>2.0GB</td>
            <td>443.0MB</td>
            <td>21.2%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_battery</td>
            <td>2.3GB</td>
            <td>770.5MB</td>
            <td>33.0%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_tape</td>
            <td>2.5GB</td>
            <td>829.6MB</td>
            <td>32.5%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_candy</td>
            <td>2.6GB</td>
            <td>833.4MB</td>
            <td>31.5%</td>
        </tr>
        <tr>
            <td>lerobot/conq_hose_manipulation</td>
            <td>2.7GB</td>
            <td>634.9MB</td>
            <td>23.4%</td>
        </tr>
        <tr>
            <td>lerobot/columbia_cairlab_pusht_real</td>
            <td>2.8GB</td>
            <td>84.8MB</td>
            <td>3.0%</td>
        </tr>
        <tr>
            <td>lerobot/dlr_sara_pour</td>
            <td>2.9GB</td>
            <td>153.1MB</td>
            <td>5.1%</td>
        </tr>
        <tr>
            <td>lerobot/dlr_edan_shared_control</td>
            <td>3.1GB</td>
            <td>138.4MB</td>
            <td>4.4%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_vinh_cup</td>
            <td>3.1GB</td>
            <td>1.0GB</td>
            <td>32.3%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_vinh_cup_left</td>
            <td>3.5GB</td>
            <td>1.1GB</td>
            <td>32.1%</td>
        </tr>
        <tr>
            <td>lerobot/ucsd_pick_and_place_dataset</td>
            <td>3.5GB</td>
            <td>125.8MB</td>
            <td>3.5%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_elevator</td>
            <td>3.7GB</td>
            <td>558.5MB</td>
            <td>14.8%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_shrimp</td>
            <td>3.9GB</td>
            <td>1.3GB</td>
            <td>34.6%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_wash_pan</td>
            <td>4.0GB</td>
            <td>1.1GB</td>
            <td>26.5%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_wipe_wine</td>
            <td>4.3GB</td>
            <td>1.2GB</td>
            <td>28.0%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_fork_pick_up</td>
            <td>4.6GB</td>
            <td>1.4GB</td>
            <td>31.6%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_cable_routing</td>
            <td>4.7GB</td>
            <td>309.3MB</td>
            <td>6.5%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_coffee</td>
            <td>4.7GB</td>
            <td>1.5GB</td>
            <td>31.3%</td>
        </tr>
        <tr>
            <td>lerobot/nyu_franka_play_dataset<sup>*</sup></td>
            <td>5.2GB</td>
            <td>192.1MB</td>
            <td>3.6%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_coffee_new</td>
            <td>6.1GB</td>
            <td>1.9GB</td>
            <td>31.5%</td>
        </tr>
        <tr>
            <td>lerobot/austin_sirius_dataset</td>
            <td>6.5GB</td>
            <td>428.7MB</td>
            <td>6.4%</td>
        </tr>
        <tr>
            <td>lerobot/cmu_play_fusion</td>
            <td>6.7GB</td>
            <td>470.2MB</td>
            <td>6.9%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_gnm_sac_son<sup>*</sup></td>
            <td>7.0GB</td>
            <td>501.4MB</td>
            <td>7.0%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_cabinet</td>
            <td>7.0GB</td>
            <td>1.6GB</td>
            <td>23.2%</td>
        </tr>
        <tr>
            <td>lerobot/nyu_door_opening_surprising_effectiveness</td>
            <td>7.1GB</td>
            <td>378.4MB</td>
            <td>5.2%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_mobile_chair</td>
            <td>7.4GB</td>
            <td>2.0GB</td>
            <td>27.2%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_fanuc_manipulation</td>
            <td>8.9GB</td>
            <td>312.8MB</td>
            <td>3.5%</td>
        </tr>
        <tr>
            <td>lerobot/jaco_play</td>
            <td>9.2GB</td>
            <td>411.1MB</td>
            <td>4.3%</td>
        </tr>
        <tr>
            <td>lerobot/viola</td>
            <td>10.4GB</td>
            <td>873.6MB</td>
            <td>8.2%</td>
        </tr>
        <tr>
            <td>lerobot/kaist_nonprehensile</td>
            <td>11.7GB</td>
            <td>203.1MB</td>
            <td>1.7%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_mvp</td>
            <td>12.3GB</td>
            <td>127.0MB</td>
            <td>1.0%</td>
        </tr>
        <tr>
            <td>lerobot/uiuc_d3field<sup>*</sup></td>
            <td>15.8GB</td>
            <td>1.4GB</td>
            <td>9.1%</td>
        </tr>
        <tr>
            <td>lerobot/umi_cup_in_the_wild</td>
            <td>16.8GB</td>
            <td>2.9GB</td>
            <td>17.6%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_sim_transfer_cube_human</td>
            <td>17.9GB</td>
            <td>66.7MB</td>
            <td>0.4%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_sim_insertion_scripted</td>
            <td>17.9GB</td>
            <td>67.6MB</td>
            <td>0.4%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_sim_transfer_cube_scripted</td>
            <td>17.9GB</td>
            <td>68.5MB</td>
            <td>0.4%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_gnm_recon<sup>*</sup></td>
            <td>18.7GB</td>
            <td>29.3MB</td>
            <td>0.2%</td>
        </tr>
        <tr>
            <td>lerobot/austin_sailor_dataset</td>
            <td>18.8GB</td>
            <td>1.1GB</td>
            <td>6.0%</td>
        </tr>
        <tr>
            <td>lerobot/utaustin_mutex</td>
            <td>20.8GB</td>
            <td>1.4GB</td>
            <td>6.6%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_static_pro_pencil</td>
            <td>21.1GB</td>
            <td>504.0MB</td>
            <td>2.3%</td>
        </tr>
        <tr>
            <td>lerobot/aloha_sim_insertion_human</td>
            <td>21.5GB</td>
            <td>87.3MB</td>
            <td>0.4%</td>
        </tr>
        <tr>
            <td>lerobot/stanford_kuka_multimodal_dataset</td>
            <td>32.0GB</td>
            <td>269.9MB</td>
            <td>0.8%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_rpt</td>
            <td>40.6GB</td>
            <td>1.1GB</td>
            <td>2.7%</td>
        </tr>
        <tr>
            <td>lerobot/roboturk<sup>*</sup></td>
            <td>45.4GB</td>
            <td>1.9GB</td>
            <td>4.1%</td>
        </tr>
        <tr>
            <td>lerobot/iamlab_cmu_pickup_insert</td>
            <td>50.3GB</td>
            <td>1.8GB</td>
            <td>3.6%</td>
        </tr>
        <tr>
            <td>lerobot/stanford_hydra_dataset</td>
            <td>72.5GB</td>
            <td>2.9GB</td>
            <td>4.0%</td>
        </tr>
        <tr>
            <td>lerobot/berkeley_autolab_ur5<sup>*</sup></td>
            <td>76.4GB</td>
            <td>14.4GB</td>
            <td>18.9%</td>
        </tr>
        <tr>
            <td>lerobot/stanford_robocook<sup>*</sup></td>
            <td>124.6GB</td>
            <td>3.8GB</td>
            <td>3.1%</td>
        </tr>
        <tr>
            <td>lerobot/toto</td>
            <td>127.7GB</td>
            <td>5.3GB</td>
            <td>4.1%</td>
        </tr>
        <tr>
            <td>lerobot/fmb<sup>*</sup></td>
            <td>356.5GB</td>
            <td>4.2GB</td>
            <td>1.2%</td>
        </tr>
    </tbody>
  </table>
  <p style="font-size: 0.9em;"><sup>*</sup>These datasets contain depth maps which were not included in our format.</p>
</details>

### Loading times
Thanks to video encoding, our loading times scale much better with the resolution. This is especially true in advantageous scenarios where we decode multiple successive frames.
<!-- TODO: changes urls when https://huggingface.co/datasets/huggingface/documentation-images/discussions/349 is merged -->
| 1 frame | 2 frames | 6 frames |
| ------- | -------- | -------- |
| ![Load_times_1_frame.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/Load_times_1_frame.png) | ![Load_times_2_frames.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/Load_times_2_frames.png) | ![Load_times_6_frames.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/Load_times_6_frames.png) |

### Summary
The full results of our study are available in [this spreadsheet](https://docs.google.com/spreadsheets/d/1OYJB43Qu8fC26k_OyoMFgGBBKfQRCi4BIuYitQnq3sw/edit?usp=sharing). The tables below show the averaged results for `g=2` and `crf=30`, using `backend=pyav` and in all timestamps-modes (`1_frame`, `2_frames`, `6_frames`).

<!-- #### Table 2: Ratio of video size and images size (lower is better)
|                                    |             | libx264    |         | libx265   |           | libsvtav1 |
| repo_id                            | Mega Pixels | yuv420p    | yuv444p | yuv420p   | yuv444p   | yuv420p   |
| ---------------------------------- | ----------- | ---------- | ------- | --------- | --------- | -------   |
| lerobot/pusht_image                | 0.01        | **16.97%** | 17.58%  | 18.57%    | 18.86%    | 22.06%    |
| aliberts/aloha_mobile_shrimp_image | 0.31        | 2.14%      | 2.11%   | 1.38%     | **1.37%** | 5.59%     |
| aliberts/paris_street              | 0.92        | 2.12%      | 2.13%   | **1.54%** | **1.54%** | 4.43%     |
| aliberts/kitchen                   | 2.07        | 1.40%      | 1.39%   | **1.00%** | **1.00%** | 2.52%     | -->

<details>
  <summary><b> Table 2: Ratio of video size and images size (lower is better) </b></summary>
  <table>
    <thead>
        <tr>
            <th rowspan="2">repo_id</th>
            <th rowspan="2">Mega Pixels</th>
            <th colspan="2">libx264</th>
            <th colspan="2">libx265</th>
            <th colspan="1">libsvtav1</th>
        </tr>
        <tr>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>lerobot/pusht_image</td>
            <td>0.01</td>
            <td><strong style="color:green;">16.97%</strong></td>
            <td>17.58%</td>
            <td>18.57%</td>
            <td>18.86%</td>
            <td>22.06%</td>
        </tr>
        <tr>
            <td>aliberts/aloha_mobile_shrimp_image</td>
            <td>0.31</td>
            <td>2.14%</td>
            <td>2.11%</td>
            <td>1.38%</td>
            <td><strong style="color:green;">1.37%</strong></td>
            <td>5.59%</td>
        </tr>
        <tr>
            <td>aliberts/paris_street</td>
            <td>0.92</td>
            <td>2.12%</td>
            <td>2.13%</td>
            <td><strong style="color:green;">1.54%</strong></td>
            <td><strong style="color:green;">1.54%</strong></td>
            <td>4.43%</td>
        </tr>
        <tr>
            <td>aliberts/kitchen</td>
            <td>2.07</td>
            <td>1.40%</td>
            <td>1.39%</td>
            <td><strong style="color:green;">1.00%</strong></td>
            <td><strong style="color:green;">1.00%</strong></td>
            <td>2.52%</td>
        </tr>
    </tbody>
  </table>
</details>

<!-- #### Table 3: Ratio of video and images loading times (lower is better)
|                                    |             | libx264 |         | libx265  |         | libsvtav1 |
| repo_id                            | Mega Pixels | yuv420p | yuv444p | yuv420p  | yuv444p | yuv420p   |
| ---------------------------------- | ----------- | ------- | ------- | -------- | ------- | --------  |
| lerobot/pusht_image                | 0.01        | 25.04   | 29.14   | **4.16** | 4.66    | 4.52      |
| aliberts/aloha_mobile_shrimp_image | 0.31        | 63.56   | 58.18   | 1.60     | 2.04    | **1.00**  |
| aliberts/paris_street              | 0.92        | 3.89    | 3.76    | 0.51     | 0.71    | **0.48**  |
| aliberts/kitchen                   | 2.07        | 2.68    | 1.94    | **0.36** | 0.58    | 0.38      | -->

<details>
  <summary><b> Table 3: Ratio of video and images loading times (lower is better) </b></summary>
  <table>
    <thead>
        <tr>
            <th rowspan="2">repo_id</th>
            <th rowspan="2">Mega Pixels</th>
            <th colspan="2">libx264</th>
            <th colspan="2">libx265</th>
            <th colspan="1">libsvtav1</th>
        </tr>
        <tr>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>lerobot/pusht_image</td>
            <td>0.01</td>
            <td>25.04</td>
            <td>29.14</td>
            <td><strong style="color:green;">4.16</strong></td>
            <td>4.66</td>
            <td>4.52</td>
        </tr>
        <tr>
            <td>aliberts/aloha_mobile_shrimp_image</td>
            <td>0.31</td>
            <td>63.56</td>
            <td>58.18</td>
            <td>1.60</td>
            <td>2.04</td>
            <td><strong style="color:green;">1.00</strong></td>
        </tr>
        <tr>
            <td>aliberts/paris_street</td>
            <td>0.92</td>
            <td>3.89</td>
            <td>3.76</td>
            <td>0.51</td>
            <td>0.71</td>
            <td><strong style="color:green;">0.48</strong></td>
        </tr>
        <tr>
            <td>aliberts/kitchen</td>
            <td>2.07</td>
            <td>2.68</td>
            <td>1.94</td>
            <td><strong style="color:green;">0.36</strong></td>
            <td>0.58</td>
            <td>0.38</td>
        </tr>
    </tbody>
  </table>
</details>

<!-- #### Table 4: Quality (mse: lower is better, psnr & ssim: higher is better)
|                                    |             |        | libx264  |              | libx265  |          | libsvtav1    |
| repo_id                            | Mega Pixels | Values | yuv420p  | yuv444p      | yuv420p  | yuv444p  | yuv420p      |
| ---------------------------------- | ----------- | ------ | -------- | ------------ | -------- | -------- | ------------ |
| lerobot/pusht_image                | 0.01        | mse    | 2.93E-04 | **2.09E-04** | 3.84E-04 | 3.02E-04 | 2.23E-04     |
|                                    |             | psnr   | 35.42    | 36.97        | 35.06    | 36.69    | **37.12**    |
|                                    |             | ssim   | 98.29%   | **98.83%**   | 98.17%   | 98.69%   | 98.70%       |
| aliberts/aloha_mobile_shrimp_image | 0.31        | mse    | 3.19E-04 | 3.02E-04     | 5.30E-04 | 5.17E-04 | **2.18E-04** |
|                                    |             | psnr   | 35.80    | 36.10        | 35.01    | 35.23    | **39.83**    |
|                                    |             | ssim   | 95.20%   | 95.20%       | 94.51%   | 94.56%   | **97.52%**   |
| aliberts/paris_street              | 0.92        | mse    | 5.34E-04 | 5.16E-04     | 9.18E-03 | 9.17E-03 | **3.09E-04** |
|                                    |             | psnr   | 33.55    | 33.75        | 29.96    | 30.06    | **35.41**    |
|                                    |             | ssim   | 93.94%   | 93.93%       | 83.11%   | 83.11%   | **95.50%**   |
| aliberts/kitchen                   | 2.07        | mse    | 2.32E-04 | 2.06E-04     | 6.87E-04 | 6.75E-04 | **1.32E-04** |
|                                    |             | psnr   | 36.77    | 37.38        | 35.27    | 35.50    | **39.20**    |
|                                    |             | ssim   | 95.47%   | 95.58%       | 95.11%   | 95.13%   | **96.84%**   | -->

<details>
  <summary><b> Table 4: Quality (mse: lower is better, psnr & ssim: higher is better) </b></summary>
  <table>
    <thead>
        <tr>
            <th rowspan="2">repo_id</th>
            <th rowspan="2">Mega Pixels</th>
            <th rowspan="2">Values</th>
            <th colspan="2">libx264</th>
            <th colspan="2">libx265</th>
            <th colspan="1">libsvtav1</th>
        </tr>
        <tr>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
            <th>yuv444p</th>
            <th>yuv420p</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">lerobot/pusht_image</td>
            <td rowspan="3">0.01</td>
            <td>mse</td>
            <td>2.93E-04</td>
            <td><strong style="color:green;">2.09E-04</strong></td>
            <td>3.84E-04</td>
            <td>3.02E-04</td>
            <td>2.23E-04</td>
        </tr>
        <tr>
            <td>psnr</td>
            <td>35.42</td>
            <td>36.97</td>
            <td>35.06</td>
            <td>36.69</td>
            <td><strong style="color:green;">37.12</strong></td>
        </tr>
        <tr>
            <td>ssim</td>
            <td>98.29%</td>
            <td><strong style="color:green;">98.83%</strong></td>
            <td>98.17%</td>
            <td>98.69%</td>
            <td>98.70%</td>
        </tr>
        <tr>
            <td rowspan="3">aliberts/aloha_mobile_shrimp_image</td>
            <td rowspan="3">0.31</td>
            <td>mse</td>
            <td>3.19E-04</td>
            <td>3.02E-04</td>
            <td>5.30E-04</td>
            <td>5.17E-04</td>
            <td><strong style="color:green;">2.18E-04</strong></td>
        </tr>
        <tr>
            <td>psnr</td>
            <td>35.80</td>
            <td>36.10</td>
            <td>35.01</td>
            <td>35.23</td>
            <td><strong style="color:green;">39.83</strong></td>
        </tr>
        <tr>
            <td>ssim</td>
            <td>95.20%</td>
            <td>95.20%</td>
            <td>94.51%</td>
            <td>94.56%</td>
            <td><strong style="color:green;">97.52%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">aliberts/paris_street</td>
            <td rowspan="3">0.92</td>
            <td>mse</td>
            <td>5.34E-04</td>
            <td>5.16E-04</td>
            <td>9.18E-03</td>
            <td>9.17E-03</td>
            <td><strong style="color:green;">3.09E-04</strong></td>
        </tr>
        <tr>
            <td>psnr</td>
            <td>33.55</td>
            <td>33.75</td>
            <td>29.96</td>
            <td>30.06</td>
            <td><strong style="color:green;">35.41</strong></td>
        </tr>
        <tr>
            <td>ssim</td>
            <td>93.94%</td>
            <td>93.93%</td>
            <td>83.11%</td>
            <td>83.11%</td>
            <td><strong style="color:green;">95.50%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">aliberts/kitchen</td>
            <td rowspan="3">2.07</td>
            <td>mse</td>
            <td>2.32E-04</td>
            <td>2.06E-04</td>
            <td>6.87E-04</td>
            <td>6.75E-04</td>
            <td><strong style="color:green;">1.32E-04</strong></td>
        </tr>
        <tr>
            <td>psnr</td>
            <td>36.77</td>
            <td>37.38</td>
            <td>35.27</td>
            <td>35.50</td>
            <td><strong style="color:green;">39.20</strong></td>
        </tr>
        <tr>
            <td>ssim</td>
            <td>95.47%</td>
            <td>95.58%</td>
            <td>95.11%</td>
            <td>95.13%</td>
            <td><strong style="color:green;">96.84%</strong></td>
        </tr>
    </tbody>
  </table>
</details>


### Policies

We validated that this new format did not impact performance on trained policies by training some of them on our format. The performance of those policies were on par with those trained on the image versions.

<details>
  <summary><b>Figure 1: Training curves for Diffusion policy on pusht dataset</b></summary>
  <div style="text-align: center; margin-bottom: 20px;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/train-pusht.png" target="_blank">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/train-pusht.png" alt="train-pusht" style="width: 85%;">
    </a>
  </div>
</details>

<details>
  <summary><b>Figure 2: Training curves for ACT policy on an aloha dataset</b></summary>
  <div style="text-align: center; margin-bottom: 20px;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/train-aloha.png" target="_blank">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/e16c03dc5cd17c3614310ba32267698b2398de45/blog/video-encoding/train-aloha.png" alt="train-aloha" style="width: 85%;">
    </a>
  </div>
</details>
<br>

Policies have also been trained and evaluated on AV1-encoded datasets and compared against our previous reference (h264):

- Diffusion on pusht:
  - [h264-encoded run](https://wandb.ai/aliberts/lerobot/runs/zubx2fwe/workspace?nw=nwuseraliberts)
  - [AV1-encoded run](https://wandb.ai/aliberts/lerobot/runs/mlestyd6/workspace?nw=nwuseraliberts)
- ACT on aloha_sim_transfer_cube_human:
  - [h264-encoded run](https://wandb.ai/aliberts/lerobot/runs/le8scox9?nw=nwuseraliberts)
  - [AV1-encoded run](https://wandb.ai/aliberts/lerobot/runs/rz454evx/workspace?nw=nwuseraliberts)
- ACT on aloha_sim_insertion_scripted:
  - [h264-encoded run](https://wandb.ai/aliberts/lerobot/runs/r6q2bsq4/workspace?nw=nwuseraliberts)
  - [AV1-encoded run](https://wandb.ai/aliberts/lerobot/runs/4abyvtcz/workspace?nw=nwuseraliberts)


## Future work

Video encoding/decoding is a vast and complex subject, and we're only scratching the surface here. Here are some of the things we left over in this experiment:

For the encoding, additional encoding parameters exist that are not included in this benchmark. In particular:
- `-preset` which allows for selecting encoding presets. This represents a collection of options that will provide a certain encoding speed to compression ratio. By leaving this parameter unspecified, it is considered to be `medium` for libx264 and libx265 and `8` for libsvtav1.
- `-tune` which allows to optimize the encoding for certain aspects (e.g. film quality, live, etc.). In particular, a `fast decode` option is available to optimise the encoded bit stream for faster decoding.
- two-pass encoding would also be interesting to look at as it increases quality, although it is likely to increase encoding time significantly. Note that since we are primarily interested in decoding performance (as encoding is only done once before uploading a dataset), we did not measure encoding times nor have any metrics regarding encoding. Using a 1-pass encoding did not pose any issue and it didn't take a significant amount of time during this benchmark (with the condition of using `libsvtav1` instead of `libaom` for AV1 encoding).

The more detailed and comprehensive list of these parameters and others is available on the codecs documentations:
- h264: https://trac.ffmpeg.org/wiki/Encode/H.264
- h265: https://trac.ffmpeg.org/wiki/Encode/H.265
- AV1: https://trac.ffmpeg.org/wiki/Encode/AV1

Similarly on the decoding side, other decoders exist but are not implemented in our current benchmark. To name a few:
- `torchcodec`
- `torchaudio`
- `ffmpegio`
- `decord`
- `nvc`

Finally, we did not look into video encoding with depth maps. Although we did port datasets that include depth maps images, we are not using that modality for now.