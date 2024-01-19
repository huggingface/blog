---
title: "How to host a Unity game in a Space"
thumbnail: /blog/assets/124_ml-for-games/unity-in-spaces-thumbnail.png
authors:
- user: dylanebert
---

# How to host a Unity game in a Space

<!-- {authors} --> 


Did you know you can host a Unity game in a Hugging Face Space? No? Well, you can!

Hugging Face Spaces are an easy way to build, host, and share demos. While they are typically used for Machine Learning demos, 
they can also host playable Unity games. Here are some examples:
- [Huggy](https://huggingface.co/spaces/ThomasSimonini/Huggy)
- [Farming Game](https://huggingface.co/spaces/dylanebert/FarmingGame) 
- [Unity API Demo](https://huggingface.co/spaces/dylanebert/UnityDemo)

Here's how you can host your own Unity game in a Space.

## Step 1: Create a Space using the Static HTML template

First, navigate to [Hugging Face Spaces](https://huggingface.co/new-space) to create a space.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/1.png">
</figure> 

Select the "Static HTML" template, give your Space a name, and create it.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/2.png">
</figure> 

## Step 2: Use Git to Clone the Space

Clone your newly created Space to your local machine using Git. You can do this by running the following command in your terminal or command prompt:

```
git clone https://huggingface.co/spaces/{your-username}/{your-space-name}
```

## Step 3: Open your Unity Project

Open the Unity project you want to host in your Space.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/3.png">
</figure> 

## Step 4: Switch the Build Target to WebGL

Navigate to `File > Build Settings` and switch the Build Target to WebGL.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/4.png">
</figure> 

## Step 5: Open Player Settings

In the Build Settings window, click the "Player Settings" button to open the Player Settings panel.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/5.png">
</figure> 

## Step 6: Optionally, Download the Hugging Face Unity WebGL Template

You can enhance your game's appearance in a Space by downloading the Hugging Face Unity WebGL template, available [here](https://github.com/huggingface/Unity-WebGL-template-for-Hugging-Face-Spaces). Just download the repository and drop it in your project files.

Then, in the Player Settings panel, switch the WebGL template to Hugging Face. To do so, in Player Settings, click "Resolution and Presentation", then select the Hugging Face WebGL template.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/6.png">
</figure> 

## Step 7: Change the Compression Format to Disabled

In the Player Settings panel, navigate to the "Publishing Settings" section and change the Compression Format to "Disabled".

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/7.png">
</figure> 

## Step 8: Build your Project

Return to the Build Settings window and click the "Build" button. Choose a location to save your build files, and Unity will build the project for WebGL.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/8.png">
</figure> 

## Step 9: Copy the Contents of the Build Folder

After the build process is finished, navigate to the folder containing your build files. Copy the files in the build folder to the repository you cloned in [Step 2](#step-2-use-git-to-clone-the-space).
<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/9.png">
</figure> 

## Step 10: Enable Git-LFS for Large File Storage

Navigate to your repository. Use the following commands to track large build files.

```
git lfs install
git lfs track Build/* 
```

## Step 11: Push your Changes

Finally, use the following Git commands to push your changes:

```
git add .
git commit -m "Add Unity WebGL build files"
git push
```

## Done!

Congratulations! Refresh your Space. You should now be able to play your game in a Hugging Face Space.

We hope you found this tutorial helpful. If you have any questions or would like to get more involved in using Hugging Face for Games, join the [Hugging Face Discord](https://hf.co/join/discord)!