---
title: "如何在 🤗 Space 上托管 Unity 游戏"
thumbnail: /blog/assets/124_ml-for-games/unity-in-spaces-thumbnail.png
authors:
- user: dylanebert
translators:
- user: SuSung-boy
- user: zhongdongy
  proofreader: true
---

# 如何在 🤗 Space 上托管 Unity 游戏


你知道吗？Hugging Face Space 可以托管自己开发的 Unity 游戏！惊不惊喜，意不意外？来了解一下吧！

Hugging Face Space 是一个能够以简单的方式来构建、托管和分享项目或应用样例的平台。虽然通常更多地是应用在机器学习样例中，不过实际上 Space 还可以用来托管 Unity 游戏，并且支持点击即玩。这里有一些游戏的 Space 示例:

- [Huggy](https://huggingface.co/spaces/ThomasSimonini/Huggy)。Huggy 是一个基于强化学习构建的简易游戏，玩家可以点击鼠标扔出小木棍，来教宠物狗把木棍捡回来
- [农场游戏](https://huggingface.co/spaces/dylanebert/FarmingGame)。农场游戏是我们在 [<五天创建一个农场游戏>](https://huggingface.co/blog/zh/ml-for-games-1) 系列中完成的游戏，玩家可以通过种植、收获和升级农作物来打造一个自己的繁荣农场
- [Unity API Demo](https://huggingface.co/spaces/dylanebert/UnityDemo)。一个 Unity 样例

本文将详细介绍如何在 🤗 Space 上托管自己的 Unity 游戏。

## 第 1 步: 使用静态 HTML 模板创建 Space 应用

首先，导航至 [Hugging Face Spaces](https://huggingface.co/new-space) 页面，创建一个新的 Space 应用。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/1.png">
</figure>

选择 “静态 HTML” 模板，并为该 Space 取个名字，然后点击创建 Space。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/2.png">
</figure>

## 第 2 步: 使用 Git 克隆 Space 库到本地

使用 Git 将上一步创建的 Space 库克隆到本地。克隆命令如下:

```
git clone https://huggingface.co/spaces/{your-username}/{your-space-name}
```

## 第 3 步: 打开 Unity 项目

打开你希望在 🤗 Space 上托管的 Unity 项目

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/3.png">
</figure>

## 第 4 步: 将构建目标切换为 WebGL

点击菜单栏的 `File > Build Settings`，将构建目标切换为 WebGL。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/4.png">
</figure>

## 第 5 步: 打开 Player Settings 面板

在上一步打开的 Build Settings 窗口中，点击左下角的 “Player Settings” 按钮，打开 Player Settings 面板。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/5.png">
</figure>

## 第 6 步:(可选) 下载 Hugging Face Unity WebGL 模板

Hugging Face Unity WebGL 模板可以使得你制作的游戏在 🤗 Space 上展示地更加美观。可以点击 [此处](https://github.com/huggingface/Unity-WebGL-template-for-Hugging-Face-Spaces) 下载模板库，并将其放到你的游戏项目目录，然后在 Player Settings 面板中将 WebGL 模板切换为 Hugging Face 即可。

如下图所示，在 Player Settings 面板中点击 “Resolution and Presentation”，然后选择 Hugging Face WebGL 模板。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/6.png">
</figure>

## 第 7 步: 禁用压缩

在 Player Settings 面板中点击 “Publishing Settings”，将 Compression Format 改为 “Disabled” 来禁用压缩。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/7.png">
</figure>

## 第 8 步: 构建游戏项目

返回 Build Settings 窗口，并点击 “Build” 按钮，选择一个本地目录来保存构建的游戏项目文件。按照前几步的设置，Unity 将会把项目构建为 WebGL。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/8.png">
</figure>

## 第 9 步: 将构建完成的文件复制到 Space 库

构建过程完成之后，打开上一步中项目保存的本地目录，将该目录下的文件复制到 [第 2 步](#第-2-步-使用-git-克隆-space-库到本地) 中克隆的 Space 库里。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/124_ml-for-games/games-in-spaces/9.png">
</figure>

## 第 10 步: 为大文件存储启用 Git-LFS

打开 Space 库， 在该目录执行以下命令来追踪构建的大型文件。

```
git lfs install
git track Build/*
```

## 第 11 步: Push 到 Hugging Face Space

最后，将本地的 Space 库的所有改动推送到 Hugging Face Space 上。执行以下 Git 命令即可完成推送:

```
git add .
git commit -m "Add Unity WebGL build files"
git push
```

## 完成！

至此，在 🤗 Space 上托管 Unity 游戏的所有步骤就都完成了。恭喜！现在请刷新你的 Space 页面，你就可以在 Space 上玩游戏了！

希望本教程对你有所帮助。如果你有任何疑问，或想更多地参与到 Hugging Face 游戏相关的应用中，可以加入 Hugging Face 的官方 [Discord](https://hf.co/join/discord) 频道来与我们取得联系！