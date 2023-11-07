---
title: "HF Hub 现已加入存储区域功能"
thumbnail: /blog/assets/172_regions/thumbnail.png
authors:
- user: coyotte508
- user: rtrm
- user: XciD
- user: michellehbn
- user: violette
- user: julien-c
translators:
- user: chenglu
---

# HF Hub 现已加入存储区域功能

我们在 [企业版 Hub 服务](https://huggingface.co/enterprise) 方案中推出了 **存储区域（Storage Regions）** 功能。

通过此功能，用户能够自主决定其组织的模型和数据集的存储地点，这带来两大显著优势，接下来的内容会进行简要介绍：
- **法规和数据合规**，此外还能增强数字主权
- **性能提升**（下载和上传速度更快，减少延迟）

目前，我们支持以下几个存储区域：
- 美国 🇺🇸
- 欧盟 🇪🇺
- 即将到来：亚太地区 🌏

在深入了解如何设置这项功能之前，先来看看如何在您的组织中配置它 🔥

## 组织设置

如果您的组织还未开通企业版 Hub 服务，您将会看到以下界面：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/storage-regions/no-feature.png)

订阅服务后，您将能够访问到区域设置页面：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/storage-regions/feature-annotated.png)

在这个页面上，您能：
- 审核当前组织仓库的存储位置
- 通过下拉菜单为新建仓库选择存储位置

## 仓库标签

储存在非默认位置的仓库（模型或数据集）将直接在标签中显示其所在的区域，使组织成员能够直观地了解仓库位置。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/storage-regions/tag-on-repo.png)

## 法规和数据合规

在许多规定严格的行业，按照法规要求在指定地域存储数据是必须的。

对于欧盟的公司，这意味着他们能利用企业版 Hub 服务构建符合 GDPR 标准的机器学习解决方案：确保数据集、模型和推理端点全部存储在欧盟的数据中心。

如果您已是企业版 Hub 服务客户，并有更多相关疑问，请随时联系我们！

## 性能优势

把模型或数据集存放在离您的团队和基础设施更近的地方，可以显著提高上传和下载的效率。

鉴于模型权重和数据集文件通常体积庞大，这一点尤其重要。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/storage-regions/upload-speed.png)

例如，如果您的团队位于欧洲，并选择将仓库存储在欧盟区域，与存储在美国相比，上传和下载速度可以提升大约 4 到 5 倍。