---
title: "Fine tune Florence 2 for new (unseen) tasks" 
thumbnail: /blog/assets/182_finetune-florence/thumbnail.webp
authors:
- user: andito
- user: your_coauthor
  guest: true

---

# Fine tune Florence 2 for new tasks

Florence 2, released by Microsoft in June 2024, is a vision foundation model. It's super attractive because of it small size (200M and 700M), and because it supports a variety of computer vision and vision-language tasks.

Out of the box, Florence supports captioning, object detection, OCR, and more. But the task you need might not be supported, or you might need the model to perform particularly well in a given task. That's when you will need to fine tune the model.

In this blog, we focus on fine tuning Florence on DocVQA, since the authors report that Florence 2 can perform visual question answering, but their released model didn't include this capability. 

