---
title: "Object Detection Leaderboard"
thumbnail: /blog/assets/object-detection-leaderboard/thumbnail.png
authors:
- user: rafaelpadilla
- user: amyeroberts
---

# Object Detection Leaderboard

<!-- {blog_metadata} -->
<!-- {authors} -->

In the field of Computer Vision, Object Detection refers to the technique of identifying and localizing individual objects within an image or a video frame. Unlike image classification, where the task is to determine the predominant object or scene in the image, object detection not only categorizes the object classes present but also provides spatial information drawing bounding boxes around each detected object. An object detector can also output a “score”, also referred to as “confidence”, for each box, representing the probability that the detected object truly belongs to the predicted class.

The following image, for instance, shows 5 detections, being 1 “ball” with confidence of 98% and 4 “person” with confidences 98%, 95%, 97% and 97%.

<!-- ![image](assets/object-detection-leaderboard/intro_object_detection.png)
Figure 1: Example of outputs performed by an object detector. -->

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/intro_object_detection.png">
<center> Figure 1: Example of outputs performed by an object detector. </center>

Object detection models are versatile and have a wide range of applications across various domains. Some use cases where they are applied are **autonomous vehicles**, **face detection**, **surveillance and security**, **medical imaging**, **augmented reality**, **sport analysis**, **smart cities**, **gesture recognition**, etc.

[Hugging Face’s hub](https://huggingface.co/models?pipeline_tag=object-detection) has hundreds of object detection models (*a total of 671 models by August 29th, 2023*) pre trained in different datasets, able to identify and localize various object classes. 

Some object detection models can receive additional text queries to search for target objects described in the text. This way, these detectors (called zero-shot) are not limited to detecting objects seen during training.  

However, the diversity of detectors go beyond the range of output classes they can recognize. They vary in terms of underlying architectures, model sizes, processing speeds and prediction accuracy.

A popular metric used to evaluate the accuracy of predictions made by an object detection model is a metric known as **Average Precision (AP)** and its variants, which will be explained further.

The process to evaluate an object detection model encompassing several components like dataset with ground-truth annotations, detections (output prediction)  and metrics. This process is depicted in the schematic provided in Figure 2:

<!-- ![image](assets/object-detection-leaderboard/pipeline_object_detection.png)
Figure 2: Schematic illustrating the evaluation process for a traditional object detection model -->

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/pipeline_object_detection.png">
<center> Figure 2: Schematic illustrating the evaluation process for a traditional object detection model. </center>


It is not depicted in Figure 2, but as previously mentioned, certain models require text prompt inputs, to provide guidance on the specific classes the model is intended to detect.

First, a benchmarking dataset containing images with ground-truth bounding box annotations is chosen and fed into the object detection model. For each image, the model predicts bounding boxes, assigning associated class labels and confidence scores to each box. During the evaluation phase, these predicted bounding boxes are compared with the ground-truth boxes present in the dataset. The evaluation yields a set of metrics, each ranging between [0, 1], reflecting a specific evaluation criteria. We will cover each of them in a separate section.

In the following sections, we will delve into the definition of Average Precision, its variations and the computation methodologies associated with them. Next, we will explore the influence of pre-processing and post-processing parameters on a model’s outcomes. Following this, we will present metrics pertaining to object detectors available in the Hugging Face hub and published in our [Object Detection Leaderboard](https://huggingface.co/spaces/rafaelpadilla/object_detection_leaderboard). Concluding the discussion, we will shed light on the reasons why certain models might exhibit divergent results across different repositories. 



## What is Average Precision and how to compute it?


Average Precision is a single-number metric that summarizes the precision-recall curve. It captures the ability of a model to classify and localize objects correctly, while taking into account both false positive and false negative detections.

Every box predicted by the model is considered a “positive” detection. Based on a criteria known as Intersection over Union (IoU) between the predicted box and a ground-truth annotation, a detection is categorized either as a true positive (TP) or a false positive (FP). 

The IoU measures the overlap between the predicted bounding box and the actual (ground truth) bounding box. It's computed by dividing the area where the two boxes overlap by the area covered by both boxes combined. Figure 3 visually demonstrates the IoU using an example of a predicted box and its corresponding ground-truth box.


<!-- ![image](assets/object-detection-leaderboard/iou.png)
Figure 3: Intersection over Union (IoU) between a detection (in green) and ground-truth (in blue). -->

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/iou.png">
<center> Figure 3: Intersection over Union (IoU) between a detection (in green) and ground-truth (in blue). </center>

Clearly, if both the ground-truth and detected boxes share identical coordinates, representing the same region in the image, their IoU value is 1. Conversely, if the boxes do not overlap at any pixel, the IoU is considered to be 0.

In scenarios where high precision in detections is expected (e.g. an autonomous vehicle), the predicted bounding boxes should closely align with the ground-truth boxes. For that, a IoU threshold (\\( \text{T}_{\text{IOU}} \\)) approaching 1 is preferred. On the other hand, for applications where the exact position of the detected bounding boxes relative to the target object isn’t critical, the threshold can be relaxed, setting \\( \text{T}_{\text{IOU}} \\) closer to 0.

Based on predefined \\( \text{T}_{\text{IOU}} \\), we can define True Positives and True Negatives:
* **True Positive (TP)**: A correct detection where IoU ≥ \\( \text{T}_{\text{IOU}} \\).
* **False Positive (FP)**: An incorrect detection (missed object), where the IoU < \\( \text{T}_{\text{IOU}} \\).

Conversely, Negatives are evaluated  based on a ground-truth bounding and can be defined as False Negative (FN) or True Negative (TN):
* **False Negative (FN)**: Refers to a ground-truth object that the model failed to detect.
* **True Negative (TN)**: Denotes a correct non-detection. Within the domain of object detection, countless bounding boxes within an image should NOT be identified, as they don't represent the target object. Consider all possible boxes in an image that don’t represent the target object - quite a vast number, isn’t it? :) That's why we do not consider TN to compute object detection metrics.

Now that we can identify our TPs, FPs and FNs, we can define Precision and Recall:

* **Precision** is the ability of a model to identify only the relevant objects. It is the percentage of correct positive predictions and is given by:

<p style="text-align: center;">
\\( \text{Precision} = \frac{TP}{(TP + FP)} = \frac{TP}{\text{all detections}} \\)
</p>

which translates to the ratio of true positives over all detected boxes.

* **Recall** gauges a model’s competence in finding all the relevant cases (all ground truth bounding boxes). It indicates the proportion of TP detected among all ground truths and is given by:


<p style="text-align: center;">
\\( \text{Recall} = \frac{TP}{(TP + FN)} = \frac{TP}{\text{all ground truths}} \\)
</p>


Note that TP, FP and FN depend on a predefined IoU threshold, and so do Precision and Recall.

Now, we'll illustrate the relationship between Precision and Recall by plotting their respective curves for a specific target class, say "dog".  We’ll adopt a moderate IoU threshold = 75% to delineate our TP, FP and FN. Subsequently we can compute the Precision and Recall values. For that, we need to vary the confidence scores of our detections. 

Figure 4 shows an example of the Precision and Recall curve. For a deeper exploration into the computation of this curve, the papers “[A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit](https://www.mdpi.com/2079-9292/10/3/2790)” (Padilla, et al) and “[A Survey on Performance Metrics for Object-Detection Algorithms](https://ieeexplore.ieee.org/document/9145130)” (Padilla, et al) offer more detailed toy examples demonstrating how to compute this curve.

<!-- ![image](assets/object-detection-leaderboard/pxr_te_iou075.png)
Figure 4: Precision x Recall curve for a target object “dog” considering TP detections using IoU_thresh = 0.75  -->

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/pxr_te_iou075.png"> <center> Figure 4: Precision x Recall curve for a target object “dog” considering TP detections using IoU_thresh = 0.75. </center>


The precision-recall curve illustrates the balance between precision and recall based on different confidence levels of a detector's bounding boxes. Each point of the plot is computed using a different confidence value. 

Let’s borrow the practical example presented in the paper [A Survey on performance metrics for object-detection algorithms](https://ieeexplore.ieee.org/document/9145130) to illustrate how to compute the Average Precision plot. Consider a dataset made of 7 images with 15 ground-truth objects of the same class, as shown in Figure 5. For simplification purposes let’s consider that all boxes belong to the same class “dog”.

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/dataset_example.png"> <center> Figure 5: : Example of 24 detections (red boxes) performed by an object detector trained to detect 15 ground-truth objects (green boxes) belonging to the same class. </center>

Our hypothetical object detector retrieved 24 objects in our dataset, illustrated by the red boxes. To evaluate how well the detector performed for this specific class in our benchmarking dataset, we need to compute the precision and recall using equations XX TODO and XX TODO for all confidence levels. For that, we need to establish some rules:
* **Rule 1**: For a matter of simplicity let’s consider our detections a True Positive (TP) if the IoU >= 30%, otherwise, it is a False Positive (FP). 
* **Rule 2**: For cases where a detection overlaps more than one ground-truth (as in Images 2 to 7), the predicted box with the highest IoU is considered TP, and the other is FP.

Based on these rules, we can classify each detection as TP or FP, as shown in Table 1:

<center> Table 1: Detections from Figure 5 classified as TP or FP considering (\\( \text{T}_{\text{IOU}} = 30% \\)) </center>
<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/table_1.png"> 

Note that by rule 2, in image 1, “E” is TP while “D” is FP because IoU between “E” and the ground-truth is greater than IoU between “D” and the ground-truth.

Now, we need to compute Precision and Recall for all confidence levels. A good way to do it, is to sort the detections by their confidences and, for each confidence level,   count how many TP would be left in the dataset. Then we compute the precision and recall values for that particular confidence level, as shown in Table 2. The computation of each value of Table 2 can be viewed in [this Spread Sheet](https://docs.google.com/spreadsheets/d/1mc-KPDsNHW61ehRpI5BXoyAHmP-NxA52WxoMjBqk7pw/edit?usp=sharing).

<center> Table 2: Computation of Precision and Recall values of detections from Table 1 </center>
<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/table_1.png"> 


From top down, the accumulative TP (acc TP) column of Table 2 is increased in 1 every time a TP is noted, and the accumulative FP (acc FP) column is increased in 1 always when a FP is noted. Columns "acc TP" and "acc FP" basically tell us what are the TP and FP values given a particular confidence level. 

For example, consider the 12th row (detection “P”) of Table 2. The value "acc TP = 4" means that if we benchmark our model in this particular dataset with a confidence of 0.62, we would correctly detect 4 target objects and incorrectly detect 8 target objects. This would result in Precision = 0.3333 (\\( \frac{\text{acc TP}}{(\text{acc TP} + \text{acc FP})} = \frac{4}{(4+8)} \\) ) and Recall = 0.2667 (\\( \frac{\text{acc TP}}{\text{all ground truths}} = \frac{4}{15} \\) ).

Now, we can plot the Precision x Recall curve with the values, as shown in Figure 6:

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/precision_recall_example.png"> <center> Figure 6: Precision x Recall curve for the detections computed in Table 2. </center>

By examining the curve, one may infer the potential trade-offs between precision and recall and expect to have a model’s optimal operating point based on a selected confidence threshold, even if this threshold is not explicitly depicted on the curve.

If a detector's confidence results in few false positives (FP), it is likely to have high precision. However, this might lead to missing many true positives (TP), causing a high false negative (FN) rate and subsequently, low recall. On the other hand, accepting more positive detections can boost recall but might also raise the FP count, thereby reducing precision.

**The area under the Precision and Recall curve (AUC) computed for a target class represents the Average Precision value for that particular class.** COCO evaluation approach refers to “AP” as the mean AUC value among all target classes in the image dataset, which is also referred to as Mean Average Precision (mAP) by other approaches.

For a very large dataset, the detector is likely to output boxes with a wide range of confidence levels, resulting in a jagged Precision x Recall line, making it challenging to precisely compute its AUC (Average Precision). Different methods approximate the area of the curve with different approaches A popular approach is the called N-interpolation approach, where N represents how many points are sampled from the Precision x Recall blue line.

COCO’s approach, for instance, uses 101-interpolation, which computes 101 points for equally spaced  recall values (0.  , 0.01, 0.02, … 1.00), while other approaches use 11 points, referred to as 11-interpolation.

Figure 7 illustrates a Precision Recall curve (in blue) with 11 recall points equally spaced.

<img class="mx-auto" style="float: left;" padding="5px" width="*" src="/blog/assets/object-detection-leaderboard/11-pointInterpolation.png"> <center> Figure 7: Example of a Precision x Recall curve using the  11-interpolation approach. The 11 red dots are computed with Equation XXXX TODO. </center>

The red points are placed according to the Equation XXXX TODO:


\\( \rho_{\text{interp}} (R) = \max_{\tilde{r}:\tilde{r} \geq r} \rho \left( \tilde{r} \right) \\)

where \\( \rho \left( \tilde{r} \right) \\) is the measured precision at recall \\( \tilde{r} \\).

In this definition, instead of using the precision value \\( \rho (R)} \\) observed in each recall level \\( R \\), the precision \\( \rho_{\text{interp}} (R) \\) is obtained by considering the maximum precision whose recall value is greater than \\( R \\).

For this type of approach, the AUC, which represents the Average Precision, is approximated by the average of all points, and given by:


\\( \text{AP}_{11} = \frac{1}{11} = \sum\limits_{R\in \left \{ 0, 0.1, ...,1 \right \}} \rho_{\text{interp}} (R) \\)


## What is Average Recall and how to compute it?

Average Recall (AR) is a metric that's often used alongside AP to evaluate object detection models. While AP evaluates both precision and recall across different confidence thresholds to provide a single-number summary of model performance, AR focuses solely on the recall aspect, not taking the confidences into account, considering all detections into positives.

COCO’s approach computes AR as the mean of the maximum obtained recall over IOUs > 0.5 and classes. 

When using IOUs in the range of [0.5, 1] for AR, by averaging recall values within this interval, the model is assessed based on the premise that the object's location is significantly accurate. Hence, if your goal is to evaluate your model for both high recall and precise object localization, AR could be a valuable evaluation metric to consider.


## What are the variants of Average Precision and Average Recall?


Based on predefined IoU thresholds and the areas associated with ground-truth objects, different versions of AP and AR can be obtained:

* **AP@.5**: It sets IoU threshold = 0.5 and computes the Precision-Recall AUC for each target class in the image dataset. Then, the computed results for each class are summed up and divided by the number of classes.
* **AP@.75**: It uses the same methodology as AP@.50, but it considers IoU threshold = 0.75. With this higher IoU requirement, AP@.75 is considered stricter than AP@.5 and should be considered to evaluate models that need to achieve a high level of localization accuracy in their detections.
* **AP@[.5:.05:.95]**: also referred to AP by cocoeval tools: This is an expanded version of AP@.5 and AP@.75, as it computes AP@ with different IoU thresholds (0.5, 0.55, 0.6,...,0.95) and averages the computed results as shown in Equation XXX TODO. In comparison to AP@.5 and AP@.75, this metric provides a holistic evaluation, capturing a model’s performance across a broader range of localization accuracies.


\\( \text{AP@[.5:.05:0.95} = \frac{\text{AP}_{0.5} + \text{AP}_{0.55} + ... + \text{AP}_{0.95}}{10} \\)

* **AP-S**: It applies AP@[.5:.05:.95] considering (small) ground-truth objects with area < 32^2 pixels.
* **AP-M**: It applies AP@[.5:.05:.95] considering (medium-sized) ground-truth objects with 32^2 < area < 96^2 pixels.
* **AP-L**: It applies AP@[.5:.05:.95] considering (large) ground-truth objects with 32^2 < area < 96^2 pixels.

For Average Recall (AR), 10 IoU thresholds (0.5, 0.55, 0.6,...,0.95) are used to compute the recall values. AR is computed by either limiting the number of detections per image or by limiting the detections based on the object's area.

* **AR-1**: considers up to 1 detection per image.
* **AR-10**: considers up to 10 detection per image.
* **AR-100**: considers up to 100 detection per image.
* **AR-S**: considers (small) objects with area < 32^2 pixels.
* **AR-M**: considers (medium-sized) objects with area 32^2 < area < 96^2 pixels.
* **AR-L**: considers (large) objects with area > 96^2 pixels.


## How to pick the best model based on the metrics?



Selecting an appropriate metric to evaluate and compare object detectors takes into account several factors. The primary considerations include the purpose of the application and the characteristics of the dataset used to train and evaluate the models.

For general performance, **AP (AP@[.5:.05:.95])** is a good choice if you want an all-rounded model across different IoU thresholds, without a hard requirement on the localization of the detected objects.

If you want a model with good object recognition and objects generally in the right place, you can look at the **AP@.5**. But if you prefer a more accurate model in placing the bounding boxes, **AP@.75** is more appropriate.

If you have restrictions on object sizes, **AP-S**, **AP-M** and **AP-L** come into play. For example, if your dataset or application predominantly features small objects, AP-S provides insights into the detector's efficacy in recognizing such small targets. This becomes crucial in scenarios such as detecting distant vehicles or small artifacts in medical imaging.


## Which parameters can impact the Average Precision results?

By picking an object detection models in 🤗 Hugging Face’s hub, we can vary the output boxes by trying different parameters in the model’s post processing function.

Let’s take the DEtection TRansformer (DETR) ([facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)) as an example. It uses a DetrImageProcessor object to process the bounding boxes and logits, as shown in the snippet below:

```python 

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor=DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# PIL images have their size in (w, h) format
target_sizes = torch.tensor([image.size[::-1]])
results=processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)

```






