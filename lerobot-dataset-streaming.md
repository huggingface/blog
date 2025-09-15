---
title: "`LeRobotDatasetStreaming`: large-scale robot training with no memory"
thumbnail: 
authors:
- user: fracapuano
- user: lhoestq
- user: aractingi
- user: aliberts
- user: rcadene
---

**TL;DR**
Training foundational models on large amounts of data takes a toll on the infrastructure required to store increasingly larger datasets. 
LeRobot hosts TBs of robotics data contributed by the community to train the next generation of models. 
Traditionally, training on large quantities of data requires custom infrastructure to organize storing PBs of data on disk, and moving batches to memory at training time.
Today, we introduce streaming datasets, further democratizing robotics enabling users to stream batches while training without storing prohibitively large datasets locally, leveraging the features of the Hugging Face Hub. 


## Getting started
Get started using streaming datasets for training by following [our tutorial](https://huggingface.co/docs/lerobot/en/streaming_datasets).
