---
title: "Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference With Unstructured Sparsity" 
thumbnail: /blog/assets/flash-llm/thumbnail.webp
authors:
- user: adamg012
---


# Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference With Unstructured Sparsity

Authors:

_Haojun Xia* (FSA lab, University of Sydney), Zhen Zheng* (Alibaba), Yuchao Li (Alibaba), Donglin Zhuang (FSA lab, University of Sydney), Zhongzhu Zhou (FSA lab, University of Sydney), Xiafei Qiu (Alibaba), Yong Li (Alibaba), Wei Lin (Alibaba), Shuaiwen Leon Song (FSA lab, University of Sydney)_

Paper Link: https://arxiv.org/abs/2309.10285 (paper is accepted to appear in VLDB 2024)

Source Code: https://github.com/AlibabaResearch/flash-llm

## Abstract:

Today, we release Flash-LLM, a large language model (LLM) inference acceleration library for unstructured model pruning, which can effectively accelerate large generative model inference with random sparsity on modern Tensor Core architectures. We are mainly optimizing the execution of four types of skinny Matrix Multiplications (Skinny MatMuls), which dominate both the execution time and the peak GPU memory usage for the overall LLM inference.

With the observation that these skinny MatMuls are bounded by off-chip memory access rather than arithmetic computations, we propose a basic strategy called `Load-as-Sparse and Compute-as-Dense’ (LSCD) that targets drastic global memory access reduction for higher inference performance, even with slightly increased shared memory access from on-the-fly sparse-to-dense format transformation as a trade-off. To achieve this goal, we carefully crafted a solution with a series of optimizations on data locality, on-chip sparse-to-dense transformation, and overlapping between memory and computation instructions. For single MatMul execution, Flash-LLM outperforms state-of-the-art sparse MatMul kernels Sputnik/SparTA by up to 3.6X/1.6X and outperforms the dense MatMul kernel (i.e., NVIDIA cuBLAS) by up to 2.1X. For LLMs after effective model pruning and finetuning, Flash-LLM achieves up to 3.6X token generation throughput improvement over FasterTransformer [3] on OPT-30B/66B/175B models with significantly lower inference cost.

## Background:

Large language models (LLM) have demonstrated their effectiveness across a wide range of tasks. However, with the rapid growth of the parameter size, it has become increasingly challenging to efficiently deploy these models. On one hand, their weights could be too large to be placed on a single GPU. For example, GPT-3 model requires 350GB memory to store only the parameters with FP16 data type, whereas the NVIDIA A100 GPU only has a maximum of 80 GB memory. On the other hand, large language models usually cause very high inference latency. This high latency is even the case when using multiple high-end GPUs as token generation requires large amounts of computation and memory bandwidth.

Model weight pruning methods (sparsification) have been demonstrated to be effective to reduce memory usage and computation for model inference while retaining good accuracy through removing a portion of less salient connections in neural networks. There are studies indicating that larger models are more robust to pruning in terms of model accuracy. In our practice we achieve 80% sparsity on OPT-30B and GPT-NEOX-20B with only 1.44% and 0.72% accuracy decrease. Thus, weight pruning could be an effective approach to address the LLM deployment problem. There are two typical types of pruning principals given a dense network: structured pruning (resulting in structured sparsity) and unstructured pruning (resulting in unstructured sparsity). In practice, unstructured pruning typically retains better accuracy than more restrictive structured pruning as it has less constraints.

## Opportunities and Insights:

Fig.1 illustrates the typical decoder architecture of a single layer in modern attention-based generative models. The memory consumption of model weights mainly comes from four major components in the decoder layer: QKV Projection, Output Projection, MLP1, and MLP2. In total, there are four key MatMuls to optimize. According to our experiments, we observe that the LLM inference performance is heavily bounded by these four MatMuls. With unstructured weight pruning, the memory consumption of the four components can be effectively reduced. We aim to further increase the performance of the four sparse MatMuls (SpMM) after weight pruning. We observe that the four MatMuls are very skinny: the output matrices of these MatMuls are tall and thin (H is much bigger than B in Fig.1). We take advantage of this characteristic to optimize the SpMM-centric LLM inference on modern GPUs.


![Fig.1 The typical LLM decoder layer structure. B: batch-size, H: hidden-dim](https://miro.medium.com/v2/resize:fit:720/format:webp/1*w0HZEfDneuAG27nmyIPAEg.png "Fig.1 The typical LLM decoder layer structure. B: batch-size, H: hidden-dim")
Fig.1 The typical LLM decoder layer structure. B: batch-size, H: hidden-dim

In theory, our analysis found that the skinny MatMuls are bounded by memory accesses rather than computation. As for MatMul of shape M/N/K, the computational intensity (_CI_) is:

It is easy to demonstrate that the smaller the N is, the smaller the _CI_ will become. The small _CI_ tends to indicate memory access bottleneck. Empirically, we have also analyzed the detailed GPU hardware utilization of typical MatMuls in common LLM models. As shown in Fig.2, the utilization of global memory and L2 cache access is quite high, while the Tensor Core utilization is very low, indicating the bottleneck of global memory access.

![Fig.2 The GPU hardware utilization of typical MatMuls in common LLM models. MatMul shape is M/K/batch-size.](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Gb5rQQoUDboTEE0ZikqxcQ.png "Fig.2 The GPU hardware utilization of typical MatMuls in common LLM models. MatMul shape is M/K/batch-size.")
Fig.2 The GPU hardware utilization of typical MatMuls in common LLM models. MatMul shape is M/K/batch-size.

The key take-away is that the main bottleneck for LLM inference is the insufficient bandwidth of global memory, instead of the peak computational throughout of tensor cores. Thus, Flash-LLM can obtain significant kernel speedups once the memory bottleneck is effectively mitigated, even still performing dense computation without any skipping for multiply–accumulate operations.

According to the observation above, we propose our basic idea of Load-as-Sparse and Compute-as-Dense (LSCD): GPU kernel loads the weight matrices from global memory in sparse format with reduced size, reconstructs the corresponding dense format in high-speed on-chip buffers, and feeds the dense data to tensor cores for dense computations. After applying LSCD, the global memory access is significantly reduced, and the computational intensity is increased to
![https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ri_WzR5hjxsCnK_Kr4Nww.png](https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ri_WzR5hjxsCnK_Kr4Nww.png "https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ri_WzR5hjxsCnK_Kr4Nww.png")
, where \\(\beta\\) indicates the sparsity ratio of the weight matrix.
Design and Implementation

Flash-LLM differs from existing works by enabling tensor cores for efficiently processing unstructured sparsity, while most of the existing sparse kernels, e.g., Sputnik [1] and cuSPARSE, can only utilize SIMT cores. It is clearly a huge waste leaving tensor cores not utilized for existing sparse kernels, as tensor cores can provide an order of magnitude higher computational throughput than SIMT cores. As a result, cuSPARSE and Sputnik cannot even outperform its dense counterpart implemented with cuBLAS until the model sparsity is higher than 98% and 90%, respectively.

However, it is not trivial to enable tensor cores for unstructured SpMM computations, as tensor cores are originally designed for highly structured dense MatMul computations. Firstly, data locality must be fully exploited at each GPU memory hierarchy via the adoption of sophisticated tiling mechanisms when designing tensor-core-centric kernels. Otherwise, the GPU memory hierarchy would not be capable of providing operands for tensor cores in a timely manner as tensor cores consume operands at a very fast speed. Secondly, it is unclear how to provide tensor cores with dense input while the weight matrices are stored in global memory with a sparse format. Finally, it is important but challenging to effectively overlap the execution of memory operations, SIMT instructions and tensor core instructions during runtime.

To address the first challenge, we adopted a tiling-based approach (i.e., block tiling for dense MatMul implementation) for the SpMM computations in Flash-LLM. Fig.3(a) shows the tiling method of Flash-LLM, where matrix A is a weight matrix stored in a sparse format in global memory. Each thread block (TB) is responsible for calculating a tile (e.g., the green tile in the shape of (\\(M \cdot N\\)) in the output matrix \\(C\\). For each iteration, each thread block loads \\(A\\)-Tile with shape \\([M, K]\\) in sparse format and \\(B\\)-Tile (shape \\([K,N]\\)) in dense format from global memory. \\(A\\)-Tile is then transformed to dense format with our efficient Sparse-to-Dense Transformation strategy (i.e., designed to solve the second challenge) and stored in shared memory while \\(B\\)-Tile is directly stored in shared memory. Finally, each thread block consumes the dense data in shared memory and generates the output tile through tensor core computations.

To solve the second challenge, we propose a new technique called Sparse-to-Dense Transformation, where GPU shared memory is used as the workspace to transform the matrix which is in a sparse format loaded from global memory to the equivalent dense format. Specifically, non-zero elements within the sparse matrix are extracted to their corresponding locations in the dense format on shared memory while zeros are filled to other locations. We use the distributed registers as the intermediate buffer to store the sparse data before extracting them to shared memory. We do not use shared memory as this intermediate buffer to avoid the turn-around shared memory access of the sparse data, which is essential to mitigate the new bottleneck of shared memory bandwidth. However, there are special considerations when using registers as intermediate buffers for sparse data as registers are very different with shared memory and global memory. Firstly, registers are not addressable, which means we cannot access an array of registers using a variable offset. As a result, forcing an array defined in CUDA into registers requires that, all the indices used to access the array can be determined statically at compile-time. Otherwise, the array will be stored in global memory instead, resulting in very poor performance. We provide special tips for more effectively using distributed registers as temporary buffers in section 4.3.2 of our paper. Secondly, each register is only visible to only one CUDA thread while shared/global memory is visible to all CUDA threads within the thread block. Thus, each thread should be able to do the sparse-to-dense transformation using only the small portion of the sparse data stored in its private registers. To satisfy this requirement, we propose the new sparse format called “Tiled-CSL” in section 4.3.1. Based on all the considerations above, the A weight matrix is first loaded to register files (RF), then extracted to shared memory by the Sparse-to-Dense Transformation, and then finally loaded to register files to be consumed by tensor cores as shown in Fig.3(b). Please refer to our paper for more technical details, where we also described the ahead-of-time sparse data reordering technique to reduce shared memory bank-conflict during dense format re-construction.

Given that each thread consumes a large fraction of the overall registers/shared-memory as buffers for tiling, the GPU thread-level parallelism (TLP) is inherently low. Thus, it is important to optimize the instruction-level parallelism. To solve the third challenge, we carefully designed a software pipeline for Flash-LLM. Fig.3(c) depicts the software pipeline of Flash-LLM, where the memory operations (regarding global memory access and shared memory access), SIMT core operations (mainly used for Sparse-to-Dense Transformation), and tensor core computations can be effectively overlapped. The decompression process of matrix A from sparse format to dense format is executed concurrently with the reading process of matrix B. Besides, Flash-LLM utilizes a double-buffer mechanism to effectively overlap the memory access and Tensor Core computations.

![Fig.3 The design of LSCD and the computation pipeline of Flash-LLM on the GPU](https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ri_WzR5hjxsCnK_Kr4Nww.png "Fig.3 The design of LSCD and the computation pipeline of Flash-LLM on the GPU")
Fig.3 The design of LSCD and the computation pipeline of Flash-LLM on the GPU

## Performance Evaluation

Flash-LLM presents superior performance in both single SpMM kernel execution and end-to-end LLM inference.

### SpMM kernel level comparison:

Fig.4 shows the performance of Flash-LLM and state-of-the-art solutions in performing common LLM MatMul calculations. Flash-LLM outperforms Sputnik[1]/SparTA[2] by 3.6x/1.4x, 3.0x/1.4x, and 2.0x/1.6x under 70%, 80%, and 90% sparsity, respectively. Besides, Flash-LLM can also outperform the state-of-the-art dense kernels cuBLAS with tensor core enabled by 1.4x, 1.7x, and 2.1x. CuSparse performs poorly in these SpMM calculations.

![Fig.4 The kernel performance of common LLM matrix multiplications. X axis: M/K/Sparsity. MatMul shapes are M/K/Batch Size.](https://miro.medium.com/v2/resize:fit:720/format:webp/1*pu-4ajUcQVrPe4bufaMmBA.png "Fig.4 The kernel performance of common LLM matrix multiplications. X axis: M/K/Sparsity. MatMul shapes are M/K/Batch Size.")
Fig.4 The kernel performance of common LLM matrix multiplications. X axis: M/K/Sparsity. MatMul shapes are M/K/Batch Size.

### End-to-end LLM inference comparison against the SOTA framework:

Fig.5, Fig.6 and Fig.7 show the performance of Flash-LLM and FasterTransformer [3] respectively on the OPT-30B, OPT-66B and OPT-175B models. The performance metric we use is #Token / GPU-Second, which can express the efficiency of token generation without considering the number of GPUs used. It should be noted that different optimization methods require different numbers of GPUs when executing different models. As shown in the figures, firstly, Flash-LLM can often support larger batch sizes because it requires less storage resources. Secondly, Flash-LLM has significantly higher token generation efficiency than FasterTransformer. Finally, Flash-LLM often requires fewer GPUs to execute the same LLM model, so the deployment cost of Flash-LLM optimized models is lower.

![Fig.5 The performance of Flash-LLM and FasterTransformer (FT) on OPT-30B model.](https://miro.medium.com/v2/resize:fit:720/format:webp/1*RhtW1GZnPcC55aHV_4Az4A.png "Fig.5 The performance of Flash-LLM and FasterTransformer (FT) on OPT-30B model.")
Fig.5 The performance of Flash-LLM and FasterTransformer (FT) on OPT-30B model.

![Fig.6 The performance of Flash-LLM and FasterTransformer (FT) on OPT-66B model.](https://miro.medium.com/v2/resize:fit:720/format:webp/1*EPRH3UdpI9AzZy0lxV3X9A.png "Fig.6 The performance of Flash-LLM and FasterTransformer (FT) on OPT-66B model.")
Fig.6 The performance of Flash-LLM and FasterTransformer (FT) on OPT-66B model.

![Fig.7 The performance and breakdown of Flash-LLM and FasterTransformer (FT) on OPT-175B model.](https://miro.medium.com/v2/resize:fit:720/format:webp/1*7ezKNKVruj7slqDQhpAKwQ.png "Fig.7 The performance and breakdown of Flash-LLM and FasterTransformer (FT) on OPT-175B model.")
Fig.7 The performance and breakdown of Flash-LLM and FasterTransformer (FT) on OPT-175B model.

Fig.7(b) presents the performance breakdown of Flash-LLM and FasterTransformer to further illustrate the performance advantage of Flash-LLM. On one hand, Flash-LLM’s matrix calculation is more efficient; on the other hand, because Flash-LLM requires fewer GPUs, its communication cost is also lower.

## Conclusion

The development of LLM systems is rapid. In just over a year, a large number of scientific research and engineering works have proposed many creative optimization solutions in terms of computing, storage, and scheduling. Quantization-based compression methods have been widely used to optimize the deployment of large language models. We explored the new LLM deployment optimization method based on unstructured sparsity in Flash-LLM and demonstrated its superior effect. We hope this work can bring some inspiration to more practitioners, and we also hope that this work will eventually make some contributions to promoting the efficient deployment of large models.

## References

[1] Trevor Gale, Matei Zaharia, Cliff Young, and Erich Elsen. Sparse GPU Kernels for Deep Learning. SC 2020.

[2] Ningxin Zheng, Bin Lin, Quanlu Zhang, Lingxiao Ma, Yuqing Yang, Fan Yang, Yang Wang, Mao Yang, and Lidong Zhou. SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute. OSDI 2022.

[3] FasterTransformer. https://github.com/NVIDIA/FasterTransformer
