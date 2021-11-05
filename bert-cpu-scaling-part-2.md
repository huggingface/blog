<style>
  .centered {
      display: block;
      margin: 0 auto;
  }

  figure {
      text-align: center;
      display: table;
      max-width: 100%; /* demo; set some amount (px or %) if you can */
      margin: 0px auto; /* not needed unless you want centered */
  }
</style>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<div class="blog-metadata">
    <small>Published April 19, 2021.</small>
</div>

<div class="author-card">
    <a href="/mfuntowicz">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1583858935715-5e67c47c100906368940747e.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>mfuntowicz</code>
            <span class="fullname">Morgan Funtowicz</span>
        </div>
    </a>
    <a href="/michaelbenayoun">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1615890856777-6047a3315da6ba4b1dfb9e18.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>michaelbenayoun</code>
            <span class="fullname">Michael Benayoun</span>
        </div>
    </a>
    <a href="/jeffboudier">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1605114051380-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>jeffboudier</code>
            <span class="fullname">Jeff Boudier</span>
        </div>
    </a>
    <a href="/echarlaix">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1615915889033-6050eb5aeb94f56898c08e57.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>echarlaix</code>
            <span class="fullname">Ella Charlaix</span>
        </div>
    </a>
</div>

# Scaling up BERT-like model Inference on modern CPU  - Part 2

## Introduction: Using Intel Software to Optimize AI Efficiency on CPU

As we detailed in our [previous blog post](https://huggingface.co/blog/bert-cpu-scaling-part-1), Intel Xeon CPUs provide a set of features especially designed for AI workloads such as AVX512 or VNNI (Vector Neural Network Instructions) 
for efficient inference using integer quantized neural network for inference along with additional system tools to ensure the work is being done in the most efficient way. 
In this blog post, we will focus on software optimizations and give you a sense of the performances of the new Ice Lake generation of Xeon CPUs from Intel. Our goal is to give you a full picture of what’s available on the software side to make the most out of your Intel hardware. 
As in the previous blog post, we show the performance with benchmark results and charts, along with new tools to make all these knobs and features easy to use.

Back in April, Intel launched its [latest generation of Intel Xeon processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), codename Ice Lake, targeting more efficient and performant AI workloads. 
More precisely, Ice Lake Xeon CPUs can achieve up to 75% faster inference on a variety of NLP tasks when comparing against the previous generation of Cascade Lake Xeon processors. 
This is achieved by a combination of both hardware and software improvements, [such as new instructions](https://en.wikichip.org/wiki/x86/avx512_vnni) and PCIe 4.0 featured on the new Sunny Cove architecture to supports Machine Learning and Deep Learning workloads. 
Last but not least, Intel worked on dedicated optimizations for various frameworks which now come with Intel’s flavors like 
[Intel’s Extension for Scikit Learn](https://intel.github.io/scikit-learn-intelex/), 
[Intel TensorFlow](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html) and 
[Intel PyTorch Extension](https://www.intel.com/content/www/us/en/developer/articles/containers/pytorch-extension.html).

All these features are very low-level in the stack of what Data Scientists and Machine Learning Engineers use in their day-to-day toolset. 
In a vast majority of situations, it is more common to rely on higher level frameworks and libraries to handle multi-dimensional arrays manipulation such as 
[PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org/) and make use of highly tuned mathematical operators such as [BLAS (Basic Linear Algebra Subroutines)](http://www.netlib.org/blas/) for the computational part.

In this area, Intel plays an essential role by providing software components under the oneAPI umbrella which makes it very easy to use highly efficient linear algebra routines through 
Intel [oneMKL (Math Kernel Library)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-math-kernel-library-onemkl.html), 
higher-level parallelization framework with Intel OpenMP or the [Threading Building Blocks (oneTBB)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html).
Also, oneAPI provides some domain-specific libraries such as Intel [oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) for deep neural network primitives (ReLU, fully-connected, etc.) or 
[oneCCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html) for collective communication especially useful when using distributed setups to access efficient all-reduce operations over multiple hosts.

Some of these libraries, especially MKL or oneDNN, are natively included in frameworks such as PyTorch and TensorFlow ([since 2.5.0](https://medium.com/intel-analytics-software/leverage-intel-deep-learning-optimizations-in-tensorflow-129faa80ee07)) to bring all the performance improvements to the end user out of the box. 
When one would like to target very specific hardware features, Intel provides custom versions of the most common software, especially optimized for the Intel platform. 
This is for instance the case with TensorFlow, [for which Intel provides custom, highly tuned and optimized versions of the framework](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html),
or with the Intel PyTorch Extension (IPEX) framework which can be considered as a feature laboratory before upstreaming to PyTorch.

## Deep Dive: Leveraging advanced Intel features to improve AI performances

### Performance tuning knobs

As highlighted above, we are going to cover a new set of tunable items to improve the performance of our AI application. From a high-level point of view, every machine learning and deep learning framework is made of the same ingredients:
1. A structural way of representing data in memory (vector, matrices, etc.)
2. Implementation of mathematical operators
3. Efficient parallelization of the computations on the target hardware

_In addition to the points listed above, deep learning frameworks provide ways to represent data flow and dependencies to compute gradients. 
This falls out of the scope of this blog post, and it leverages the same components as the ones listed above!_

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel libraries overview under the oneAPI umbrella" src="assets/35_bert_cpu_scaling_part_2/oneapi.jpg" />
  <figcaption>Figure 1. Intel libraries overview under the oneAPI umbrella</figcaption>
</figure>
<br>

### 1. Memory allocation and management libraries

This blog post will deliberately skip the first point about the data representation as it is something rather framework specific. 
For reference, PyTorch uses its very own implementation, called [ATen](https://github.com/pytorch/pytorch/tree/master/aten/src), 
while TensorFlow relies on the open source library [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for this purpose.

While it’s very complex to apply generic optimizations to different object structures and layouts, there is one area where we can have an impact: Memory Allocation. 
As a short reminder, memory allocation here refers to the process of programmatically asking the operating system a dynamic (unknown beforehand) area on the system where we will be able to store items into, such as the malloc and derived in C or the new operator in C++. 
Memory efficiency, both in terms of speed but also in terms of fragmentation, is a vast scientific and engineering subject with multiple solutions depending on the task and underlying hardware. 
Over the past years we saw more and more work in this area, with notably:
- [jemalloc](http://jemalloc.net/) (Facebook - 2005)
- [mimalloc](https://microsoft.github.io/mimalloc/) (Microsoft - 2019)
- [tcmalloc](https://abseil.io/blog/20200212-tcmalloc) (Google - 2020) 

Each pushes forward different approaches to improve aspects of the memory allocation and management on various software.

### 2. Efficient parallelization of computations

Now that we have an efficient way to represent our data, we need a way to take the most out of the computational hardware at our disposal. 
Interestingly, when it comes to inference, CPUs have a potential advantage over GPUs in the sense they are everywhere, and they do not require specific application components and administration staff to operate them.

Modern CPUs come with many cores and complex mechanisms to increase the general performances of software. 
Yet, as we highlighted on [the first blog post](https://hf.co/blog/bert-cpu-scaling-part-1), they also have features which can be tweaked depending on the kind of workload (CPU or I/O bound) you target, to further improve performances for your application. 

Still, implementing parallel algorithms might not be as simple as throwing more cores to do the work. 
Many factors, such as data structures used, concurrent data access, CPU caches invalidation - all of which might prevent your algorithm from being effectively faster. 
As a reference talk, we recommend the talk from [**Scott Meyers: CPU Caches and Why You Care**](https://www.youtube.com/watch?v=WDIkqP4JbkE) if you are interested in diving more into the subject.

Thankfully, there are libraries which make the development process of such parallel algorithms easier and less error-prone. 
Among the most common parallel libraries we can mention OpenMP and TBB (Threading Building Blocks), which work at various levels, from programming API in C/C++ to environment variable tuning and dynamic scheduling. 
On Intel hardware, it is advised to use the Intel implementation of the OpenMP specification often referred as "IOMP" available as part of the [Intel oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).

<br>

<figure class="image">
    <medium-zoom background="rgba(0,0,0,.7)" alt="Code snippet showing parallel computation done through OpenMP" src="assets/35_bert_cpu_scaling_part_2/openmp.png"></medium-zoom>
    <figcaption>Figure 2. Code snippet showing parallel computation done through OpenMP</figcaption>
</figure>

[comment]: <> (<br>)

### 3. Optimized mathematical operators

Now that we covered the necessary building blocks for designing efficient data structures and parallel algorithms, the last remaining piece is the one running the computation, 
the one implementing the variety of mathematical operators and neural network layers to do what we love most, designing neural networks! 😊

In every programmer toolkit, there are multiple levels which can bring mathematical operations support, which can then be optimized differently depending on various factors such as the data storage layout 
being used (Contiguous memory, Chunked, Packed, etc.), the data format representing each scalar element (Float32, Integer, Long, Bfloat16, etc.) and of course the various instructions being supported by your processor.

Nowadays, almost all processors support basic mathematical operations on scalar items (one single item at time) or in vectorized mode (meaning they operate on multiple items within the same CPU instructions, referred as SIMD “Single Instruction Multiple Data”).
Famous sets of SIMD instructions are SSE2, AVX, AVX2 and the AVX-512 present on the latest generations of Intel CPUs being able to operate over 16 bytes of content within a single CPU clock.

Most of the time, one doesn't have to worry too much about the actual assembly being generated to execute a simple element-wise addition between two vectors, but if you do, 
again there are some libraries which allow you to go one level higher than writing code calling CPU specific intrinsic to implement efficient mathematical kernels. 
This is for instance what Intel’s MKL “Math Kernel Library” provides, along with the famous BLAS “Basic Linear Algebra Subroutines” interface to implement all the basic operations for linear algebra.

Finally, on top of this, one can find some domain specific libraries such as Intel's oneDNN which brings all the most common and essential building blocks required to implement neural network layers. 
Intel MKL and oneDNN are natively integrated within the PyTorch framework, where it can enable some performance speedup for certain operations such as Linear + ReLU or Convolution. 
On the TensorFlow side, oneDNN can be enabled by setting the environment variable `TF_ENABLE_ONEDNN_OPTS=1` (_TensorFlow >= 2.5.0_) to achieve similar machinery under the hood.

## More Efficient AI Processing on latest Intel Ice Lake CPUs

In order to report the performances of the Ice Lake product lineup we will closely follow [the methodology we used for the first blog](https://hf.co/blog/bert-cpu-scaling-part-1#2-benchmarking-methodology) post of this series. As a reminder, we will adopt the exact same schema to benchmark the various setups we will highlight through this second blog post. More precisely, the results presented in the following sections are based on:
- PyTorch: 1.9.0
- TensorFlow: 2.5.0
- Batch Sizes: 1, 4, 8, 16, 32, 128
- Sequence Lengths: 8, 16, 32, 64, 128, 384, 512 

We will present the results through metrics accepted by the field to establish the performances of the proposed optimizations: 
- Latency: Time it takes to execute a single inference request (i.e., “forward call”) through the model, expressed in millisecond.
- Throughput: Number of inference requests (i.e., “forward calls”) the system can sustain within a defined period, expressed in call/sec.

We will also provide an initial baseline showing out-of-the-box results and a second baseline applying all the different optimizations we highlighted in the first blogpost. 
Everything was run on an Intel provided cloud instance featuring the [Ice Lake Xeon Platinum 8380](https://ark.intel.com/content/www/fr/fr/ark/products/205684/intel-xeon-platinum-8380hl-processor-38-5m-cache-2-90-ghz.html) CPU operating on Ubuntu 20.04.2 LTS.

You can find the same processors on the various cloud providers: 
- [AWS m6i / c6i instances](https://aws.amazon.com/fr/blogs/aws/new-amazon-ec2-c6i-instances-powered-by-the-latest-generation-intel-xeon-scalable-processors/)
- [Azure Ev5 / Dv5 series](https://azure.microsoft.com/en-us/blog/upgrade-your-infrastructure-with-the-latest-dv5ev5-azure-vms-in-preview/)

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel Ice Lake Xeon 8380 Specifications" src="assets/35_bert_cpu_scaling_part_2/intel_xeon_8380_specs.svg" />
  <figcaption>Figure 3. Intel Ice Lake Xeon 8380 Specifications</figcaption>
</figure>
<br>


### Establishing the baseline

As mentioned previously, the baselines will be composed of two different setups: 
-	Out-of-the-box: We are running the workloads as-is, without any tuning
-	Optimized: We apply the various knobs present in [Blog #1](https://hf.co/blog/bert-cpu-scaling-part-1#2-benchmarking-methodology)

Also, from the comments we had about the previous blog post, we wanted to change the way we present the framework within the resulting benchmarks. 
As such, through the rest of this second blog post, we will split framework benchmarking results according to the following:
- Frameworks using “eager” mode for computations (PyTorch, TensorFlow)
- Frameworks using “graph” mode for computations (TorchScript, TensorFlow Graph, Intel Tensorflow)


#### Baseline: Eager frameworks latencies 

Frameworks operating in eager mode usually discover the actual graph while executing it. 
More precisely, the actual computation graph is not known beforehand and you gradually (_eagerly_) execute one operator
which will become the input of the next one, etc. until you reach leaf nodes (outputs).

These frameworks usually provide more flexibility in the algorithm you implement at the cost of increased runtime overhead
and slightly potential more memory usage to keep track of all the required elements for the backward pass.

Last but not least, it is usually harder through these frameworks to enable graph optimizations such as operator fusion.
For instance, many deep learning libraries such as oneDNN have optimized kernels for Convolution + ReLU but you actually need
to know before executing the graph that this pattern will occur within the sequence of operation, which is, by design, not
something possible within eager frameworks.

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="PyTorch latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/eager_mode_pytorch_baseline.svg" />
  <figcaption>Figure 4. PyTorch latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/eager_mode_tensorflow_baseline.svg" />
  <figcaption> Figure 5. Google's TensorFlow latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow with oneDNN enabled latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/eager_mode_tensorflow_onednn_baseline.svg" />
  <figcaption>Figure 6. Google's TensorFlow with oneDNN enabled latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel TensorFlow latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/eager_mode_intel_tensorflow_baseline.svg" />
  <figcaption>Figure 7. Intel TensorFlow latencies with respect to the number of cores involved</figcaption>
</figure>
<br>

The global trend highlights the positive impact of the number of cores on the observed latencies. 
In most of the cases, increasing the number of cores reduces the computation time across the different workload sizes. 
Still, putting more cores to the task doesn't result in monotonic latency reductions, there is always a trade-off between the workload’s size and the number of resources you allocate to execute the job.


As you can see on the charts above, one very common pattern tends to arise from using all the cores available on systems with more than one CPU (more than one socket). 
The inter-socket communication introduces a significant latency overhead and results in very little improvement to increased latency overall. 

Also, this inter-socket communication overhead tends to be less and less perceptive as the workload becomes larger, 
meaning the usage of all computational resources benefits from using all the available cores. 
In this domain, it seems PyTorch (Figure 1.) and Intel TensorFlow (Figure 4.) seem to have slightly better parallelism support, 
as showed on the sequence length 384 and 512 for which using all the cores still reduces the observed latency.



#### Baseline: Graph frameworks latencies 

This time we compare performance when using frameworks in “Graph” mode, where the graph is fully known beforehand,
and all the allocations and optimizations such as graph pruning and operators fusing can be made.

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="TorchScript latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/graph_mode_torchscript_baseline.svg" />
  <figcaption>Figure 8. TorchScript latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/graph_mode_tensorflow_baseline.svg" />
  <figcaption>Figure 9. Google's TensorFlow latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow with oneDNN enabled latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/graph_mode_tensorflow_onednn_baseline.svg" />
  <figcaption>Figure 10. Google's TensorFlow with oneDNN enabled latencies with respect to the number of cores involved</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel TensorFlow latencies with respect to the number of cores involved" src="assets/35_bert_cpu_scaling_part_2/baselines/graph_mode_intel_tensorflow_baseline.svg" />
  <figcaption>Figure 11. Intel TensorFlow latencies with respect to the number of cores involved</figcaption>
</figure>
<br>


This is often referred to as “tracing” the graph and, as you can see here, the results are not that different from TorchScript (Graph execution mode from PyTorch) vs TensorFlow(s). 
All TensorFlow implementations seem to perform better than TorchScript when the parallelization is limited (low number of cores involved in the intra operation computations) but this seems not to scale efficiently 
as we increase the computation resources, whereas TorchScript seems to be able to better leverage the power of modern CPUs. 

Still, the margin between all these frameworks in most cases very limited.


### Tuning the Memory Allocator: Can this impact the latencies observed?

One crucial component every program dynamically allocating memory relies on is the memory allocator. 
If you are familiar with C/C++ programming this component provides the low bits to malloc/free or new/delete. 
Most of the time you don’t have to worry too much about it and the default ones (glibc for instance on most Linux distributions) will provide great performances out of the box. 
Still, in some situations it might not provide the most efficient performances, as these default allocators are most of the time designed to be “good” most of the time, 
and not fine-tuned for specific workloads or parallelism. 

So, what are the alternatives, and when are they more suitable than the default ones? Well, again, it depends on the kind of context around your software. 

Possible situations are a heavy number of allocations/de-allocations causing fragmentation over time, 
specific hardware and/or architecture you’re executing your software on and finally the level of parallelism of your application.

Do you see where this is going? Deep learning and by extension all the applications doing heavy computations are heavily multi-threaded, 
that’s also the case for software libraries such as PyTorch, TensorFlow and any other frameworks targeting Machine Learning workloads. 

The default memory allocator strategies often rely on global memory pools which require the usage of synchronization primitives to operate, 
increasing the overall pressure on the system, reducing the performance of your application.
Some recent works by companies such as Google, Facebook and Microsoft provided alternative memory allocation strategies implemented in custom memory allocator libraries 
one can easily integrate directly within its software components or use dynamic shared library preload to swap the library being used to achieve the allocation/de-allocation. 

Among these libraries, we can cite a few of them such as [tcmalloc](), [jemalloc]() and [mimalloc]().

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Legend - Various allocator benchmarked on different tasks" src="assets/35_bert_cpu_scaling_part_2/allocator_benchmark_legend.png" />
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Various allocator benchmarked on different tasks" src="assets/35_bert_cpu_scaling_part_2/allocator_benchmark.png" />
  <figcaption>Figure 12. Various memory allocators benchmarked on different tasks</figcaption>
</figure>
<br>


Through this blog post we will only focus on benchmarking tcmalloc and jemalloc as potential memory allocators drop-in candidates. 
To be fully transparent, for the scope of the results below we used tcmalloc as part of the gperftools package available on Ubuntu distributions version 2.9 and jemalloc 5.1.0-1.

#### Memory allocator benchmarks

Again, we first compare performance against frameworks executing in an eager fashion. 
This is potentially the use case where the allocator can play the biggest role: As the graph is unknown before its execution, each framework must manage the memory required for each operation when it meets the actual execution of the above node, no planning ahead possible. 
In this context, the allocator is a major component due to all the system calls to allocate and reclaim memory.

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="PyTorch memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_pytorch_latency.svg" />
  <figcaption>Figure 13. PyTorch memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_tensorflow_latency.svg" />
  <figcaption>Figure 14. Google's TensorFlow memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow with oneDNN enabled memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_tensorflow_onednn_latency.svg" />
  <figcaption>Figure 15. Google's TensorFlow with oneDNN enabled memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel TensorFlow memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_intel_tensorflow_latency.svg" />
  <figcaption>Figure 16. Intel TensorFlow memory allocator and cores scaling latencies</figcaption>
</figure>
<br>

As per the graph above, you can notice that the standard library allocator (glibc) is often behind performance-wise but provides reasonable performance. 
Jemalloc allocator is sometimes the fastest around but in very specific situations, where the concurrency is not that high, this can be explained by the underlying structure jemalloc uses 
internally which is out of the scope of this blog, but you can read the [Facebook Engineering blog](https://engineering.fb.com/2011/01/03/core-data/scalable-memory-allocation-using-jemalloc/) if you want to know more about it.

Finally, tcmalloc seems to be the one providing generally best performances across all the workloads benchmarked here. 
Again, tcmalloc has a different approach than Jemalloc in the way it allocates resources, especially tcmalloc maintains a pool of memory segments locally for each thread, which reduces the necessity to have global, exclusive, critical paths. 

Again, for more details, I invite you to read the full [blog by Google Abseil team](https://abseil.io/blog/20200212-tcmalloc).

Now, back to the graph mode where we benchmark framework having an omniscient representation of the overall computation graph.

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="TorchScript memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_torchscript_latency.svg" />
  <figcaption>Figure 17. TorchScript memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_tensorflow_graph_latency.svg" />
  <figcaption>Figure 18. Google's TensorFlow memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Google's TensorFlow with oneDNN enabled memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_tensorflow_onednn_graph_latency.svg" />
  <figcaption>Figure 19. Google's TensorFlow with oneDNN enabled memory allocator and cores scaling latencies</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Intel TensorFlow memory allocator and cores scaling latencies" src="assets/35_bert_cpu_scaling_part_2/allocators/allocator_and_cores_intel_tensorflow_graph_latency.svg" />
  <figcaption>Figure 20. Intel TensorFlow memory allocator and cores scaling latencies</figcaption>
</figure>
<br>


This time, by knowing the underlying structure of the operator flows and matrix shapes involved then the framework can plan and reserve the required resources beforehand. 
In this context, and as it is shown in the chart above, the difference between framework is very small and there is no clear winner between jemalloc and tcmalloc. 
Of course, glibc is still slightly behind as a general-purpose memory allocator, but the margin is less significant than in the eager setup.
To sum it up, tuning the memory allocator can provide an interesting item to grab the last milliseconds' improvement at the end of the optimization process, especially if you are already using traced computation graphs.


### OpenMP

In the previous section we talked about the memory management within machine learning software involving mostly CPU-bound workloads. 
Such software often relies on intermediary frameworks such as PyTorch or TensorFlow for Deep Learning which commonly abstract away all the underlying, highly parallelized, operator implementations. 

Writing such highly parallel and optimized algorithms is a real engineering challenge, and it requires a very low-level understanding of all the actual elements coming into play 
operated by the CPU (synchronization, memory cache, cache validity, etc.). 
In this context, it is very important to be able to leverage primitives to implement such powerful algorithms, reducing the delivery time and computation time by a large margin
compared to implementing everything from scratch.

There are many libraries available which provide such higher-level features to accelerate the development of algorithms. 
Among the most common, one can look at OpenMP, Thread Building Blocks and directly from the C++ when targeting a recent version of the standard. 
In the following part of this blog post, we will restrict ourselves to OpenMP and especially comparing the GNU, open source and community-based implementation, to the Intel OpenMP one. 
The latter especially targets Intel CPUs and is optimized to provide best of class performances when used as a drop-in replacement against the GNU OpenMP one.

OpenMP exposes [many environment variables](https://www.openmp.org/spec-html/5.0/openmpch6.html) to automatically configure the underlying resources which will be involved in the computations, 
such as the number of threads to use to dispatch computation to (intra-op threads), the way the system scheduler should bind each of these threads with respect to the CPU resources (threads, cores, sockets) 
and some other variables which bring further control to the user. 
Intel OpenMP exposes [more of these environment variables](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compilation/supported-environment-variables.html) to provide the user even more flexibility to adjust the performance of its software.

<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="OpenMP vs Intel OpenMP latencies running PyTorch" src="assets/35_bert_cpu_scaling_part_2/openmp/openmp_pytorch_latencies.svg" />
  <figcaption>Figure 21. OpenMP vs Intel OpenMP latencies running PyTorch</figcaption>
</figure>
<br>
<br>
<figure class="image">
  <medium-zoom background="rgba(0,0,0,.7)" alt="OpenMP vs Intel OpenMP latencies running PyTorch" src="assets/35_bert_cpu_scaling_part_2/openmp/openmp_torchscript_latency.svg" />
  <figcaption>Figure 22. OpenMP vs Intel OpenMP latencies running PyTorch</figcaption>
</figure>
<br>

As stated above, tuning OpenMP is something you can start to tweak when you tried all the other, system related, tuning knobs. 
It can bring a final speed up to you model with just a single environment variable to set. 

Also, it is important to note that tuning OpenMP library will only work within software that uses the OpenMP API internally. 
More specially, now only PyTorch and TorchScript really make usage of OpenMP and thus benefit from OpenMP backend tuning. 

This also explains why we reported latencies only for these two frameworks.


## Automatic Performances Tuning: Bayesian Optimization with Intel SigOpt

As mentioned above, many knobs can be tweaked to improve latency and throughput on Intel CPUs, but because there are many, tuning all of them to get optimal performance can be cumbersome. 
For instance, in our experiments, the following knobs were tuned:

- The number of cores: although using as many cores as you have is often a good idea, it does not always provide the best performance because it also means more communication between the different threads. On top of that, having better performance with fewer cores can be very useful as it allows to run multiple instances at the same time, resulting in both better latency and throughput.
- The memory allocator: which memory allocator out of the default malloc, Google's tcmalloc and Facebook's jemalloc provides the best performance?
- The parallelism library: which parallelism library out of GNU OpenMP and Intel OpenMP provides the best performance?
- Transparent Huge Pages: does enabling Transparent Huge Pages (THP) on the system provide better performance?
- KMP block time parameter: sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping.

Of course, the brute force approach, consisting of trying out all the possibilities will provide the best knob values to use to get optimal performance but, 
the size of the search space being `N x 3 x 2 x 2 x 2 = 24N`, it can take a lot of time: on a machine with 80 physical cores, this means trying out at most `24 x 80 = 1920` different setups! 😱

Fortunately, Intel's [SigOpt](https://sigopt.com/), through Bayesian optimization, allows us to make these tuning experiments both faster and more convenient to analyse, while providing similar performance than the brute force approach.

When we analyse the relative difference between the absolute best latency and what SigOpt provides, we observe that although it is often not as good as brute force (except for sequence length = 512 in that specific case),
it gives very close performance, with **8.6%** being the biggest gap on this figure.


<table class="centered">
  <tr>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="Absolute best latency found by SigOpt automatic tuning vs brute force" src="assets/35_bert_cpu_scaling_part_2/sigopt/Intel%20Ice%20lake%20Xeon%208380%20-%20TorchScript%20-%20Batch%20Size%201%20-%20Absolute%20Best%20Latency%20vs%20SigOpt%20Best%20Latency.svg" />
            <figcaption>Figure 23. Absolute best latency found by SigOpt automatic tuning vs brute force</figcaption>
        </figure>
    </td>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="Relative best latency found by SigOpt automatic tuning vs brute force" src="assets/35_bert_cpu_scaling_part_2/sigopt/Intel%20Ice%20lake%20Xeon%208380%20-%20TorchScript%20-%20Batch%20Size%201%20-%20Relative%20Difference%20Absolute%20Best%20Latency%20vs%20SigOpt%20Best%20Latency.svg" />
            <figcaption>Figure 24. Relative best latency found by SigOpt automatic tuning vs brute force</figcaption>
        </figure>
    </td>
   </tr>
</table>


SigOpt is also very useful for analysis: it provides a lot of figures and valuable information.
First, it gives the best value it was able to find, the corresponding knobs, and the history of trials and how it improved as trials went, for example, with sequence length = 20:


<table>
  <tr>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="SigOpt best value display" src="assets/35_bert_cpu_scaling_part_2/sigopt/sigopt_best_value.png" />
            <figcaption>Figure 25. SigOpt best value reporting</figcaption>
        </figure>
    </td>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="SigOpt best value display" src="assets/35_bert_cpu_scaling_part_2/sigopt/sigopt_improvements_over_time.png" />
            <figcaption>Figure 26. SigOpt best value reporting</figcaption>
        </figure>
    </td>
   </tr>
</table>

In this specific setup, 16 cores along with the other knobs were able to give the best results, that is very important to know, because as mentioned before,
that means that multiple instances of the model can be run in parallel while still having the best latency for each.

It also shows that it had converged at roughly 20 trials, meaning that maybe 25 trials instead of 40 would have been enough.
A wide range of other valuable information is available, such as Parameter Importance:

As expected, the number of cores is, by far, the most important parameter, but the others play a part too, and it is very experiment dependent. 
For instance, for the sequence length = 512 experiment, this was the Parameter Importance:


<table>
  <tr>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="SigOpt best value for Batch Size = 1, Sequence Length = 20" src="assets/35_bert_cpu_scaling_part_2/sigopt/sigopt_parameters_importance_seq_20.png" />
            <figcaption>Figure 27. SigOpt best value for Batch Size = 1, Sequence Length = 20</figcaption>
        </figure>
    </td>
    <td>
        <figure class="image">
            <medium-zoom background="rgba(0,0,0,.7)" alt="SigOpt best value for Batch Size = 1, Sequence Length = 512" src="assets/35_bert_cpu_scaling_part_2/sigopt/sigopt_parameters_importance_seq_512.png" />
            <figcaption>Figure 28. SigOpt best value for Batch Size = 1, Sequence Length = 512</figcaption>
        </figure>
    </td>
   </tr>
</table>


Here not only the impact of using OpenMP vs Intel OpenMP was bigger than the impact of the allocator, the relative importance of each knob is more balanced than in the sequence length = 20 experiment.
And many more figures, often interactive, are available on SigOpt such as:
- 2D experiment history, allowing to compare knobs vs knobs or knobs vs objectives
- 3D experiment history, allowing to do the same thing as the 2D experiment history with one more knob / objective.


## Conclusion - Accelerating Transformers for Production

In this post, we...

At Hugging Face, we are on a mission to democratize state of the art Machine Learning, and a critical part of our work is to make these state of the art models as efficient as possible, to use less energy and memory at scale, and to be more affortable to run by companies of all sizes. 

Our collaboration with Intel through the 🤗 [Hardware Partner Program](https://huggingface.co/hardware) enables us to make advanced efficiency and optimization techniques easily available to the community, through our new 🤗 [Optimum open source library](https://github.com/huggingface/optimum) dedicated to production performance.

For companies looking to accelerate their Transformer models inference, our new 🤗 [Infinity product offers a plug-and-play containerized solution](https://huggingface.co/infinity), achieving down to 1ms latency on GPU and 2ms on Intel Xeon Ice Lake CPUs.

If you found this post interesting or useful to your work, please consider giving Optimum a star. And if this post was music to your ears, consider [joining our Machine Learning Optimization team](https://apply.workable.com/huggingface/)!


