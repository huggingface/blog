---
title: "Rearchitecting Hugging Face Uploads and Downloads"
thumbnail: /blog/assets/rearchitecting-uploads-and-downloads/thumbnail.png
authors:
  - user: port8080
  - user: jsulz
  - user: erinys
---

# Rearchitecting Hugging Face Uploads and Downloads

As part of Hugging Face's Xet team’s work to [improve Hugging Face Hub’s storage backend](https://huggingface.co/blog/xethub-joins-hf), we analyzed a 24 hour window of Hugging Face upload requests to better understand access patterns. On October 11th, 2024, we saw:

- Uploads from 88 countries
- 8.2 million upload requests
- 130.8 TB of data transferred

The map below visualizes this activity, with countries colored by bytes uploaded per hour.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/animated-uploads-choropleth.gif" alt="Animated view of uploads" width=100%>
</p>

Currently, uploads are stored in an [S3 bucket](https://aws.amazon.com/s3/) in **`us-east-1`** and optimized using [S3 Transfer Acceleration](https://aws.amazon.com/s3/transfer-acceleration/). Downloads are cached and served using [AWS Cloudfront](https://aws.amazon.com/cloudfront/) as a CDN. Cloudfront’s [400+ convenient edge locations](https://aws.amazon.com/blogs/networking-and-content-delivery/400-amazon-cloudfront-points-of-presence/) provide global coverage and low-latency data transfers. However, like most CDNs, it is optimized for web content and has a file size limit of 50GB.

While this size restriction is reasonable for typical internet file transfers, the ever-growing size of files in model and dataset repositories presents a challenge. For instance, the weights of [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) total 131GB and are split across 30 files to meet the Hub’s recommendation of chunking weights into [20 GB segments](https://huggingface.co/docs/hub/en/repositories-recommendations#recommendations).

## A Custom Protocol for Uploads and Downloads

To push Hugging Face infrastructure beyond its current limits, we are redesigning the Hub’s upload and download architecture. We plan to insert a [content-addressed store (CAS)](https://en.wikipedia.org/wiki/Content-addressable_storage) as the first stop for content distribution. This enables us to implement a custom protocol built on a guiding philosophy of **_dumb reads and smart writes_**. Unlike Git LFS, which treats files as opaque blobs, our approach analyzes files at the byte level, uncovering opportunities to improve transfer speeds for the massive files found in model and dataset repositories.

The read path prioritizes simplicity and speed to ensure high throughput with minimal latency. Requests for a file are routed to a CAS server, which provides reconstruction information. The data itself remains backed by an S3 bucket in **`us-east-1`**, with AWS CloudFront continuing to serve as the CDN for downloads.

The write path is more complex to optimize upload speeds and provide additional security guarantees. Like reads, upload requests are routed to a CAS server, but instead of querying at the file level [we operate on chunks](https://huggingface.co/blog/from-files-to-chunks). As matches are found, the CAS server instructs the client (e.g., [huggingface_hub](https://github.com/huggingface/huggingface_hub)) to transfer only the necessary (new) chunks. The chunks are validated by CAS before uploading them to S3.

There are many implementation details to address, such as network constraints or storage overhead, which we’ll cover in future posts. For now, let's look at how reads currently look. The first diagram below shows the read and write paths as they currently look today:

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/old-read-write-path.png" alt="Old read and write sequence diagram" width=100%>
    <figcaption> Reads are represented on the left; writes are to the right. Note that writes go directly to S3 without any intermediary.</figcaption>
</figure>

Meanwhile, in the new design, reads will take the following path:

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/new-reads.png" alt="New read path in proposed architecture">
    <figcaption>New read path with a content addressed store (CAS) providing reconstruction information. Cloudfront continues to act as a CDN.</figcaption>
</figure>

and finally here is the updated write path:

<figure class="image text-center" width=90%>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/new-writes.png" alt="New read path in proposed architecture" >
    <figcaption>New write path with CAS speeding up and validating uploads. S3 continues to provide backing storage.</figcaption>
</figure>

By managing files at the byte level, we can adapt optimizations to suit different file formats. For instance, we have explored [improving the dedupeability of Parquet files](https://huggingface.co/blog/improve_parquet_dedupe), and are now investigating compressing tensor files (e.g., [Safetensors](https://github.com/huggingface/safetensors)) which have the potential to trim 10-25% off upload speeds. As new formats emerge, we are uniquely positioned to develop further enhancements that improve the development experience on the Hub.

This protocol also introduces significant improvements for enterprise customers and power users. Inserting a control plane for file transfers provides added guarantees to ensure malicious or invalid data cannot be uploaded. Operationally, uploads are no longer a black box. Enhanced telemetry provides audit trails and detailed logging, enabling the Hub infrastructure team to identify and resolve issues quickly and efficiently.

## Designing for Global Access

To support this custom protocol, we need to determine the optimal geographic distribution for the CAS service. [AWS Lambda@Edge](https://aws.amazon.com/lambda/edge/) was initially considered for its extensive global coverage to help minimize the round-trip time. However, its reliance on Cloudfront triggers made it incompatible with our updated upload path. Instead, we opted to deploy CAS nodes in a select few of AWS’s 34 regions.

Taking a closer look at our 24-hour window of S3 PUT requests, we identified global traffic patterns that reveal the distribution of data uploads to the Hub. As expected, the majority of activity comes from North America and Europe, with continuous, high-volume uploads throughout the day. The data also highlights a strong and growing presence in Asia. By focusing on these core regions, we can place our CAS [points of presence](https://docs.aws.amazon.com/whitepapers/latest/aws-fault-isolation-boundaries/points-of-presence.html) to balance storage and network resources while minimizing latency.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/pareto-chart.png" alt="Pareto chart of uploads" width=100%>
</p>

While AWS offers 34 regions, our goal is to keep infrastructure costs reasonable while maintaining a high user experience. Out of the 88 countries represented in this snapshot, the Pareto chart above shows that the top 7 countries account for 80% of uploaded bytes, while the top 20 countries contribute 95% of the total upload volume and requests.

The United States emerges as the primary source of upload traffic, necessitating a PoP in this region. In Europe, most activity is concentrated in central and western countries (e.g., Luxembourg, the United Kingdom, and Germany) though there is some additional activity to account for in Africa (specifically Algeria, Egypt, and South Africa). Asia’s upload traffic is primarily driven by Singapore, Hong Kong, Japan, and South Korea.

If we use a simple heuristic to distribute traffic, we can divide our CAS coverage into three major regions:

- **`us-east-1`**: Serving North and South America
- **`eu-west-3`**: Serving Europe, the Middle East, and Africa
- **`ap-southeast-1`**: Serving Asia and Oceania

This ends up being quite effective. The US and Europe account for 78.4% of uploaded bytes, while Asia accounts for 21.6%.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rearchitecting-uploads-and-downloads/aws-regions.png" alt="New AWS mapping" width=100%>
</p>

This regional breakdown results in a well-balanced load across our three CAS PoPs, with additional capacity for growth in **`ap-southeast-1`** and flexibility to scale up in **`us-east-1`** and **`eu-west-3`** as needed.

Based on expected traffic, we plan to allocate resources as follows:

- **`us-east-1`**: 4 nodes
- **`eu-west-3`**: 4 nodes
- **`ap-southeast-1`**: 2 nodes

## Validating and Vetting

Even though we’re increasing the first hop distance for some users, the overall impact to bandwidth across the Hub will be limited. Our estimates predict that while the cumulative bandwidth for all uploads will decrease from 48.5 Mbps to 42.5 Mbps (a 12% reduction), the performance hit will be more than offset by other system optimizations.

We are currently working toward moving our infrastructure into production by the end of 2024, where we will start with a single CAS in **`us-east-1`**. From there, we’ll start duplicating internal repositories to our new storage system to benchmark transfer performance, and then replicate our CAS to the additional PoPs mentioned above for more benchmarking. Based on those results, we will continue to optimize our approach to ensure that everything works smoothly when our storage backend is fully in place next year.

## Beyond the Bytes

As we continue this analysis, new opportunities for deeper insights are emerging. Hugging Face hosts one of the largest collections of data from the open-source machine learning community, providing a unique vantage point to explore the modalities and trends driving AI development around the world.

For example, future analyses could classify models uploaded to the Hub by use case (such as NLP, computer vision, robotics, or large language models) and examine geographic trends in ML activity. This data not only informs our infrastructure decisions but also provides a lens into the evolving landscape of machine learning.

We invite you to explore our current findings in more detail! Visit [our interactive Space](https://huggingface.co/spaces/xet-team/cas-analysis) to see the upload distribution for your region, and [follow our team](https://huggingface.co/xet-team) to hear more about what we’re building.
