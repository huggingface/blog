---
title: "The New and Fresh analytics in Inference Endpoints"
thumbnail: /blog/assets/endpoint-analytics/thumbnail.png
authors:
- user: erikkaum
- user: beurkinger
- user: rtrm
- user: co42
- user: michellehbn
---

# Analytics is important

Analytics and metrics are the cornerstone of understanding what's happening with your deployment. Are your [Inference Endpoints](https://endpoints.huggingface.co) overloaded? How many requests are they handling? Having well-visualized, relevant metrics displayed in real-time is crucial for monitoring and debugging.

We realized that our analytics dashboard needed a refresh. Since we debug a lot of endpoints ourselves, we’ve felt the same pain as our users. That’s why we sat down to plan and make several improvements to provide a better experience for you.

# What’s New?

⏰ Real-Time Metrics: Data now updates in real-time, ensuring you get an accurate and up-to-the-second view of your endpoint’s performance. Whether you’re monitoring request latency, response times, or error rates, you can now see the events as they happen. We’ve also reworked the backend of our analytics dashboard to ensure that data loads swiftly, especially for high-traffic endpoints. No more waiting around for metrics to populate. Just open the dashboard and get instant insights.

<p align="center">
  <video width="100%" autoplay loop muted playsinline>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/endpoint-analytics/send_request.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

🔬 Customizable Time Ranges & Auto-Refresh: We know that different users need different views, so we’ve made it easier to zoom in on a specific time range or track long-term trends. You can also enable auto-refresh, ensuring that your dashboard stays up to date without needing to manually reload.

<p align="center">
  <video width="100%" autoplay loop muted playsinline>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/endpoint-analytics/custom_time_zoom.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

🔄 Replica Lifecycle View: Understanding what’s happening with your replicas is crucial, so we’ve introduced a detailed view of each replica’s lifecycle. You can now track replicas from initialization to termination, observing every state transition in between. This helps understand what's going on with your endpoint even if you have several moving parts.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/endpoint-analytics/replica_status.png"><br>
</p>

Even though we’ve rolled out these updates, we’re actively iterating on them. Things will continue to improve, and we welcome all feedback.

Let us know what works, what doesn’t, and what you’d like to see next! 🙌

Head to [Inference Endpoints](https://endpoints.huggingface.co) to check out the changes!
