---
title: "Machine Learning Experts - Margaret Mitchell"
thumbnail: /blog/assets/57_meg_mitchell_interview/thumbnail.png
---

<h1>Machine Learning Experts - Margaret Mitchell</h1>

<div class="blog-metadata">
    <small>Published March 23, 2022.</small>
</div>

<div class="author-card">
    <a href="/britneymuller">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1645809068511-5ef0ce775e979253a010ef4c.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>britneymuller</code>
            <span class="fullname">Britney Muller</span>
        </div>
    </a>
</div>

Hey friends! Welcome to Machine Learning Experts. I'm your host, Britney Muller and today’s guest is none other than [Margaret Mitchell](https://twitter.com/mmitchell_ai) (Meg for short). Meg founded & co-led Google’s Ethical AI Group, is a pioneer in the field of Machine Learning, has published over 50 papers, and is a leading researcher in Ethical AI.

You’ll hear Meg talk about the moment she realized the importance of ethical AI (an incredible story!), how ML teams can be more aware of harmful data bias, and the power (and performance) benefits of inclusion and diversity in ML.

<a href="https://huggingface.co/support?utm_source=blog&utm_medium=blog&utm_campaign=ml_experts&utm_content=meg_interview_article"><img src="/blog/assets/57_meg_mitchell_interview/meg-cta.png"></a>

Very excited to introduce this powerful episode to you! Here’s my conversation with Meg Mitchell:

<iframe width="853" height="480" src="https://www.youtube.com/embed/FpIxYGyJBbs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Transcription:

*Note: Transcription has been slightly modified/reformatted to deliver the highest-quality reading experience.*

### Could you share a little bit about your background and what brought you to Hugging Face?

**Dr. Margaret Mitchell’s Background:**

- Bachelor’s in Linguistics at Reed College - Worked on NLP
- Worked on assistive and augmentative technology after her Bachelor’s and also during her graduate studies
- Master’s in Computational Linguistics at the University of Washington
- PhD in Computer Science

**Meg:** I did heavy statistical work as a postdoc at Johns Hopkins and then went to Microsoft Research where I continued doing vision to language generation that led to working on an app for people who are blind to navigate the world a bit easier called [Seeing AI](https://www.microsoft.com/en-us/ai/seeing-ai).

After a few years at Microsoft, I left to work at Google to focus on big data problems inherent in deep learning. That’s where I started focusing on things like fairness, rigorous evaluation for different kinds of issues, and bias. While at Google, I founded and co-led the Ethical AI Team which focuses on inclusion and transparency.

After four years at Google, I came over to Hugging Face where I was able to jump in and focus on coding.
I’m helping to create protocols for ethical AI research, inclusive hiring, systems, and setting up a good culture here at Hugging Face.

### When did you recognize the importance of Ethical AI?

**Meg:** This occurred when I was working at Microsoft while I was working on the assistance technology, Seeing AI. In general, I was working on generating language from images and I started to see was how lopsided data was. Data represents a subset of the world and it influences what a model will say.

So I began to run into issues where white people would be described as ‘people’ and black people would be described as ‘black people’ as if white was a default and black was a marked characteristic. That was concerning to me.

There was also an ah-ha moment when I was feeding my system a sequence of images, getting it to talk more about a story of what is happening. And I fed it some images of this massive blast where a lot of people worked, called the ‘Hebstad blast’. You could see that the person taking the picture was on the second or third story looking out on the blast. The blast was very close to this person. It was a very dire and intense moment and when I fed this to the system the system’s output was that “ this is awesome, this is a great view, this is beautiful’. And I thought.. this is a great view of this horrible scene but the important part here is that people may be dying. This is a massive destructive explosion. 

But the thing is, when you’re learning from images people don’t tend to take photos of terrible things, they take photos of sunsets, fireworks, etc., and a visual recognition model had learned on these images and believed that color in the sky was a positive, beautiful thing.

At that moment, I realized that if a model with that sort of thinking had access to actions it would be just one hop away from a system that would blow up buildings because it thought it was beautiful.

This was a moment for me when I realized I didn’t want to keep making these systems do better on benchmarks, I wanted to fundamentally shift how we were looking at these problems, how we were approaching data and analysis of data, how we were evaluating and all of the factors we were leaving out with these straightforward pipelines.

So that really became my shift into ethical AI work.

### In what applications is data ethics most important?

**Meg:** Human-centric technology that deals with people and identity (face recognition, pedestrian recognition). In NLP this would pertain more to the privacy of individuals, how individuals are talked about, and the biases models pick up with regards to descriptors used for people.

### How can ML teams be more aware of harmful bias?

**Meg:** A primary issue is that these concepts haven't been taught and most teams simply aren’t aware. Another problem is the lack of a lexicon to contextualize and communicate what is going on.

For example:
- This is what marginalization is
- This is what a power differential is
- Here is what inclusion is
- Here is how stereotypes work

Having a better understanding of these pillars is really important.

Another issue is the culture behind machine learning. It’s taken a bit of an ‘Alpha’ or ‘macho’ approach where the focus is on ‘beating’ the last numbers, making things ‘faster’, ‘bigger’, etc. There are lots of parallels that can be made to human anatomy.

There’s also a very hostile competitiveness that comes out where you find that women are disproportionately treated as less than.

Since women are often much more familiar with discrimination women are focusing a lot more on ethics, stereotypes, sexism, etc. within AI. This means it gets associated with women more and seen as less than which makes the culture a lot harder to penetrate.

It’s generally assumed that I’m not technical. It’s something I have to prove over and over again. I’m called a linguist, an ethicist because these are things I care about and know about but that is treated as less-than. People say or think, “You don’t program, you don’t know about statistics, you are not as important,” and it’s often not until I start talking about things technically that people take me seriously which is unfortunate.

There is a massive cultural barrier in ML.

### Lack of diversity and inclusion hurts everyone

**Meg:** Diversity is when you have a lot of races, ethnicities, genders, abilities, statuses at the table.
Inclusion is when each person feels comfortable talking, they feel welcome.

One of the best ways to be more inclusive is to not be exclusive. Feels fairly obvious but is often missed. People get left out of meetings because we don’t find them helpful or find them annoying or combative (which is a function of various biases). To be inclusive you need to not be exclusive so when scheduling a meeting pay attention to the demographic makeup of the people you’re inviting. If your meeting is all-male, that’s a problem. 

It’s incredibly valuable to become more aware and intentional about the demographic makeup of the people you’re including in an email. But you’ll notice in tech, a lot of meetings are all male, and if you bring it up that can be met with a lot of hostility. Air on the side of including people.

We all have biases but there are tactics to break some of those patterns. When writing an email I’ll go through their gender and ethnicities to ensure I’m being inclusive. It’s a very conscious effort. That sort of thinking through demographics helps. However, mention this before someone sends an email or schedules a meeting. People tend to not respond as well when you mention these things after the fact.

### Diversity in AI - Isn’t there proof that having a more diverse set of people on an ML project results in better outcomes?

**Meg:** Yes, since you have different perspectives you have a different distribution over options and thus, more options. One of the fundamental aspects of machine learning is that when you start training you can use a randomized starting point and what kind of distribution you want to sample from.

Most engineers can agree that you don’t want to sample from one little piece of the distribution to have the best chance of finding a local optimum.

You need to translate this approach to the people sitting at the table.

Just how you want to have a Gaussian approach over different start states, so too do you want that at the table when you’re starting projects because it gives you this larger search space making it easier to attain a local optimum.

### Can you talk about Model Cards and how that project came to be?

**Meg:** This project started at Google when I first started working on fairness and what a rigorous evaluation of fairness would look like.

In order to do that you need to have an understanding of context and understanding of who would use it. This revolved around how to approach model biases and it wasn’t getting a lot of pick up. 

I was talking to [Timnit Gebru](https://twitter.com/timnitGebru) who was at that time someone in the field with similar interest to me and she was talking about this idea of datasheets; a kind of documentation for data (based on her experience at Apple) doing engineering where you tend to have specifications of hardware. But we don’t have something similar for data and she was talking about how crazy that is. 

So Timnit had this idea of datasheets for datasets. It struck me that by having an ‘artifact’ people in tech who are motivated by launches would care a lot more about it. So if we say you have to produce this artifact and it will count as a launch suddenly people would be more incentivized to do it.

The way we came up with the name was that a comparable word to ‘data sheet’ that could be used for models was card (plus it was shorter). Also decided to call it ‘model cards’ because the name was very generic and would have longevity over time.


Timnit’s paper was called [‘Data Sheets for Datasets’](https://arxiv.org/abs/1803.09010). So we called ours [‘Model Cards for Model Reporting’](https://arxiv.org/abs/1810.03993) and once we had the published paper people started taking us more seriously. Couldn’t have done this without Timnit Gebru’s brilliance suggesting “You need an artifact, a standardized thing that people will want to produce.”


### Where are model cards headed?

**Meg:** There’s a pretty big barrier to entry to do model cards in a way that is well informed by ethics. Partly because the people who need to fill this out are often engineers and developers who want to launch their model and don’t want to sit around thinking about documentation and ethics.

Part of why I wanted to join Hugging Face is because it gave me an opportunity to standardize how these processes could be filled out and automated as much as possible. One thing I really like about Hugging Face is there is a focus on creating end-to-end machine learning processes that are as smooth as possible. Would love to do something like that with model cards where you could have something largely automatically generated as a function of different questions asked or even based on model specifications directly.

We want to work towards having model cards as filled out as possible and interactive. Interactivity would allow you to see the difference in false-negative rate as you move the decision threshold. Normally with classification systems, you set some threshold at which you say yes or no, like .7, but in practice, you actually want to vary the decision threshold to trade off different errors. 

A static report of how well it works isn’t as informative as you want it to be because you want to know how well it works as different decision thresholds are chosen, and you could use that to decide what decision threshold to be used with your system. So we created a model card where you could interactively change the decision threshold and see how the numbers change. Moving towards that direction in further automation and interactivity is the way to go.

### Decision thresholds & model transparency

**Meg:** When Amazon first started putting out facial recognition and facial analysis technology it was found that the gender classification was disproportionately bad for black women and Amazon responded by saying “this was done using the wrong decision threshold”. And then one of the police agencies who had been using one of these systems had been asked what decision threshold they had been using and said, “Oh we’re not using a decision threshold,”. 

Which was like oh you really don’t understand how this works and are using this out of the box with default parameter settings?! That is a problem. So minimally having this documentary brings awareness to decisions around the various types of parameters.

Machine learning models are so different from other things we put out into the public. Toys, medicine, and cars have all sorts of regulations to ensure products are safe and work as intended. We don’t have that in machine learning, partly because it’s new so the laws and regulations don’t exist yet. It’s a bit like the wild west, and that’s what we’re trying to change with model cards.

### What are you working on at Hugging Face?

- Working on a few different tools designed for engineers.
- Working on philosophical and social science research: Just did a deep dive into UDHR (Universal Declaration of Human Rights) and how those can be applied with AI. Trying to help bridge the gaps between AI, ML, law, and philosophy.
- Trying to develop some statistical methods that are helpful for testing systems as well as understanding datasets.
- We also recently [put out a tool](https://huggingface.co/spaces/huggingface/data-measurements-tool) that shows how well a language maps to Zipfian distributions (how natural language tends to go) so you can test how well your model is matching with natural language that way.
- Working a lot on the culture stuff: spending a lot of time on hiring and what processes we should have in place to be more inclusive.
- Working on [Big Science](https://bigscience.huggingface.co/): a massive effort with people from all around the world, not just hugging face working on data governance (how can big data be used and examined without having it proliferate all over the world/being tracked with how it’s used).
- Occasionally I’ll do an interview or talk to a Senator, so it’s all over the place.
- Try to answer emails sometimes.

*Note: Everyone at Hugging Face wears several hats.* :)

### Meg’s impact on AI

Meg is featured in the book [Genius Makers ‘The Mavericks who brought AI to Google, Facebook, and the World’](https://www.amazon.com/Genius-Makers-Mavericks-Brought-Facebook/dp/1524742678). Cade Metz interviewed Meg for this while she was at Google.

Meg’s pioneering research, systems, and work have played a pivotal role in the history of AI. (we are so lucky to have her at Hugging Face!)

### Rapid Fire Questions:

### Best piece of advice for someone looking to get into AI?

**Meg:** Depends on who the person is. If they have marginalized characteristics I would give very different advice. For example, if it was a woman I would say, 'Don’t listen to your supervisors saying you aren’t good at this. Chances are you are just thinking about things differently than they are used to so have confidence in yourself.'

If it’s someone with more majority characteristics I’d say, 'Forget about the pipeline problem, pay attention to the people around you and make sure that you hold them up so that the pipeline you’re in now becomes less of a problem.'

Also, 'Evaluate your systems'.

### What industries are you most excited to see ML applied (or ML Ethics be applied)

**Meg:** The health and assistive domains continue to be areas I care a lot about and see a ton of potential.

Also want to see systems that help people understand their own biases. Lots of technology is being created to screen job candidates for job interviews but I feel that technology should really be focused on the interviewer and how they might be coming at the situation with different biases. Would love to have more technology that assists humans to be more inclusive instead of assisting humans to exclude people.

### You frequently include incredible examples of biased models in your Keynotes and interviews. One in particular that I love is the criminal detection model you've talked about that was using patterns of mouth angles to identify criminals (which you swiftly debunked).

**Meg:** Yes, [the example is that] they were making this claim that there was this angle theta that was more indicative of criminals when it was a smaller angle. However, I was looking at the math and I realized that what they were talking about was a smile! Where you would have a wider angle for a smile vs a smaller angle associated with a straight face. They really missed the boat on what they were actually capturing there. Experimenter's bias: wanting to find things that aren’t there.

### Should people be afraid of AI taking over the world?

**Meg:** There are a lot of things to be afraid of with AI. I like to see it as we have a distribution over different kinds of outcomes, some more positive than others, so there’s not one set one that we can know. There are a lot of different things where AI can be super helpful and more task-based over more generalized intelligence. You can see it going in another direction, similar to what I mentioned earlier about a model thinking something destructive is beautiful is one hop away from a system that is able to press a button to set off a missile. Don’t think people should be scared per se, but they should think about the best and worst-case scenarios and try to mitigate or stop the worst outcomes.

I think the biggest thing right now is these systems can widen the divide between the haves and have nots. Further giving power to people who have power and further worsening things for people who don’t. The people designing these systems tend to be people with more power and wealth and they design things for their kinds of interest. I think that’s happening right now and something to think about in the future.

Hopefully, we can focus on the things that are most beneficial and continue heading in that direction.

### Fav ML papers?

**Meg:** Most recently I’ve really loved what [Abeba Birhane](https://abebabirhane.github.io) has been doing on [values that are encoded in machine learning](https://arxiv.org/abs/2106.15590). My own team at Google had been working on [data genealogies](https://journals.sagepub.com/doi/full/10.1177/20539517211035955), bringing critical analysis on how ML data is handled which they have a few papers on - for example, [Data and its (dis)contents: A survey of dataset development and use in machine learning research](https://arxiv.org/abs/2012.05345). Really love that work and might be biased because it included my team and direct reports, I’m very proud of them but it really is fundamentally good work.

Earlier papers that I’m interested in are more reflective of what I was doing at that time. Really love the work of [Herbert Clark](https://neurotree.org/beta/publications.php?pid=4636) who was a psycholinguistics/communications person and he did a lot of work that is easily ported to computational models about how humans communicate. Really love his work and cite him a lot throughout my thesis.

### Anything else you would like to mention?

**Meg:** One of the things I’m working on, that I think other people should be working on, is lowering the barrier of entry to AI for people with different academic backgrounds.

We have a lot of people developing technology, which is great, but we don’t have a lot of people in a situation where they can really question the technology because there is often a bottleneck.

For example, if you want to know about data directly you have to be able to log into a server and write a SQL query. So there is a bottleneck where engineers have to do it and I want to remove that barrier. How can we take things that are fundamentally technical code stuff and open it up so people can directly query the data without knowing how to program?

We will be able to make better technology when we remove the barriers that require engineers to be in the middle.

### Outro

**Britney:** Meg had a hard stop on the hour but I was able to ask her my last question offline: What’s something you’ve been interested in lately? Meg’s response: "How to propagate and grow plants in synthetic/controlled settings." Just when I thought she couldn’t get any cooler. 🤯

I’ll leave you with a recent quote from Meg in a [Science News article on Ethical AI](https://www.sciencenews.org/article/computer-science-history-ethics-future-robots-ai):

*“The most pressing problem is the diversity and inclusion of who’s at the table from the start. All the other issues fall out from there.” -Meg Mitchell.*

Thank you for listening to Machine Learning Experts!

<a href="https://huggingface.co/support?utm_source=blog&utm_medium=blog&utm_campaign=ml_experts&utm_content=meg_interview_article"><img src="/blog/assets/57_meg_mitchell_interview/meg-cta.png"></a>

**Honorable mentions + links:**
- [Emily Bender](https://twitter.com/emilymbender?lang=en)
- [Ehud Reiter](https://mobile.twitter.com/ehudreiter)
- [Abeba Birhane](https://abebabirhane.github.io/)
- [Seeing AI](https://www.microsoft.com/en-us/ai/seeing-ai)
- [Data Sheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Model Cards](https://modelcards.withgoogle.com/about)
- [Model Cards Paper](https://arxiv.org/abs/1810.03993)
- [Abeba Birhane](https://arxiv.org/search/cs?searchtype=author&query=Birhane%2C+A)
- [The Values Encoded in Machine Learning Research](https://arxiv.org/abs/2106.15590)
- [Data and its (dis)contents:](https://arxiv.org/abs/2012.05345)
- [Herbert Clark](https://neurotree.org/beta/publications.php?pid=4636)

**Follow Meg Online:**
- [Twitter](https://twitter.com/mmitchell_ai)
- [Website](http://www.m-mitchell.com)
- [LinkedIn](https://www.linkedin.com/in/margaret-mitchell-9b13429)

