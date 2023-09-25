---
title: "Machine Learning Experts - Lewis Tunstall"
thumbnail: /blog/assets/60_lewis_tunstall_interview/thumbnail.png
authors:
- user: britneymuller
---

<h1>Machine Learning Experts - Lewis Tunstall</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->


## 🤗 Welcome to Machine Learning Experts - Lewis Tunstall

Hey friends! Welcome to Machine Learning Experts. I'm your host, Britney Muller and today’s guest is [Lewis Tunstall](https://twitter.com/_lewtun). Lewis is a Machine Learning Engineer at Hugging Face where he works on applying Transformers to automate business processes and solve MLOps challenges.

Lewis has built ML applications for startups and enterprises in the domains of NLP, topological data analysis, and time series. 

You’ll hear Lewis talk about his [new book](https://transformersbook.com/), transformers, large scale model evaluation, how he’s helping ML engineers optimize for faster latency and higher throughput, and more.

In a previous life, Lewis was a theoretical physicist and outside of work loves to play guitar, go trail running, and contribute to open-source projects.

<a href="https://huggingface.co/support?utm_source=blog&utm_medium=blog&utm_campaign=ml_experts&utm_content=lewis_interview_article"><img src="/blog/assets/60_lewis_tunstall_interview/lewis-cta.png"></a>

Very excited to introduce this fun and brilliant episode to you! Here’s my conversation with Lewis Tunstall:

<iframe width="100%" style="aspect-ratio: 16 / 9;"src="https://www.youtube.com/embed/igW5VWewuLE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

*Note: Transcription has been slightly modified/reformatted to deliver the highest-quality reading experience.*

### Welcome, Lewis! Thank you so much for taking time out of your busy schedule to chat with me today about your awesome work!

**Lewis:** Thanks, Britney. It’s a pleasure to be here.

### Curious if you can do a brief self-introduction and highlight what brought you to Hugging Face? 

**Lewis:** What brought me to Hugging Face was transformers. In 2018, I was working with transformers at a startup in Switzerland. My first project was a question answering task where you input some text and train a model to try and find the answer to a question within that text.

In those days the library was called: pytorch-pretrained-bert, it was a very focused code base with a couple of scripts and it was the first time I worked with transformers. I had no idea what was going on so I read the original [‘Attention Is All You Need’](https://arxiv.org/abs/1706.03762) paper but I couldn’t understand it. So I started looking around for other resources to learn from. 

In the process, Hugging Face exploded with their library growing into many architectures and I got really excited about contributing to open-source software. So around 2019, I had this kinda crazy idea to write a book about transformers because I felt there was an information gap that was missing. So I partnered up with my friend, [Leandro](https://twitter.com/lvwerra) (von Werra) and we sent [Thom](https://twitter.com/Thom_Wolf) (Wolf) a cold email out of nowhere saying, “Hey we are going to write a book about transformers, are you interested?” and I was expecting no response. But to our great surprise, he responded “Yea, sure let’s have a chat.” and around 1.5 years later this is our book: [NLP with Transformers](https://transformersbook.com/).

This collaboration set the seeds for Leandro and I to eventually join Hugging Face. And I've been here now for around nine months. 

### That is incredible. How does it feel to have a copy of your book in your hands? 

**Lewis:** I have to say, I just became a parent about a year and a half ago and it feels kind of similar to my son being born. You're holding this thing that you created. 

It's quite an exciting feeling and so different to actually hold it (compared to reading a PDF). Confirms that it’s actually real and I didn't just dream about it.

### Exactly. Congratulations!

Want to briefly read one endorsement that I love about this book;

“_Complexity made simple. This is a rare and precious book about NLP, transformers, and the growing ecosystem around them, Hugging Face. Whether these are still buzzwords to you or you already have a solid grasp of it all, the authors will navigate you with humor, scientific rigor, and plenty of code examples into the deepest secrets of the coolest technology around. From “off-the-shelf pre-trained” to “from-scratch custom” models, and from performance to missing labels issues, the authors address practically every real-life struggle of an ML engineer and provide state-of-the-art solutions, making this book destined to dictate the standards in the field for years to come._” 
—Luca Perrozi Ph.D., Data Science and Machine Learning Associate Manager at Accenture.

Checkout [Natural Language Processing with Transformers](https://transformersbook.com/).

### Can you talk about the work you've done with the transformers library?

**Lewis:** One of the things that I experienced in my previous jobs before Hugging Face was there's this challenge in the industry when deploying these models into production; these models are really large in terms of the number of parameters and this adds a lot of complexity to the requirements you might have. 

So for example, if you're trying to build a chatbot you need this model to be very fast and responsive. And most of the time these models are a bit too slow if you just take an off-the-shelf model, train it, and then try to integrate it into your application. 

So what I've been working on for the last few months on the transformers library is providing the functionality to export these models into a format that lets you run them much more efficiently using tools that we have at Hugging Face, but also just general tools in the open-source ecosystem.

In a way, the philosophy of the transformers library is like writing lots of code so that the users don't have to write that code.

In this particular example, what we're talking about is something called the ONNX format. It's a special format that is used in industry where you can basically have a model that's written in PyTorch but you can then convert it to TensorFlow or you can run it on some very dedicated hardware.

And if you actually look at what's needed to make this conversion happen in the transformers library, it's fairly gnarly. But we make it so that you only really have to run one line of code and the library will take care of you. 

So the idea is that this particular feature lets machine learning engineers or even data scientists take their model, convert it to this format, and then optimize it to get faster latency and higher throughput. 

### That's very cool. Have there been, any standout applications of transformers?

**Lewis:** I think there are a few. One is maybe emotional or personal, for example many of us when OpenAI released GPT-2, this very famous language model which can generate text.

OpenAI actually provided in their blog posts some examples of the essays that this model had created. And one of them was really funny. One was an essay about why we shouldn't recycle or why recycling is bad.

And the model wrote a compelling essay on why recycling was bad. Leandro and I were working at a startup at the time and I printed it out and stuck it right above the recycling bin in the office as a joke. And people were like, “Woah, who wrote this?” and I said, “An algorithm.”

I think there's something sort of strangely human, right? Where if we see generated text we get more surprised when it looks like something I (or another human) might have written versus other applications that have been happening like classifying text or more conventional tasks.

### That's incredible. I remember when they released those examples for GPT-2, and one of my favorites (that almost gave me this sense of, whew, we're not quite there yet) were some of the more inaccurate mentions like “underwater fires”.

**Lewis:** Exactly!

**Britney:** But, then something had happened with an oil spill that next year, where there were actually fires underwater! And I immediately thought about that text and thought, maybe AI is onto something already that we're not quite aware of?

### You and other experts at Hugging Face have been working hard on the Hugging Face Course. How did that come about & where is it headed? 

**Lewis:** When I joined Hugging Face, [Sylvian](https://twitter.com/GuggerSylvain) and [Lysandre](https://twitter.com/LysandreJik), two of the core maintainers of the transformers library, were developing a course to basically bridge the gap between people who are more like software engineers who are curious about natural language processing but specifically curious about the transformers revolution that's been happening. So I worked with them and others in the open-source team to create a free course called the [Hugging Face Course](https://huggingface.co/course/chapter1/1). And this course is designed to really help people go from knowing kind of not so much about ML all the way through to having the ability to train models on many different tasks.

And, we've released two parts of this course and planning to release the third part this year. I'm really excited about the next part that we're developing right now where we're going to explore different modalities where transformers are really powerful. Most of the time we think of transformers for NLP, but likely there's been this explosion where transformers are being used in things like audio or in computer vision and we're going to be looking at these in detail. 

### What are some transformers applications that you're excited about? 

**Lewis:** So one that's kind of fun is in the course we had an event last year where we got people in the community to use the course material to build applications.

And one of the participants in this event created a cover letter generator for jobs. So the idea is that when you apply for a job there's always this annoying thing you have to write a cover letter and it's always like a bit like you have to be witty. So this guy created a cover letter generator where you provide some information about yourself and then it generates it from that.

And he actually used that to apply to Hugging Face.

### No way?!

**Lewis:** He's joining the Big Science team as an intern. So. I mean this is a super cool thing, right? When you learn something and then use that thing to apply which I thought was pretty awesome. 

### Where do you want to see more ML applications?

**Lewis:** So I think personally, the area that I'm most excited about is the application of machine learning into natural sciences. And that's partly because of my background. I used to be a Physicist in a previous lifetime but I think what's also very exciting here is that in a lot of fields. For example, in physics or chemistry you already know what the say underlying laws are in terms of equations that you can write down but it turns out that many of the problems that you're interested in studying often require a simulation. Or they often require very hardcore supercomputers to understand and solve these equations. And one of the most exciting things to me is the combination of deep learning with the prior knowledge that scientists have gathered to make breakthroughs that weren't previously possible.

And I think a great example is [DeepMind’s Alpha Fold](https://www.deepmind.com/research/highlighted-research/alphafold) model for protein structure prediction where they were basically using a combination of transformers with some extra information to generate predictions of proteins that I think previously were taking on the order of months and now they can do them in days.

So this accelerates the whole field in a really powerful way. And I can imagine these applications ultimately lead to hopefully a better future for humanity.

### How you see the world of model evaluation evolving?

**Lewis:** That's a great question. So at Hugging Face, one of the things I've been working on has been trying to build the infrastructure and the tooling that enables what we call 'large-scale evaluation'. So you may know that the [Hugging Face Hub](https://huggingface.co/models) has thousands of models and datasets. But if you're trying to navigate this space you might ask yourself, 'I'm interested in question answering and want to know what the top 10 models on this particular task are'.

And at the moment, it's hard to find the answer to that, not just on the Hub, but in general in the space of machine learning this is quite hard. You often have to read papers and then you have to take those models and test them yourself manually and that's very slow and inefficient.

So one thing that we've been working on is to develop a way that you can evaluate models and datasets directly through the Hub. We're still trying to experiment there with the direction. But I'm hoping that we have something cool to show later this year. 

And there's another side to this which is that a large part of the measuring progress in machine learning is through the use of benchmarks. These benchmarks are traditionally a set of datasets with some tasks but what's been maybe missing is that a lot of researchers speak to us and say, “Hey, I've got this cool idea for a benchmark, but I don't really want to implement all of the nitty-gritty infrastructure for the submissions, and the maintenance, and all those things.”

And so we've been working with some really cool partners on hosting benchmarks on the Hub directly. So that then people in the research community can use the tooling that we have and then simplify the evaluation of these models. 

### That is super interesting and powerful.

**Lewis:** Maybe one thing to mention is that the whole evaluation question is a very subtle one. We know from previous benchmarks, such as SQuAD, a famous benchmark to measure how good models are at question answering, that many of these transformer models are good at taking shortcuts.

Well, that's the aim but it turns out that many of these transformer models are really good at taking shortcuts. So, what they’re actually doing is they're getting a very high score on a benchmark which doesn't necessarily translate into the actual thing you were interested in which was answering questions.

And you have all these subtle failure modes where the models will maybe provide completely wrong answers or they should not even answer at all. And so at the moment in the research community there's a very active and vigorous discussion about what role benchmarks play in the way we measure progress.

But also, how do these benchmarks encode our values as a community? And one thing that I think Hugging Face can really offer the community here is the means to diversify the space of values because traditionally most of these research papers come from the U.S. which is a great country but it's a small slice of the human experience, right?

### What are some common mistakes machine learning engineers or teams make?

**Lewis:** I can maybe tell you the ones that I've done.

Probably a good representative of the rest of the things. So I think the biggest lesson I learned when I was starting out in the field is using baseline models when starting out. It’s a common problem that I did and then later saw other junior engineers doing is reaching for the fanciest state-of-the-art model.

Although that may work, a lot of the time what happens is you introduce a lot of complexity into the problem and your state-of-the-art model may have a bug and you won't really know how to fix it because the model is so complex. It’s a very common pattern in industry and especially within NLP is that you can actually get quite far with regular expressions and linear models like logistic regression and these kinds of things will give you a good start. Then if you can build a better model then great, you should do that, but it's great to have a reference point. 

And then I think the second big lesson I’ve learned from building a lot of projects is that you can get a bit obsessed with the modeling part of the problem because that's the exciting bit when you're doing machine learning but there's this whole ecosystem. Especially if you work in a large company there'll be this whole ecosystem of services and things that are around your application. 

So the lesson there is you should really try to build something end to end that maybe doesn't even have any machine learning at all. But it's the scaffolding upon which you can build the rest of the system because you could spend all this time training an awesome mode, and then you go, oh, oops.

It doesn't integrate with the requirements we have in our application. And then you've wasted all this time. 

### That's a good one! Don't over-engineer. Something I always try to keep in mind. 

**Lewis:** Exactly. And it's a natural thing I think as humans especially if you're nerdy you really want to find the most interesting way to do something and most of the time simple is better.

### If you could go back and do one thing differently at the beginning of your career in machine learning, what would it be? 

**Lewis:** Oh, wow. That's a tough one. Hmm. So, the reason this is a really hard question to answer is that now that I’m working at  Hugging Face, it's the most fulfilling type of work that I've really done in my whole life. And the question is if I changed something when I started out maybe I wouldn't be here, right? 

It's one of those things where it's a tricky one in that sense. I suppose one thing that maybe I would've done slightly differently is when I started out working as a data scientist you tend to develop the skills which are about mapping business problems to software problems or ultimately machine learning problems.

And this is a really great skill to have. But what I later discovered is that my true driving passion is doing open source software development. So probably the thing I would have done differently would have been to start that much earlier. Because at the end of the day most open source is really driven by community members.

So that would have been maybe a way to shortcut my path to doing this full-time. 

### I love the idea of had you done something differently maybe you wouldn't be at Hugging Face. 

**Lewis:** It’s like the butterfly effect movie, right? You go back in time and then you don't have any legs or something.

### Totally. Don't want to mess with a good thing!

**Lewis:** Exactly.

### Rapid Fire Questions:

### Best piece of advice for someone looking to get into AI/Machine Learning?

**Lewis:** Just start. Just start coding. Just start contributing if you want to do open-source. You can always find reasons not to do it but you just have to get your hands dirty.

### What are some of the industries you're most excited to see machine learning applied? 

**Lewis:** As I mentioned before, I think the natural sciences is the area I’m most excited about

This is where I think that's most exciting. If we look at something, say at the industrial side, I guess some of the development of new drugs through machine learning is very exciting. Personally, I'd be really happy if there were advancements in robotics where I could finally have a robot to like fold my laundry because I really hate doing this and it would be nice if like there was an automated way of handling that. 

### Should people be afraid of AI taking over the world?

**Lewis:** Maybe. It’s a tough one because I think we have reasons to think that we may create systems that are quite dangerous in the sense that they could be used to cause a lot of harm. An analogy is perhaps with weapons you can use within the sports like archery and shooting, but you can also use them for war. One big risk is probably if we think about combining these techniques with the military perhaps this leads to some tricky situations.

But, I'm not super worried about the Terminator. I'm more worried about, I don't know, a rogue agent on the financial stock market bankrupting the whole world. 

### That's a good point.

**Lewis:** Sorry, that's a bit dark.

### No, that was great. The next question is a follow-up on your folding laundry robot. When will AI-assisted robots be in homes everywhere?

**Lewis:** Honest answer. I don't know. Everyone, I know who's working on robotics says this is still an extremely difficult task in the sense that robotics hasn't quite experienced the same kind of revolutions that NLP and deep learning have had. But on the other hand, you can see some pretty exciting developments in the last year, especially around the idea of being able to transfer knowledge from a simulation into the real world.

I think there's hope that in my lifetime I will have a laundry-folding robot.

### What have you been interested in lately? It could be a movie, a recipe, a podcast, literally anything. And I'm just curious what that is and how someone interested in that might find it or get started. 

**Lewis:** It's a great question. So for me, I like podcasts in general. It’s my new way of reading books because I have a young baby so I'm just doing chores and listening at the same time. 

One podcast that really stands out recently is actually the [DeepMind podcast](https://www.deepmind.com/the-podcast) produced by Hannah Fry who's a mathematician in the UK and she gives this beautiful journey through not just what Deep Mind does, but more generally, what deep learning and especially reinforcement learning does and how they're impacting the world. Listening to this podcast feels like you're listening to like a BBC documentary because you know the English has such great accents and you feel really inspired because a lot of the work that she discusses in this podcast has a strong overlap with what we do at Hugging Face. You see this much bigger picture of trying to pave the way for a better future.

It resonated strongly. And I just love it because the explanations are super clear and you can share it with your family and your friends and say, “Hey, if you want to know what I'm doing? This can give you a rough idea.”

It gives you a very interesting insight into the Deep Mind researchers and their backstory as well.

### I'm definitely going to give that a listen. [Update: It’s one of my new favorite podcasts. :) Thank you, Lewis!]

### What are some of your favorite Machine Learning papers?

**Lewis:** Depends on how we measure this, but there's [one paper that stands out to me, which is quite an old paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). It’s by the creator of random forests, Leo Breiman. Random forests is a very famous classic machine learning technique that's useful for tabular data that you see in industry and I had to teach random forests at university a year ago.

And I was like, okay, I'll read this paper from the 2000s and see if I understand it. And it's a model of clarity. It's very short, and very clearly explains how the algorithm is implemented. You can basically just take this paper and implement the code very very easily. And that to me was a really nice example of how papers were written in medieval times. 

Whereas nowadays, most papers, have this formulaic approach of, okay, here's an introduction, here's a table with some numbers that get better, and here's like some random related work section. So, I think that's one that like stands out to me a lot.

But another one that's a little bit more recent is [a paper by DeepMind](https://www.nature.com/articles/d41586-021-03593-1) again on using machine learning techniques to prove fundamental theorems like algebraic topology, which is a special branch of abstract mathematics. And at one point in my life, I used to work on these related topics.

So, to me, it's a very exciting, perspective of augmenting the knowledge that a mathematician would have in trying to narrow down the space of theorems that they might have to search for. I think this to me was surprising because a lot of the time I've been quite skeptical that machine learning will lead to this fundamental scientific insight beyond the obvious ones like making predictions.

But this example showed that you can actually be quite creative and help mathematicians find new ideas. 

### What is the meaning of life?

**Lewis:** I think that the honest answer is, I don't know. And probably anyone who does tell you an answer probably is lying. That's a bit sarcastic. I dunno, I guess being a site scientist by training and especially a physicist, you develop this worldview that is very much that there isn't really some sort of deeper meaning to this.

It's very much like the universe is quite random and I suppose the only thing you can take from that beyond being very sad is that you derive your own meaning, right? And most of the time this comes either from the work that you do or from the family or from your friends that you have.

But I think when you find a way to derive your own meaning and discover what you do is actually interesting and meaningful that that's the best part. Life is very up and down, right? At least for me personally, the things that have always been very meaningful are generally in creating things. So, I used to be a musician, so that was a way of creating music for other people and there was great pleasure in doing that. And now I kind of, I guess, create code which is a form of creativity.

### Absolutely. I think that's beautiful, Lewis! Is there anything else you would like to share or mention before we sign off? 

**Lewis:** Maybe [buy my book](https://transformersbook.com/).  

### It is so good!

**Lewis:** [shows book featuring a parrot on the cover] Do you know the story about the parrot?  

### I don't think so.

**Lewis:** So when O’Reilly is telling you “We're going to get our illustrator now to design the cover,” it's a secret, right?

They don't tell you what the logic is or you have no say in the matter. So, basically, the illustrator comes up with an idea and in one of the last chapters of the book we have a section where we basically train a GPT-2 like model on Python code, this was Thom's idea, and he decided to call it code parrot.

I think the idea or the joke he had was that there's a lot of discussion in the community about this paper that Meg Mitchell and others worked on called, ‘Stochastic Parrots’. And the idea was that you have these very powerful language models which seem to exhibit human-like traits in their writing as we discussed earlier but deep down maybe they're just doing some sort of like parrot parenting thing.

You know, if you talk to like a cockatoo it will swear at you or make jokes. That may not be a true measure of intelligence, right? So I think that the illustrator somehow maybe saw that and decided to put a parrot which I think is a perfect metaphor for the book.

And the fact that there are transformers in it.

### Had no idea that that was the way O'Reilly's covers came about. They don't tell you and just pull context from the book and create something?

**Lewis:** It seems like it. I mean, we don't really know the process. I'm just sort of guessing that maybe the illustrator was trying to get an idea and saw a few animals in the book. In one of the chapters we have a discussion about giraffes and zebras and stuff. But yeah I'm happy with the parrot cover.

### I love it. Well, it looks absolutely amazing. A lot of these types of books tend to be quite dry and technical and this one reads almost like a novel mixed with great applicable technical information, which is beautiful.

**Lewis:** Thanks. Yeah, that’s one thing we realized afterward because it was the first time we were writing a book we thought we should be sort of serious, right? But if you sort of know me I'm like never really serious about anything. And in hindsight, we should have been even more silly in the book.

I had to control my humor in various places but maybe there'll be a second edition one day and then we can just inject it with memes.

### Please do, I look forward to that!

**Lewis:** In fact, there is one meme in the book. We tried to sneak this in past the Editor and have the DOGE dog inside the book and we use a special vision transformer to try and classify what this meme is.

### So glad you got that one in there. Well done! Look forward to many more in the next edition. Thank you so much for joining me today. I really appreciate it. Where can our listeners find you online?

**Lewis:** I'm fairly active on Twitter. You can just find me my handle [@_lewtun](https://twitter.com/_lewtun). LinkedIn is a strange place and I'm not really on there very much. And of course, there's [Hugging Face](https://huggingface.co/lewtun), the [Hugging Face Forums](https://discuss.huggingface.co/), and [Discord](https://discuss.huggingface.co/t/join-the-hugging-face-discord/11263).

### Perfect. Thank you so much, Lewis. And I'll chat with you soon!

**Lewis:** See ya, Britney. Bye.

Thank you for listening to Machine Learning Experts!

<a href="https://huggingface.co/support?utm_source=blog&utm_medium=blog&utm_campaign=ml_experts&utm_content=lewis_interview_article"><img src="/blog/assets/60_lewis_tunstall_interview/lewis-cta.png"></a>
