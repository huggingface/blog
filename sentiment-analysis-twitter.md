---
title: "Getting Started with Sentiment Analysis on Twitter"
thumbnail: /blog/assets/85_sentiment_analysis_twitter/thumbnail.png
---

<h1>Getting Started with Sentiment Analysis on Twitter</h1>

<div class="blog-metadata">
    <small>Published July 5, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/sentiment-analysis-twitter.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/federicopascual">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1624043388143-noauth.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>federicopascual</code>
            <span class="fullname">Federico Pascual</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

Sentiment analysis is the automatic process of classifying text data according to their polarity, such as positive, negative and neutral. Companies leverage sentiment analysis of tweets to get a sense of how customers are talking about their products and services, get insights to drive business decisions, and identify product issues and potential PR crises early on.

In this guide, we will cover everything you need to learn to get started with sentiment analysis on Twitter. We'll share a step-by-step process to do sentiment analysis, for both, coders and non-coders. If you are a coder, you'll learn how to use the [Inference API](https://huggingface.co/inference-api), a plug & play machine learning API for doing sentiment analysis of tweets at scale in just a few lines of code. If you don't know how to code, don't worry! We'll also cover how to do sentiment analysis with Zapier, a no-code tool that will enable you to gather tweets, analyze them with the Inference API, and finally send the results to Google Sheets ⚡️

Read along or jump to the section that sparks 🌟 your interest:

1. [What is sentiment analysis?](#what-is-sentiment-analysis)
2. [How to do Twitter sentiment analysis with code?](#how-to-do-twitter-sentiment-analysis-with-code)
3. [How to do Twitter sentiment analysis without coding?](#how-to-do-twitter-sentiment-analysis-without-coding)

Buckle up and enjoy the ride! 🤗

## What is Sentiment Analysis?

Sentiment analysis uses [machine learning](https://en.wikipedia.org/wiki/Machine_learning) to automatically identify how people are talking about a given topic. The most common use of sentiment analysis is detecting the polarity of text data, that is, automatically identifying if a tweet, product review or support ticket is talking positively, negatively, or neutral about something.

As an example, let's check out some tweets mentioning [@Salesforce](https://twitter.com/Salesforce) and see how they would be tagged by a sentiment analysis model:

- *"The more I use @salesforce the more I dislike it. It's slow and full of bugs. There are elements of the UI that look like they haven't been updated since 2006. Current frustration: app exchange pages won't stop refreshing every 10 seconds"* --> This first tweet would be tagged as "Negative".

- *"That’s what I love about @salesforce. That it’s about relationships and about caring about people and it’s not only about business and money. Thanks for caring about #TrailblazerCommunity"* --> In contrast, this tweet would be classified as "Positive".

- *"Coming Home: #Dreamforce Returns to San Francisco for 20th Anniversary. Learn more: http[]()://bit.ly/3AgwO0H via @Salesforce"* --> Lastly, this tweet would be tagged as "Neutral" as it doesn't contain an opinion or polarity.

Up until recently, analyzing tweets mentioning a brand, product or service was a very manual, hard and tedious process; it required someone to manually go over relevant tweets, and read and label them according to their sentiment. As you can imagine, not only this doesn't scale, it is expensive and very time-consuming, but it is also prone to human error.

Luckily, recent advancements in AI allowed companies to use machine learning models for sentiment analysis of tweets that are as good as humans. By using machine learning, companies can analyze tweets in real-time 24/7, do it at scale and analyze thousands of tweets in seconds, and more importantly, get the insights they are looking for when they need them.

Why do sentiment analysis on Twitter? Companies use this for a wide variety of use cases, but the two of the most common use cases are analyzing user feedback and monitoring mentions to detect potential issues early on. 

**Analyze Feedback on Twitter** 

Listening to customers is key for detecting insights on how you can improve your product or service. Although there are multiple sources of feedback, such as surveys or public reviews, Twitter offers raw, unfiltered feedback on what your audience thinks about your offering. 

By analyzing how people talk about your brand on Twitter, you can understand whether they like a new feature you just launched. You can also get a sense if your pricing is clear for your target audience. You can also see what aspects of your offering are the most liked and disliked to make business decisions (e.g. customers loving the simplicity of the user interface but hate how slow customer support is).

**Monitor Twitter Mentions to Detect Issues**

Twitter has become the default way to share a bad customer experience and express frustrations whenever something goes wrong while using a product or service. This is why companies monitor how users mention their brand on Twitter to detect any issues early on. 

By implementing a sentiment analysis model that analyzes incoming mentions in real-time, you can automatically be alerted about sudden spikes of negative mentions. Most times, this is caused is an ongoing situation that needs to be addressed asap (e.g. an app not working because of server outages or a really bad experience with a customer support representative). 

Now that we covered what is sentiment analysis and why it's useful, let's get our hands dirty and actually do sentiment analysis of tweets!💥

## How to do Twitter sentiment analysis with code?

Nowadays, getting started with sentiment analysis on Twitter is quite easy and straightforward 🙌 

With a few lines of code, you can automatically get tweets, run sentiment analysis and visualize the results. And you can learn how to do all these things in just a few minutes! 

In this section, we'll show you how to do it with a cool little project: we'll do sentiment analysis of tweets mentioning [Notion](https://twitter.com/notionhq)! 

First, you'll use [Tweepy](https://www.tweepy.org/), an open source Python library to get tweets mentioning @NotionHQ using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api). Then you'll use the [Inference API](https://huggingface.co/inference-api) for doing sentiment analysis. Once you get the sentiment analysis results, you will create some charts to visualize the results and detect some interesting insights.

You can use this [Google Colab notebook](https://colab.research.google.com/drive/1R92sbqKMI0QivJhHOp1T03UDaPUhhr6x?usp=sharing) to follow this tutorial. 

Let's get started with it! 💪

1. Install Dependencies

As a first step, you'll need to install the required dependencies. You'll use [Tweepy](https://www.tweepy.org/) for gathering tweets, [Matplotlib](https://matplotlib.org/) for building some charts and [WordCloud](https://amueller.github.io/word_cloud/) for building a visualization with the most common keywords:

```python
!pip install -q transformers tweepy matplotlib wordcloud
```

2. Setting up Twitter credentials

Then, you need to set up the [Twitter API credentials](https://developer.twitter.com/en/docs/twitter-api) so you can authenticate with Twitter and then gather tweets automatically using their API:

```python
import tweepy
 
# Add Twitter API key and secret
consumer_key = "XXXXXX"
consumer_secret = "XXXXXX"
 
# Handling authentication with Twitter
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
 
# Create a wrapper for the Twitter API
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
```

3. Search for tweets using Tweepy

Now you are ready to start collecting data from Twitter! 🎉 You will use [Tweepy Cursor](https://docs.tweepy.org/en/v3.5.0/cursor_tutorial.html) to automatically collect 1,000 tweets mentioning Notion: 

```python
# Helper function for handling pagination in our search and handle rate limits
def limit_handled(cursor):
   while True:
       try:
           yield cursor.next()
       except tweepy.RateLimitError:
           print('Reached rate limite. Sleeping for >15 minutes')
           time.sleep(15 * 61)
       except StopIteration:
           break
 
# Define the term you will be using for searching tweets
query = '@NotionHQ'
query = query + ' -filter:retweets'
 
# Define how many tweets to get from the Twitter API
count = 1000
 
# Search for tweets using Tweepy
search = limit_handled(tweepy.Cursor(api.search,
                       q=query,
                       tweet_mode='extended',
                       lang='en',
                       result_type="recent").items(count))
 
# Process the results from the search using Tweepy
tweets = []
for result in search:
 tweet_content = result.full_text
 tweets.append(tweet_content) # Only saving the tweet content.
```

4. Analyzing tweets with sentiment analysis

Now that you have data, you are ready to analyze the tweets with sentiment analysis! 💥

You will be using [Inference API](https://huggingface.co/inference-api), an easy-to-use API for integrating machine learning models via simple API calls. With the Inference API, you can use state-of-the-art models for sentiment analysis without the hassle of building infrastructure for machine learning or dealing with model scalability. You can serve the latest (and greatest!) open source models for sentiment analysis while staying out of MLOps. 🤩

For using the Inference API, first you will need to define your `model id` and your `Hugging Face API Token`: 

- The `model ID` is to specify which model you want to use for making predictions. Hugging Face has more than [400 models for sentiment analysis in multiple languages](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment), including various models specifically fine-tuned for sentiment analysis of tweets. For this particular tutorial, you will use [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a sentiment analysis model trained on ≈124 million tweets and fine-tuned for sentiment analysis. 

- You'll also need to specify your `Hugging Face token`; you can get one for free by signing up [here](https://huggingface.co/join) and then copying your token on this [page](https://huggingface.co/settings/tokens).


```python
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = "XXXXXX" 
```

Next, you will create the API call using the `model id` and `hf_token`:

```python
API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}
 
def analysis(payload):
 response = requests.post(API_URL, headers=headers, json=payload)
 while "is currently loading" in str(response.json()): # retry request while model loads
   print(response.json())
   time.sleep(15)
   response = requests.post(API_URL, headers=headers, json=payload)
 return response.json()
```

Now, you are ready to do sentiment analysis on each tweet. 🔥🔥🔥

```python
tweets_analysis = []
for tweet in tweets:
   try:
     sentiment_result = analysis(tweet)[0]
     top_sentiment = max(sentiment_result, key=lambda x: x['score']) # Get the sentiment with the higher score    
     tweets_analysis.append({'tweet': tweet, 'sentiment': top_sentiment['label']})
 
   except Exception as e:
     print(e)
```
5. Explore the results of sentiment analysis

Wondering if people on Twitter are talking positively or negatively about Notion? Or what do users discuss when talking positively or negatively about Notion? We'll use some data visualization to explore the results of the sentiment analysis and find out!

First, let's see examples of tweets that were labeled for each sentiment to get a sense of the different polarities of these tweets:

```python
import pandas as pd
 
# Load the data in a dataframe
pd.set_option('max_colwidth', None)
pd.set_option('display.width', 3000)
df = pd.DataFrame(tweets_analysis)
 
# Show a tweet for each sentiment
display(df[df["sentiment"] == 'Positive'].head(1))
display(df[df["sentiment"] == 'Neutral'].head(1))
display(df[df["sentiment"] == 'Negative'].head(1))
```

Results:

```
@thenotionbar @hypefury @NotionHQ That’s genuinely smart. So basically you’ve setup your posting queue to by a recurrent recycling of top content that runs 100% automatic? Sentiment: Positive

@itskeeplearning @NotionHQ How you've linked gallery cards? Sentiment: Neutral

@NotionHQ Running into an issue here recently were content is not showing on on web but still in the app. This happens for all of our pages. https://t.co/3J3AnGzDau. Sentiment: Negative
```

Next, you'll count the number of tweets that were tagged as positive, negative and neutral:

```python
sentiment_counts = df.groupby(['sentiment']).size()
print(sentiment_counts)
```

Remarkably, most of the tweets about Notion are positive:
```
sentiment
Negative     82
Neutral     420
Positive    498
```

Then, let's create a pie chart to visualize each sentiment in relative terms:

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
sentiment_counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=270, fontsize=12, label="")
```

It's cool to see that 50% of all tweets are positive and only 8.2% are negative:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/sentiment-pie.png"></medium-zoom>
  <figcaption>Sentiment analysis results of tweets mentioning Notion</figcaption>
</figure>

As a last step, let's create some wordclouds to see which words are the most used for each sentiment:
 
```python
from wordcloud import WordCloud
from wordcloud import STOPWORDS
 
# Wordcloud with positive tweets
positive_tweets = df['tweet'][df["sentiment"] == 'Positive']
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
positive_wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", stopwords = stop_words).generate(str(positive_tweets))
plt.figure()
plt.title("Positive Tweets - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
 
# Wordcloud with negative tweets
negative_tweets = df['tweet'][df["sentiment"] == 'Negative']
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
negative_wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", stopwords = stop_words).generate(str(negative_tweets))
plt.figure()
plt.title("Negative Tweets - Wordcloud")
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

Curiously, some of the words that stand out from the positive tweets include "notes", "cron", and "paid":

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/positive-tweets.png"></medium-zoom>
  <figcaption>Word cloud for positive tweets</figcaption>
</figure>

In contrast, "figma", "enterprise" and "account" are some of the most used words from the negatives tweets:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/negative-tweets.png"></medium-zoom>
  <figcaption>Word cloud for negative tweets</figcaption>
</figure>

That was fun, right? 

With just a few lines of code, you were able to automatically gather tweets mentioning Notion using Tweepy, analyze them with a sentiment analysis model using the [Inference API](https://huggingface.co/inference-api), and finally create some visualizations to analyze the results. 💥 

Are you interested in doing more? As a next step, you could use a second [text classifier](https://huggingface.co/tasks/text-classification) to classify each tweet by their theme or topic. This way, each tweet will be labeled with both sentiment and topic, and you can get more granular insights (e.g. are users praising how easy to use is Notion but are complaining about their pricing or customer support?).

## How to do Twitter sentiment analysis without coding?

To get started with sentiment analysis, you don't need to be a developer or know how to code. 🤯 

There are some amazing no-code solutions that will enable you to easily do sentiment analysis in just a few minutes. 

In this section, you will use [Zapier](https://zapier.com/), a no-code tool that enables users to connect 5,000+ apps with an easy to use user interface. You will create a [Zap](https://zapier.com/help/create/basics/create-zaps), that is triggered whenever someone mentions Notion on Twitter. Then the Zap will use the [Inference API](https://huggingface.co/inference-api) to analyze the tweet with a sentiment analysis model and finally it will save the results on Google Sheets:

1. Step 1 (trigger): Getting the tweets.
2. Step 2: Analyze tweets with sentiment analysis.
3. Step 3: Save the results on Google Sheets.

No worries, it won't take much time; in under 10 minutes, you'll create and activate the zap, and will start seeing the sentiment analysis results pop up in Google Sheets.

Let's get started! 🚀

### Step 1: Getting the Tweets

To get started, you'll need to [create a Zap](https://zapier.com/webintent/create-zap), and configure the first step of your Zap, also called the *"Trigger"* step. In your case, you will need to set it up so that it triggers the Zap whenever someone mentions Notion on Twitter. To set it up, follow the following steps: 

- First select "Twitter" and select "Search mention" as event on "Choose app & event". 
- Then connect your Twitter account to Zapier.
- Set up the trigger by specifying "NotionHQ" as the search term for this trigger.
- Finally test the trigger to make sure it gather tweets and runs correctly.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/zapier-getting-tweets-cropped-cut-optimized.gif"></medium-zoom>
  <figcaption>Step 1 on the Zap</figcaption>
</figure>

### Step 2: Analyze Tweets with Sentiment Analysis

Now that your Zap can gather tweets mentioning Notion, let's add a second step to do the sentiment analysis. 🤗 

You will be using [Inference API](https://huggingface.co/inference-api), an easy-to-use API for integrating machine learning models. For using the Inference API, you will need to define your "model id" and your "Hugging Face API Token": 

- The `model ID` is to tell the Inference API which model you want to use for making predictions. For this guide, you will use [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a sentiment analysis model trained on ≈124 million tweets and fine-tuned for sentiment analysis. You can explore the more than [400 models for sentiment analysis available on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment) in case you want to use a different model (e.g. doing sentiment analysis on a different language).

- You'll also need to specify your `Hugging Face token`; you can get one for free by signing up [here](https://huggingface.co/join) and then copying your token on this [page](https://huggingface.co/settings/tokens).

Once you have your model ID and your Hugging Face token ID, go back to your Zap and follow these instructions to set up the second step of the zap:

1. First select "Code by Zapier" and "Run python" in "Choose app and event".
2. On "Set up action", you will need to first add the tweet "full text" as "input_data". Then you will need to add these [28 lines of python code](https://gist.github.com/feconroses/0e064f463b9a0227ba73195f6376c8ed) in the "Code" section. This code will allow the Zap to call the Inference API and make the predictions with sentiment analysis. Before adding this code to your zap, please make sure that you do the following:
    - Change line 5 and add your Hugging Face token, that is, instead of `hf_token = "ADD_YOUR_HUGGING_FACE_TOKEN_HERE"`, you will need to change it to something like`hf_token = "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"`
    - If you want to use a different sentiment analysis model, you will need to change line 4 and specify the id of the new model here. For example, instead of using the default model, you could use [this model](https://huggingface.co/finiteautomata/beto-sentiment-analysis?text=Te+quiero.+Te+amo.) to do sentiment analysis on tweets in Spanish by changing this line `model = "cardiffnlp/twitter-roberta-base-sentiment-latest"` to `model = "finiteautomata/beto-sentiment-analysis"`.
3. Finally, test this step to make sure it makes predictions and runs correctly.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/zapier-analyze-tweets-cropped-cut-optimized.gif"></medium-zoom>
  <figcaption>Step 2 on the Zap</figcaption>
</figure>

### Step 3: Save the results on Google Sheets

As the last step to your Zap, you will save the results of the sentiment analysis on a spreadsheet on Google Sheets and visualize the results. 📊

First, [create a new spreadsheet on Google Sheets](https://docs.google.com/spreadsheets/u/0/create), and define the following columns: 
- **Tweet**: this column will contain the text of the tweet. 
- **Sentiment**: will have the label of the sentiment analysis results (e.g. positive, negative and neutral).
- **Score**: will store the value that reflects how confident the model is with its prediction.
- **Date**: will contain the date of the tweet (which can be handy for creating graphs and charts over time).

Then, follow these instructions to configure this last step:
1. Select Google Sheets as an app, and "Create Spreadsheet Row" as the event in "Choose app & Event".
2. Then connect your Google Sheets account to Zapier.
3. Next, you'll need to set up the action. First, you'll need to specify the Google Drive value (e.g. My Drive), then select the spreadsheet, and finally the worksheet where you want Zapier to automatically write new rows. Once you are done with this, you will need to map each column on the spreadsheet with the values you want to use when your zap automatically writes a new row on your file. If you have created the columns we suggested before, this will look like the following (column → value):
    - Tweet → Full Text (value from the step 1 of the zap)
    - Sentiment → Sentiment Label (value from step 2)
    - Sentiment Score → Sentiment Score (value from step 2)
    - Date → Created At (value from step 1)
4. Finally, test this last step to make sure it can add a new row to your spreadsheet. After confirming it's working, you can delete this row on your spreadsheet.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/zapier-add-to-google-sheets-cropped-cut.gif"></medium-zoom>
  <figcaption>Step 3 on the Zap</figcaption>
</figure>

### 4. Turn on your Zap

At this point, you have completed all the steps of your zap! 🔥 

Now, you just need to turn it on so it can start gathering tweets, analyzing them with sentiment analysis, and store the results on Google Sheets. ⚡️

To turn it on, just click on "Publish" button at the bottom of your screen:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/zap-turn-on-cut-optimized.gif"></medium-zoom>
  <figcaption>Turning on the Zap</figcaption>
</figure>

After a few minutes, you will see how your spreadsheet starts populating with tweets and the results of sentiment analysis. You can also create a graph that can be updated in real-time as tweets come in:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Sentiment analysis results of tweets mentioning Notion" src="assets/85_sentiment_analysis_twitter/google-sheets-results-cropped-cut.gif"></medium-zoom>
  <figcaption>Tweets popping up on Google Sheets</figcaption>
</figure>

Super cool, right? 🚀

## Wrap up

Twitter is the public town hall where people share their thoughts about all kinds of topics. From people talking about politics, sports or tech, users sharing their feedback about a new shiny app, or passengers complaining to an Airline about a canceled flight, the amount of data on Twitter is massive. Sentiment analysis allows making sense of all that data in real-time to uncover insights that can drive business decisions.

Luckily, tools like the [Inference API](https://huggingface.co/inference-api) makes it super easy to get started with sentiment analysis on Twitter. No matter if you know or don't know how to code and/or you don't have experience with machine learning, in a few minutes, you can set up a process that can gather tweets in real-time, analyze them with a state-of-the-art model for sentiment analysis, and explore the results with some cool visualizations. 🔥🔥🔥

If you have questions, you can ask them in the [Hugging Face forum](https://discuss.huggingface.co/) so the Hugging Face community can help you out and others can benefit from seeing the discussion. You can also join our [Discord](https://discord.gg/YRAq8fMnUG) server to talk with us and the entire Hugging Face community.
