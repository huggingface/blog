---
title: How to use different decoding methods for open-ended language generation with transformers
thumbnail: https://huggingface.co/blog/assets/02_how-to-generate/thumbnail.png
---

# How to use different decoding methods for open-ended language generation with transformers

<div class="blog-metadata">
    <small>Published March 17, 2020.</small>
    <a target="_blank" class="btn-readme" href="https://github.com/huggingface/blog/blob/master/how-to-generate.md">
        <img src="/front/assets/icon-github.svg">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/patrickvonplaten">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1584435275418-5dfcb1aada6d0311fd3d5448.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>patrickvonplaten</code>
            <span class="fullname">Patrick von Platen</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### **Introduction**

In recent years, there has been an increasing interest in open-ended
language generation thanks to the rise of large transformer-based
language models trained on millions of webpages, such as OpenAI's famous
[GPT2 model](https://openai.com/blog/better-language-models/). The
results on conditioned open-ended language generation are impressive,
e.g. [GPT2 on
unicorns](https://openai.com/blog/better-language-models/#samples),
[XLNet](https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e),
[Controlled language with
CTRL](https://blog.einstein.ai/introducing-a-conditional-transformer-language-model-for-controllable-generation/).
Besides the improved transformer architecture and massive unsupervised
training data, **better decoding methods** have also played an important
role.

This blog post gives a brief overview of different decoding strategies
and more importantly shows how *you* can implement them with very little
effort using the popular `transformers` library\!

All of the following functionalities can be used for **auto-regressive**
language generation ([here](http://jalammar.github.io/illustrated-gpt2/)
a refresher). In short, *auto-regressive* language generation is based
on the assumption that the probability distribution of a word sequence
can be decomposed into the product of conditional next word
distributions:

$$ P(w_{1:T} | W_0 ) = \prod_{t=1}^T P(w_{t} | w_{1: t-1}, W_0) \text{ ,with }  w_{1: 0} = \emptyset, $$

and \\(W_0\\) being the initial *context* word sequence. The length \\(T\\)
of the word sequence is usually determined *on-the-fly* and corresponds
to the timestep \\(t=T\\) the EOS token is generated from
\\(P(w_{t} | w_{1: t-1}, W_{0})\\).

Auto-regressive language generation is now available for `GPT2`,
`XLNet`, `OpenAi-GPT`, `CTRL`, `TransfoXL`, `XLM`, `Bart`, `T5` in both
PyTorch and Tensorflow \>= 2.0\!

We will give a tour of the currently most prominent decoding methods,
mainly *Greedy search*, *Beam search*, *Top-K sampling* and *Top-p
sampling*.


<div class="cell markdown" data-colab_type="text" id="Si4GyYhOQMzi">

Let's quickly install transformers and load the model. We will use GPT2
in Tensorflow 2.1 for demonstration, but the API is 1-to-1 the same for
PyTorch.

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="XbzZ_IVTtoQe">

``` python
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q tensorflow==2.1
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="ue2kOQhXTAMU">

``` python
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```

</div>

<div class="cell markdown" data-colab_type="text" id="a8Y7cgu9ohXP">

### **Greedy Search**

Greedy search simply selects the word with the highest probability as
its next word: \\(w_t = argmax_{w}P(w | w_{1:t-1})\\) at each timestep
\\(t\\). The following sketch shows greedy search.


<img src="/blog/assets/02_how-to-generate/greedy_search.png" alt="greedy search" style="margin: auto; display: block;">

Starting from the word \\(\text{"The"}\\), the algorithm greedily chooses
the next word of highest probability \\(\text{"nice"}\\) and so on, so
that the final generated word sequence is
\\(\text{"The", "nice", "woman"}\\) having an overall probability of
\\(0.5 \times 0.4 = 0.2\\).

In the following we will generate word sequences using GPT2 on the
context
\\((\text{"I", "enjoy", "walking", "with", "my", "cute", "dog"})\\). Let's
see how greedy search can be used in `transformers` by setting
`do_sample=False` when calling the `generate()` method:

</div>

<div class="cell code" data-execution_count="4" data-colab="{&quot;height&quot;:122,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="OWLd_J6lXz_t" data-outputId="3b9dfd1e-21e6-44f4-f27f-8e975010f9af">

``` python
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, do_sample=False, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll ever be able to walk with my dog.
    
    I'm not sure if I'll

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="BBn1ePmJvhrl">

Alright\! We have generated our first short text with GPT2 😊. The
generated words following the context are reasonable, but the model
quickly starts repeating itself\! This is a very common problem in
language generation in general and seems to be even more so in greedy
and beam search - check out [Vijayakumar et
al., 2016](https://arxiv.org/abs/1610.02424) and [Shao et
al., 2017](https://arxiv.org/abs/1701.03185).

The major drawback of greedy search though is that it misses high
probability words hidden behind a low probability word as can be seen in
our sketch above:

The word \\(\text{"has"}\\) with its high conditional probability of
\\(0.9\\) is hidden behind the word \\(\text{"dog"}\\), which has only the
second-highest conditional probability, so that greedy search misses the
word sequence \\(\text{"The"}, \text{"dog"}, \text{"has"}\\).

Thankfully, we have beam search to alleviate this problem\!

</div>

<div class="cell markdown" data-colab_type="text" id="g8DnXZ1WiuNd">

### **Beam search**

Beam search reduces the risk of missing hidden high probability word
sequences by keeping the most likely `num_beams` of hypotheses at each
time step and eventually choosing the hypothesis that has the overall
highest probability. Let's illustrate with `num_beams=2`:

<img src="/blog/assets/02_how-to-generate/beam_search.png" alt="beam search" style="margin: auto; display: block;">

At time step \\(1\\), besides the most likely hypothesis
\\(\text{"The", "woman"}\\), beam search also keeps track of the second
most likely one \\(\text{"The", "dog"}\\). At time step \\(2\\), beam search
finds that the word sequence \\(\text{"The", "dog", "has"}\\) has with
\\(0.36\\) a higher probability than \\(\text{"The", "nice", "woman"}\\),
which has \\(0.2\\). Great, it has found the most likely word sequence in
our toy example\!

Beam search will always find an output sequence with higher probability
than greedy search, but is not guaranteed to find the most likely
output.

Let's see how beam search can be used in `transformers`. We set
`num_beams > 1` and `early_stopping=True` so that generation is finished
when all beam hypotheses reached the EOS token.

</div>

<div class="cell code" data-execution_count="5" data-colab="{&quot;height&quot;:102,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="R1R5kx30Ynej" data-outputId="574f068b-f418-48b5-8334-8451d2221032">

``` python
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids, 
    do_sample=False, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
    
    I'm not sure if I'll ever be able to walk with him again. I'm not sure if I'll

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="AZ6xs-KLi9jT">

While the result is arguably more fluent, the output still includes
repetitions of the same word sequences.  
A simple remedy is to introduce *n-grams* (*a.k.a* word sequences of
\\(n\\) words) penalties as introduced by [Paulus et al.
(2017)](https://arxiv.org/abs/1705.04304) and [Klein et al.
(2017)](https://arxiv.org/abs/1701.02810). The most common *n-grams*
penalty makes sure that no *n-gram* appears twice by manually setting
the probability of next words that could create an already seen *n-gram*
to \\(0\\).

Let's try it out by setting `no_repeat_ngram_size=2` so that no *2-gram*
appears twice:

</div>

<div class="cell code" data-execution_count="6" data-colab="{&quot;height&quot;:102,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="jy3iVJgfnkMi" data-outputId="4d3e6511-711a-4594-a715-aaeb6e48e1a9">

``` python
# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids, 
    do_sample=False, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
    
    I've been thinking about this for a while now, and I think it's time for me to take a break

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="nxsksOGDpmA0">

Nice, that looks much better\! We can see that the repetition does not
appear anymore. Nevertheless, *n-gram* penalties have to be used with
care. An article generated about the city *New York* should not use a
*2-gram* penalty or otherwise, the name of the city would only appear
once in the whole text\!

Another important feature about beam search is that we can compare the
top beams after generation and choose the generated beam that fits our
purpose best.

In `transformers`, we simply set the parameter `num_return_sequences` to
the number of highest scoring beams that should be returned. Make sure
though that `num_return_sequences <= num_beams`\!

</div>

<div class="cell code" data-execution_count="7" data-colab="{&quot;height&quot;:306,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="5ClO3VphqGp6" data-outputId="2296891c-024f-4fd2-9071-bff7c11a3e04">

``` python
# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    do_sample=False, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    0: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
    
    I've been thinking about this for a while now, and I think it's time for me to take a break
    1: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
    
    I've been thinking about this for a while now, and I think it's time for me to get back to
    2: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with her again.
    
    I've been thinking about this for a while now, and I think it's time for me to take a break
    3: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with her again.
    
    I've been thinking about this for a while now, and I think it's time for me to get back to
    4: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.
    
    I've been thinking about this for a while now, and I think it's time for me to take a step

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="HhLKyfdbsjXc">

As can be seen, the five beam hypotheses are only marginally different
to each other - which should not be too surprising when using only 5
beams.

In open-ended generation, a couple of reasons have recently been brought
forward why beam search might not be the best possible option:

  - Beam search can work very well in tasks where the length of the
    desired generation is more or less predictable as in machine
    translation or summarization - see [Murray et al.
    (2018)](https://arxiv.org/abs/1808.10006) and [Yang et al.
    (2018)](https://arxiv.org/abs/1808.09582). But this is not the case
    for open-ended generation where the desired output length can vary
    greatly, e.g. dialog and story generation.

  - We have seen that beam search heavily suffers from repetitive
    generation. This is especially hard to control with *n-gram*- or
    other penalties in story generation since finding a good trade-off
    between forced "no-repetition" and repeating cycles of identical
    *n-grams* requires a lot of finetuning.

  - As argued in [Ari Holtzman et al.
    (2019)](https://arxiv.org/abs/1904.09751), high quality human
    language does not follow a distribution of high probability next
    words. In other words, as humans, we want generated text to surprise
    us and not to be boring/predictable. The authors show this nicely by
    plotting the probability, a model would give to human text vs. what
    beam search does.

![alt
text](https://blog.fastforwardlabs.com/images/2019/05/Screen_Shot_2019_05_08_at_3_06_36_PM-1557342561886.png)

So let's stop being boring and introduce some randomness 🤪.

</div>

<div class="cell markdown" data-colab_type="text" id="XbbIyK84wHq6">

### **Sampling**

In its most basic form, sampling means randomly picking the next word
\\(w_t\\) according to its conditional probability distribution:

$$ w_t \sim P(w|w_{1:t-1}) $$

Taking the example from above, the following graphic visualizes language
generation when sampling.

<img src="/blog/assets/02_how-to-generate/sampling_search.png" alt="sampling search" style="margin: auto; display: block;">

It becomes obvious that language generation using sampling is not
*deterministic* anymore. The word \\(\text{"car"}\\) is sampled from the
conditioned probability distribution \\(P(w | \text{"The"})\\), followed
by sampling \\(\text{"drives"}\\) from
\\(P(w | \text{"The"}, \text{"car"})\\).

In `transformers`, we set `do_sample=True` and deactivate *Top-K*
sampling (more on this later) via `top_k=0`. In the following, we will
fix `random_seed=0` for illustration purposes. Feel free to change the
`random_seed` to play around with the model.

</div>

<div class="cell code" data-execution_count="8" data-colab="{&quot;height&quot;:136,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="aRAz4D-Ks0_4" data-outputId="1b78d191-15f6-4cbe-e2b1-23c77366fc21">

``` python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog. He just gave me a whole new hand sense."
    
    But it seems that the dogs have learned a lot from teasing at the local batte harness once they take on the outside.
    
    "I take

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="mQHuo911wfT-">

Interesting\! The text seems alright - but when taking a closer look, it
is not very coherent. the *3-grams* *new hand sense* and *local batte
harness* are very weird and don't sound like they were written by a
human. That is the big problem when sampling word sequences: The models
often generate incoherent gibberish, *cf.* [Ari Holtzman et al.
(2019)](https://arxiv.org/abs/1904.09751).

A trick is to make the distribution \\(P(w|w_{1:t-1})\\) sharper
(increasing the likelihood of high probability words and decreasing the
likelihood of low probability words) by lowering the so-called
`temperature` of the
[softmax](https://en.wikipedia.org/wiki/Softmax_function#Smooth_arg_max).

An illustration of applying temperature to our example from above could
look as follows.

<img src="/blog/assets/02_how-to-generate/sampling_search_with_temp.png" alt="sampling temp search" style="margin: auto; display: block;">

The conditional next word distribution of step \\(t=1\\) becomes much
sharper leaving almost no chance for word \\(\text{"car"}\\) to be
selected.

Let's see how we can cool down the distribution in the library by
setting `temperature=0.7`:

</div>

<div class="cell code" data-execution_count="9" data-colab="{&quot;height&quot;:88,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="WgJredc-0j0Z" data-outputId="a4e79355-8e3c-4788-fa21-c4e28bf61c5b">

``` python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog, but I don't like to be at home too much. I also find it a bit weird when I'm out shopping. I am always away from my house a lot, but I do have a few friends

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="kzGuu24hZZnq">

OK. There are less weird n-grams and the output is a bit more coherent
now\! While applying temperature can make a distribution less random, in
its limit, when setting `temperature` $ \\to 0$, temperature scaled
sampling becomes equal to greedy decoding and will suffer from the same
problems as before.

</div>

<div class="cell markdown" data-colab_type="text" id="binNTroyzQBu">

### **Top-K Sampling**

[Fan et. al (2018)](https://arxiv.org/pdf/1805.04833.pdf) introduced a
simple, but very powerful sampling scheme, called ***Top-K*** sampling.
In *Top-K* sampling, the *K* most likely next words are filtered and the
probability mass is redistributed among only those *K* next words. GPT2
adopted this sampling scheme, which was one of the reasons for its
success in story generation.

We extend the range of words used for both sampling steps in the example
above from 3 words to 10 words to better illustrate *Top-K* sampling.

<img src="/blog/assets/02_how-to-generate/top_k_sampling.png" alt="Top K sampling" style="margin: auto; display: block;">

Having set \\(K = 6\\), in both sampling steps we limit our sampling pool
to 6 words. While the 6 most likely words, defined as
\\(V_{\text{top-K}}\\) encompass only *ca.* two-thirds of the whole
probability mass in the first step, it includes almost all of the
probability mass in the second step. Nevertheless, we see that it
successfully eliminates the rather weird candidates
\\(\text{"not", "the", "small", "told"}\\) in the second sampling step.

Let's see how *Top-K* can be used in the library by setting `top_k=50`:

</div>

<div class="cell code" data-execution_count="11" data-colab="{&quot;height&quot;:156,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="HBtDOdD0wx3l" data-outputId="cfc97fac-0956-42ee-a6e5-cad14fc942d3">

``` python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog. It's so good to have an environment where your dog is available to share with you and we'll be taking care of you.
    
    We hope you'll find this story interesting!
    
    I am from

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="Y77H5m4ZmhEX">

Not bad at all\! The text is arguably the most *human-sounding* text so
far. One concern though with *Top-K* sampling is that it does not
dynamically adapt the number of words that are filtered from the next
word probability distribution \\(P(w|w_{1:t-1})\\). This can be
problematic as some words might be sampled from a very sharp
distribution (distribution on the right in the graph above), whereas
others from a much more flat distribution (distribution on the left in
the graph above).

In step \\(t=1\\), *Top-K* eliminates the possibility to sample
\\(\text{"people", "big", "house", "cat"}\\), which seem like reasonable
candidates. On the other hand, in step \\(t=2\\) the method includes the
arguably ill-fitted words \\(\text{"down", "a"}\\) in the sample pool of
words. Thus, limiting the sample pool to a fixed size *K* could endanger
the model to produce gibberish for sharp distributions and limit the
model's creativity for flat distribution. This intuition led [Ari
Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751) to create
***Top-p***- or ***nucleus***-sampling.

</div>

<div class="cell markdown" data-colab_type="text" id="ki9LAaexzV3H">

### **Top-p (nucleus) sampling**

Instead of sampling only from the most likely *K* words, in *Top-p*
sampling chooses from the smallest possible set of words whose
cumulative probability exceeds the probability *p*. The probability mass
is then redistributed among this set of words. This way, the size of the
set of words (*a.k.a* the number of words in the set) can dynamically
increase and decrease according to the next word's probability
distribution. Ok, that was very wordy, let's visualize.

<img src="/blog/assets/02_how-to-generate/top_p_sampling.png" alt="Top p sampling" style="margin: auto; display: block;">

Having set \\(p=0.92\\), *Top-p* sampling picks the *minimum* number of
words to exceed together \\(p=92\%\\) of the probability mass, defined as
\\(V_{\text{top-p}}\\). In the first example, this included the 9 most
likely words, whereas it only has to pick the top 3 words in the second
example to exceed 92%. Quite simple actually\! It can be seen that it
keeps a wide range of words where the next word is arguably less
predictable, *e.g.* \\(P(w | \text{"The"})\\), and only a few words when
the next word seems more predictable, *e.g.*
\\(P(w | \text{"The", "car"})\\).

Alright, time to check it out in `transformers`\! We activate *Top-p*
sampling by setting `0 < top_p < 1`:

</div>

<div class="cell code" data-execution_count="10" data-colab="{&quot;height&quot;:170,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="EvwIc7YAx77F" data-outputId="57e2b785-5dcb-4e06-9869-078b758b6a82">

``` python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    I enjoy walking with my cute dog. He will never be the same. I watch him play.
    
    
    Guys, my dog needs a name. Especially if he is found with wings.
    
    
    What was that? I had a lot of

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="tn-8gLaR4lat">

Great, that sounds like it could have been written by a human. Well,
maybe not quite yet.

While in theory, *Top-p* seems more elegant than *Top-K*, both methods
work well in practice. *Top-p* can also be used in combination with
*Top-K*, which can avoid very low ranked words while allowing for some
dynamic selection.

Finally, to get multiple independently sampled outputs, we can *again*
set the parameter `num_return_sequences > 1`:

</div>

<div class="cell code" data-execution_count="12" data-colab="{&quot;height&quot;:190,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="3kY8P9VG8Gi9" data-outputId="6103051e-1681-4ab9-a9c1-1fad437c299d">

``` python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```

<div class="output stream stdout">

``` 
Output:
----------------------------------------------------------------------------------------------------
0: I enjoy walking with my cute dog. It's so good to have the chance to walk with a dog. But I have this problem with the dog and how he's always looking at us and always trying to make me see that I can do something
1: I enjoy walking with my cute dog, she loves taking trips to different places on the planet, even in the desert! The world isn't big enough for us to travel by the bus with our beloved pup, but that's where I find my love
2: I enjoy walking with my cute dog and playing with our kids," said David J. Smith, director of the Humane Society of the US.

"So as a result, I've got more work in my time," he said.


```

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="-vRPfMl88rk0">

Cool, now you should have all the tools to let your model write your
stories with `transformers`\!

</div>

<div class="cell markdown" data-colab_type="text" id="NsWd7e98Vcs3">

### **Conclusion**

As *ad-hoc* decoding methods, *top-p* and *top-K* sampling seem to
produce more fluent text than traditional *greedy* - and *beam* search
on open-ended language generation. Recently, there has been more
evidence though that the apparent flaws of *greedy* and *beam* search -
mainly generating repetitive word sequences - are caused by the model
(especially the way the model is trained), rather than the decoding
method, *cf.* [Welleck et al.
(2019)](https://arxiv.org/pdf/1908.04319.pdf). Also, as demonstrated in
[Welleck et al. (2020)](https://arxiv.org/abs/2002.02492), it looks as
*top-K* and *top-p* sampling also suffer from generating repetitive word
sequences.

In [Welleck et al. (2019)](https://arxiv.org/pdf/1908.04319.pdf), the
authors show that according to human evaluations, *beam* search can
generate more fluent text than *Top-p* sampling, when adapting the
model's training objective.

Open-ended language generation is a rapidly evolving field of research
and as it is often the case there is no one-size-fits-all method here,
so one has to see what works best in one's specific use case.

Good thing, that *you* can try out all the different decoding methods in
`transfomers` 🤗.

That was a short introduction on how to use different decoding methods
in `transformers` and recent trends in open-ended language generation.

Feedback and questions are very welcome on the [Github
repository](https://github.com/huggingface/transformers).

For more fun generating stories, please take a look at [Writing with Transformers](https://transformer.huggingface.co/)

Thanks to everybody, who has contributed to the blog post: Alexander Rush, Julien Chaumand, Thomas Wolf, Victor Sanh, Sam Shleifer, Clément Delangue, Yacine Jernite, Oliver Åstrand and John de Wasseige.


</div>

<div class="cell markdown" data-colab_type="text" id="w4CYi91h11yd">

### **Appendix**

There are a couple of additional parameters for the `generate` method
that were not mentioned above. We will explain them here briefly\!

  - `min_length` can be used to force the model to not produce an EOS
    token (= not finish the sentence) before `min_length` is reached.
    This is used quite frequently in summarization, but can be useful in
    general if the user wants to have longer outputs.

  - `repetition_penalty` can be used to penalize words that were already
    generated or belong to the context. It was first introduced by
    [Kesker et al. (2019)](https://arxiv.org/abs/1909.05858) and is also
    used in the training objective in [Welleck et al.
    (2019)](https://arxiv.org/pdf/1908.04319.pdf). It can be quite
    effective at preventing repetitions, but seems to be very sensitive
    to different models and use cases, *e.g.* see this
    [discussion](https://github.com/huggingface/transformers/pull/2303)
    on Github.

  - `attention_mask` can be used to mask padded tokens

  - `pad_token_id`, `bos_token_id`, `eos_token_id`: If the model does
    not have those tokens by default, the user can manually choose other
    token ids to represent them.

For more information please also look into the `generate` function
[docstring](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.TFPreTrainedModel.generate).

</div>
