---
layout: post
title: "Latent DiReddit Allocation"
subtitle: "Skull Trumpet"
date: 2022-08-30
author: "Moss"
header-img: "img/reddit_background_2.jpg"
mathjax: true
tags: []
comments: true
---

<h1>Latent Dirichlet allocation (LDA) on Reddit Data</h1>

So I did this neat LDA project for a course in Uni and proudly listed it in my resume.
(Note: I did end up copying most of the code from a <a href="https://tomvannuenen.medium.com/analyzing-reddit-communities-with-python-part-5-topic-modeling-a5b0d119add">blog</a> post I found at the time.)
Fastforward to an interview and out of **all** the listed projects on my resume, they happened to only ask me about that one.

> "Can you tell us more about what you did for the LDA project?" - Interviewer
>
> "I'm not so sure.. I just used a function from the scikit-learn library" - Me

Yea.. not good....

I did email them back with an explanation and more details on the project but I think the damage was done.
I'd like to revisit that project, run it on new data, and hopefully uncover something interesting to share with everyone.

<h2> Background </h2>

Before we start, Its worth going over what LDA is and where its used:

<h3>Latent Dirichlet Allocation</h3>

Now there are a number of blog posts out there that describe LDA really well:

- <a href="https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d#:~:text=Latent%20Dirichlet%20Allocation%20(LDA)%20is,and%20topic%20modelling%2C%20among%20others">This great Medium post with an interactive tool</a>.
- <a href="https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2">Heres another</a>

Here's my take:

Latent Dirichlet Allocation is an unsupervised clustering algorithm.

It is often used for topic modelling on documents to:
  - uncover hidden themes in the documents
  - then classify the documents into those themes
  - use this new information to help search/sort/explore the documents


Lets say you are searching up a guide to cook meals for your workouts and you use the search query "gym muscle building food". A search engine might find and return the document "Meal-Prep for bulking". One reason it might have found it is because the document has
been preprocessed and categorized with the topics *gym* and *food*.

That preprocessing step was likely done with an LDA process.

<!--
![An LDA example](/img/lda_example.jpg)
-->

Lets say we have the following documents and a list of the words in each document (sorted by frequency):

| document | words (most frequent first) |
|----------|-----------------------------|
| "beach body.txt"| {word20:muscle, word23:body, word30:gym,... } |
| "meal prep for beginners.txt"| {word4:beginner, word54:Weight, word1:Hello ... } |
| "meal prep for bulking.txt"| {word4:cook, word18:chicken, word55:excercise, ... } |
| "meal prep for bulking.txt"| {word4:cook, word18:chicken, word55:excercise, ... } |
| "meal prep for bulking.txt"| {word4:cook, word18:chicken, word55:excercise, ... } |

> *can you tell what's been on my mind lately?*

Lets say there are 3 hidden topics amongst the documents... Lets figure out the probabilities for each word belonging to each topic.

|                         | word20:muscle | word4:cook | word18:chicken | ... |
|-------------------------|---------------|------------|----------------|-----|
|Topic1:(could be gym)    | 0.4           | 0.1        | 0.14           |     |
|Topic1:(could be fat)    | 0.04          | 0.64       | 0.25           |     |
|Topic1:(could be protein)| 0.3           | 0.02       | 0.7            |     |

If we had the above information, we could pick the top 10 (probability) words for each topic and use that to categorize documents by Topics.

So.. How do we get there?


### The Algorithm:

1. First go through each document and assign each word randomly to one of your k (3 in the example above) topics.  

2. For each document again go through each word and calculate:  
    1. Frist $$p(t \vert d)$$ (excluding the current word) to get the number of words that belong to topic t for a document d.
    2. Then, $$p(w \vert t)$$ to get how many documents are in a topic t since it has the word w.

3. Update $$p(w \vert t,d)$$ by $$p(w \vert t,d) = p(t \vert d)*p(w \vert t)$$

Repeatedly execute the above algorithm to finally get to that table above.

<a href="https://lh3.googleusercontent.com/qDmNDyUh-g4qAlu7uOfW5lUqOfURGH4sL9PPMJGirN835-SbPeAYB55ILetQnC-QXOyECvl2UCOlK9-WJjWN9OXXQrYGjaDFZx8qcgt0RynXiRWIdNGUvUby0ry-0i50IvGmYLo">Heres a great walk-through.</a>

<!--
Some more math:
https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d#:~:text=Latent%20Dirichlet%20Allocation%20(LDA)%20is,and%20topic%20modelling%2C%20among%20others.
If alpha is really low then you'll end up with a document only ever having one topic.
When alpha is greater than one then the
The Beta Parameter controls the distribution of words per topic.
A Higher beta will result in topics having more words and vice versa.
We gotta estimate phi and theta.
-->

<!--
#### Lets Look at the Math:

> "Is LDA a generative or discriminative model?"
>
> "Yes. LDA assumes a generative process. since we are learning p(y|x) indirectly."
> ... Is what I should have said.
>


That means that

![LDA formula](/img/lda_formula.jpg)

θ - a distribution of topics
z - N topics for each document
B - A distribution of words, one for each topic
D - Corpus
Alpha - parameter vector for each document
n - parameter vector for each topic

This cannot be calculated since it has an intractable posterior.
So we get an estimate by minimizing the KL Divergence between an approximation and the true posterior as an optimization problem.

![kl_div](/img/kl_div.jpg)
<!--<h3>NLP</h3>-->


### The Dataset - Reddit

You've all heard of reddit.. If you havnt come across it on a browser perhaps you've come by the tiktoks that just text-to-speech spicy reddit posts.

![tiktok..](/img/reddit_tiktok_1.jpg)
*You tried to click on it didnt you?*

<!--TODO: make it say "You tried to click it didnt you? when the image is hovered over"-->
<!--
<div class="container">
  <img src="/img/reddit_tiktok_1.jpg" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Hello World</div>
  </div>
</div>
-->


I got the data from : <a href="https://files.pushshift.io/reddit/">https://files.pushshift.io/reddit/</a>

The 2018-10 data is 3.5GB compressed (in zst format) and 41GB extracted.
The 2022-07 data is 10GB condensed and 142GB extracted. (this will be important later.)

Now here is the reason why I had a month of delay actually posting this blog.
I tried to use PySpark to read the condensed JSON. I got some errors...
I tried to debug.. and went down a rabbit hole of errors..
A month later, during a long weekend, realized I can just unzstd (extract) the file and then proceed to read the JSON using Spark.

I'm not entirely sure what the performance benefit of running a local instance of Spark is, but I am slightly familiar with the library. Using SparkSQL and running a query like:
```
topScoringTitles = spark.sql("SELECT author, author_id, author_created_utc, subreddit,
subreddit_subscribers, title, score  FROM Submissions ORDER BY score DESC LIMIT 50")
```
took 5 minutes, until I adjusted a few PySpark submit arguments (the no. executor cores and amount of allocation memory) and brought the time down to a 1 minute.

Anyway here's some useless facts I found:

The top 3 scoring submissions of October 2018 had the following titles:
1. <a href="https://i.imgur.com/4a3Ch82.gifv">Kids in Elementary school hold a surprise party for their beloved school custodian.</a>  
2. <a href="https://i.redd.it/co0b4d9908s11.jpg">Tried to take a panoramic picture of the Eiffel Tower today, it went surprisingly well!</a>
3. <a href="https://i.imgur.com/JpDig9C.gifv">Trump boards Air Force One with toilet paper stuck to his shoe.</a>  


## The Algorithm:

The 25 most popular subreddits were first chosen. The top 1000 highest scoring headlines from each subreddit were then collected. These headlines were first preprocessed and then concatenated to create documents for each Subreddit. Finally the LDA process was run where 15 hidden underlying topics were discovered.

### Preprocessing:
1. Headlines were tokenized, lowercased and filtered of punctuation and stop words.
2. Porter Stemmer was used to change words to first person and present tense words. (Eg. Apples --> Apple)

![pipeline](/img/reddit_data_pipeline.jpg)

### Hyper-Parameters:
There are many hyper-parameters to be considered while executing LDA.
- Topics = 15
- Titles from one Subreddit chosen to be the document
- The use of Porter Stemmer instead of Snowball Stemmer in pre-processing
- Words that appear in less than 4 documents (or subreddits) are excluded
- Words that appear in more than 80% of the documents are excluded
- Only the first 1,000,000 tokens are retained after above steps

<!--My code is posted here:-->
### LDA
I used the gensim library to run the LDA process.

First I had to create a dictionary of the documents:
```
dictionary = gensim.corpora.Dictionary(ppdFrames)
```
I then filtered the words in each document according the hyper-parameters listed above.
```
dictionary.filter_extremes(no_below=4, no_above=0.8, keep_n=1000000)
```

<!--
A Tf-idf Model was calculated.

> Term frequency-inverse document frequency is a numerical statistic that is intended to reflect how important a word is to a document in a collection.

```
tfidf = models.TfidfModel(bow_corpus)
```
-->

And finally:
```
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=15, id2word=dictionary, passes=3, workers=2)
```

## Results:

![lda_results](/img/reddit_lda_results_1.jpg )
*Words in relation to Topics*

Topics are still abstract concepts here and only hold numbers at the moment. However, we can imagine
these topics to be centered around some loose topic. Topics 15 has high values for words revolving
around finance. Topics 14 and 15 revolve around nsfw material. One interesting observation is that the
word ‘trump’ is distributed across the greatest number of topics. This is expected, since mentions of
‘Trump’ in media were very popular at the time.

![lda_results](/img/reddit_lda_results_2.jpg )
*Top 25 Subreddits and Topics most likely to be assigned to them*

Topics 4, 6, 8, 10 and 12 were not used. This is likely do to the most popular subreddit consisting of
random headlines from a ‘ask any question’ subreddit ‘AskReddit’. There are also at least 6 subreddits
where 90% of the contributions are by bots. These bots have created headlines with ‘http’, ‘trump’ and
‘market’, all of which are in the Topics not present and describe very general topics in the news today.
More stringent exclusion of words in the preprocessing can help derive more meaningful results from
those topics.

Another Interesting thing is that since my 2017 dataset is from October, the <a href="https://knowyourmeme.com/memes/skull-trumpet-doot-doot">Skull Trumpet</a> meme was doing its rounds. Topic 2 showcases its popularity especially when the topic is assigned back to subreddits like: PewdiepieSubmissions, dankmemes, funny and memes.

## New Data:
Alright, at the time of writing the RS_2022-07 file is 142GB in size.
I havnt set up a cluster in order to read that json file so I will save that for a later date.

## Conclusion:

Reddit is ripe with tough data analytics projects. There are still many combinations of hyper-parameters
to experiment with and more room for stronger data pre-processing. It is rather difficult to say how
many hidden features should exist.

## Sources:

- D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” The Journal of Machine Learning
Research, vol. 3, pp. 993–1022, Mar. 2003.

- Jason Baumgartner, “pushshift.io – learn about big data and social media ingest and analysis.”
[Online] Available: https://pushshift.io
