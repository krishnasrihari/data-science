
# coding: utf-8

# # Analyzing unstructured data with text mining

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from twython import Twython
import time
import json


# There is a lot of unstructured data out there such as news articles, customer feedbacks, twitter tweets and various others and there are information that is there in them which would be very useful to analyze. Text Mining is a data mining technique which helps us in  performing analysis of this unstructured data.
# 
# In this chapter, we'll learn 
# 
# 1. How to preprocess the data
# 2. Plot a wordcloud from the data
# 3. Word and Sentence Tokenization
# 4. Part of Speech Tagging
# 5. Stemming and Lemmatization
# 6. Applying Stanford's Named Entity Recognizer

# ## Preprocessing the data

# We'll be using the reviews of Mad Max: Fury Road from bbc, forbes, guardian and movie pilot. We'll be performing the following on the data
# 
# 1. Removing Punctuation
# 2. Removing Numbers
# 3. Converting text to lower case
# 4. Removing stopwords like be, the, on etc
# 
# Let's start by loading the data first
# 

# In[11]:

data = {}

#data['bbc'] =

data['bbc'] = open('./Data/madmax_review/bbc.txt','r').read()
data['forbes'] = open('./Data/madmax_review/forbes.txt','r').read()
data['guardian'] = open('./Data/madmax_review/guardian.txt','r').read()
data['moviepilot'] = open('./Data/madmax_review/moviepilot.txt','r').read()


# We'll convert the text to lower case

# In[12]:

#Conversion to lower case
for k in data.keys():
    data[k] = data[k].lower()

print data['bbc'][:800]


# Now, we'll remove the punctuation from the text

# In[13]:

#Removing punctuation
for k in data.keys():
    data[k] = re.sub(r'[-./?!,":;()\']',' ',data[k]) 

print data['bbc'][:800]


# We'll remove the numbers from the text

# In[14]:

#Removing number
for k in data.keys():
    data[k] = re.sub('[-|0-9]',' ',data[k])
    
print data['bbc'][:800]


# Post this, we'll remove stopwords which are commonly occuring words such as ours, yours, that, this etc

# In[15]:

#Removing stopwords
for k in data.keys():
    data[k] = data[k].split()

stopwords_list = stopwords.words('english')
stopwords_list = stopwords_list + ['mad','max','film','fury','miller','road']

for k in data.keys():
    data[k] = [ w for w in data[k] if not w in stopwords_list ]
    
print data['bbc'][:80]


# ## Creating the Word Cloud

# We'll be using Word Cloud package by amueller. You can download it with the following command if you are using Ubuntu
# 
# sudo pip install git+git://github.com/amueller/word_cloud.git
# 
# Or you can follow the instruction through the following link https://github.com/amueller/word_cloud
# 
# Let's plot the wordcloud for BBC

# In[16]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(data['bbc']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# From the above Word Cloud, we can make out that there is mention about the time between the previous and current movie. The article talks about Mel Gibson , the cars and the villain Immortan Joe as these are the most frequently occuring keywords. There is also emphasis on different aspects of the movie given by the keyword "one"

# Now let's see how it looks like for Forbes. 

# In[18]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(data['forbes']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Forbes talks about the female characters more. 

# The following is for the Guardian

# In[20]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(data['guardian']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# The Guardian has emphasis over the female characters and lack of water in the wasteland.

# Finally, the moviepilot

# In[22]:

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(data['moviepilot']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# The moviepilot has emphasis on immortan joe, the characters in general and way boys shown in the film.

# ## Word and Sentence Tokenization

# We had done word tokenization previously but we can perform the word tokenization using NLTK as well as sentence tokenization which is quite tricky as the english language has period symbols for abbreviation and other purposes.Thankfully, the sentence tokenizer is a  instance of the PunktSentenceTokenizer from tokenize.punkt module of nltk which helps in tokenizing sentences.
# 
# Let's look at word tokenization

# In[34]:

#Loading the forbes data
data = open('./Data/madmax_review/forbes.txt','r').read()

word_data = nltk.word_tokenize(data)
word_data[:15]


# Now, let's perform the sentence tokenization of the forbes article

# In[33]:

sent_tokenize(data)[:5]


# ## Part of Speech Tagging

# Part of Speech tagging is one of the important task of text analysis. It helps in tagging each of the word based on the context of the sentence or the role the word is playing in the sentence.
# 
# Let's see how to perform part of speech taggin using NLTK

# In[42]:

pos_word_data = nltk.pos_tag(word_data)
pos_word_data[ : 10]


# You can see the tags like NNS, CC, IN , TO, DT and NN. Let's see what they mean 

# In[44]:

nltk.help.upenn_tagset('NNS')


# In[45]:

nltk.help.upenn_tagset('NN')


# In[46]:

nltk.help.upenn_tagset('IN')


# In[47]:

nltk.help.upenn_tagset('TO')


# In[48]:

nltk.help.upenn_tagset('DT')


# In[49]:

nltk.help.upenn_tagset('CC')


# You can see that the words get tagged well. This tagging can help us in creating heuristics over the data and then extracting information out of it. For example, we can take out all the nouns in our article and analyze the theme around which the article is about.

# ## Stemming and Lemmatization

# ### Stemming

# Stemming is process of reducing a word to its root form. The root form is not a word by itself but words can be formed by adding the right suffic to it.
# 
# If you take fish, fishes and fishing, they all can be be stemmed to fishing. Also study, studying and studies can be stemmed to studi which is not part of the English Language.
# 
# There are various types of stemming algorithm like Porter, Lancaster, Snowball etc. 
# 
# Let's try out the Porter Stemming Algorithm

# In[64]:

from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

for w in word_data[:20]:
    print "Actual: %s  Stem: %s"  % (w,porter_stemmer.stem(w))


# Let's try out the Lancaster Algorithm

# In[66]:

from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()

for w in word_data[:20]:
    print "Actual: %s  Stem: %s"  % (w,lancaster_stemmer.stem(w))


# Now, let's try out the Snowball Algorithm

# In[69]:

from nltk.stem.snowball import SnowballStemmer

snowball_stemmer = SnowballStemmer("english")

for w in word_data[:20]:
    print "Actual: %s  Stem: %s"  % (w,snowball_stemmer.stem(w))


# Porter is the most commonly used stemmer. It is also on of the most gentle stemmers. It is also the one of the most computationally intensive of the algorithms. 
# 
# Snowball is regarded as an improvement over porter. Porter himself in fact admits that Snowball is better than his algorithm. 
# 
# Lancaster is a very aggressive stemming algorithm and sometimes to a fault. With porter and snowball, the stemmed representations are usually fairly understandable to a reader but not so with Lancaster as many shorter words becomes totally obfuscated. Lancaster is the fastest algorithm here, and will reduce your working set of words hugely, but if you want more distinction it is not the tool that you would want.
# 
# 

# ### Lemmatization

# Lemmatization is similar to stemming but it groups together similar together to a Lemma.
# 
# We'll use WordNet's Lemmatization 

# In[114]:

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

for w in word_data[:30]:
    print "Actual: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w))
    


# ## Stanford's Named Entity Recognizer

# Named Entity Recognizer is a task to classify the elements of a sentence to categories such as Person, Organization, Locations etc.
# Stanford's Named Entity Recognizer is one of the most popular one out there.
# 
# The stanford's Named Entity Recognizer can be downloaded here http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip

# In[113]:

from nltk.tag.stanford import NERTagger

st = NERTagger('./lib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', './lib/stanford-ner/stanford-ner.jar')

st.tag('''Barrack Obama is the president of the United States of America . His father is from Kenya and Mother from United States of America. 
       He has two daughters with his wife. He has strong opposition in Congress due to Republicans'''.split()) 


# You can see that the Stanford Named Entity Tagger does a pretty good job in tagging the Person, Location and Organization

# ## Sentiment Analysis on World Leaders using twitter 

# Before we start off analyzing tweets, we'll need to install the Twython package of python to get the tweets from twitter from the following link.
# 
# https://github.com/ryanmcgrath/twython
# 
# Also, you need to get the consumer key and consumer secret key by going using this link https://apps.twitter.com/app/new
# 
# <img src="files/images/twitter_app.png">
# 
# Once you have given the details about your app. You'll get the consumer key and consumer secret key
# 
# <img src="files/images/twitter_app_token.png">
# 

# After this go to the Key and Access Tokens tabs to generate your access token .
# 
# <img src="files/images/twitter_access_token.png">
# 
# Once you have the required keys, we'll add the details to the following code

# In[2]:

#Please provide your keys here
TWITTER_APP_KEY = 'xxxxxxx' 
TWITTER_APP_KEY_SECRET = 'xxxxxxx' 
TWITTER_ACCESS_TOKEN = 'xxxxxxx'
TWITTER_ACCESS_TOKEN_SECRET = 'xxxxxxx'

t = Twython(app_key=TWITTER_APP_KEY, 
            app_secret=TWITTER_APP_KEY_SECRET, 
            oauth_token=TWITTER_ACCESS_TOKEN, 
            oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

def get_tweets(twython_object, query, n):
    count = 0
    result_generator = twython_object.cursor(twython_object.search, q = query)
    result_set = []
    for r in result_generator:
        result_set.append(r['text'])
        count += 1
        if count == n: break
    
    return result_set


# Now, we have access to the twitter tweets and we can fetch them now.
# 
# We'll be analyzing the sentiment of the tweets for Obama, Putin, Modi, Xi Jin Ping, and David Cameron. There are few assumptions that we'll be making for our analysis
# 1. The tweets are in english
# 2. We are limiting to 500 tweets 
# 
# You can load the tweets from the following json file

# In[ ]:

with open('./Data/politician_tweets.json', 'w') as fp:
    tweets=json.load(fp)


# Or you can fetch the fresh tweets of the these politicians.

# In[ ]:

tweets = {}
max_tweets = 500
tweets['obama'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#obama', max_tweets)]
tweets['putin'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#putin', max_tweets)]
tweets['modi'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#modi', max_tweets)]
tweets['xijinping'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#xijinping', max_tweets)]
tweets['davidcameron'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#davidcameron', max_tweets)]


# Now, lets define a function to score the sentiments of the tweets. We have positive and negative word list which will be used to compare the tweets and give the tweet a score. Every postive word that matches will be given +1 point and everey negative score that is captured will be given -1 point.

# In[10]:


positive_words = open('./Data/positive-words.txt').read().split('\n')
negative_words = open('./Data/negative-words.txt').read().split('\n')

def sentiment_score(text, pos_list, neg_list):
    positive_score = 0
    negative_score = 0
    
    for w in text.split(' '):
        if w in pos_list: positive_score+=1
        if w in neg_list: negative_score+=1
            
    return positive_score - negative_score

    


# Let's now score the sentiments of each tweet in the list

# In[11]:

tweets_sentiment = {}
tweets_sentiment['obama'] = [ sentiment_score(tweet,positive_words,negative_words) for tweet in  tweets['obama'] ]
tweets_sentiment['putin'] = [ sentiment_score(tweet,positive_words,negative_words) for tweet in tweets['putin'] ]
tweets_sentiment['modi'] = [ sentiment_score(tweet,positive_words,negative_words) for tweet in tweets['modi'] ]
tweets_sentiment['xijinping'] = [ sentiment_score(tweet,positive_words,negative_words) for tweet in tweets['xijinping'] ]
tweets_sentiment['davidcameron'] = [ sentiment_score(tweet,positive_words,negative_words) for tweet in tweets['davidcameron'] ]


# We have defined a dict called tweets_sentiment where we have scored the sentiments of each of the tweets for the politicians.
# 
# Now as we have the sentiment score for each of the politicians, we'll now analyze the sentiments for each of the politcians
# 
# Let's see how people are feeling about Obama

# In[25]:

obama = plt.hist(tweets_sentiment['obama'], 5, facecolor='green', alpha=0.5)
plt.xlabel('Sentiment Score')
_=plt.xlim([-4,4])


# Mostly, neutral tweets about Obama.
# 
# Let's see how its for Putin

# In[35]:

putin = plt.hist(tweets_sentiment['putin'], 5, facecolor='green', alpha=0.5)
plt.xlabel('Sentiment Score')
_=plt.xlim([-4,4])


# Mostly, tweets are neutral with slight negativity
# 
# Let's see how it is for Modi.

# In[36]:

modi = plt.hist(tweets_sentiment['modi'], 5, facecolor='green', alpha=0.5)
plt.xlabel('Sentiment Score')
_=plt.xlim([-4,4])


# Mostly, the tweets are neutral for Modi with slight positivity.
# 
# Let's see for Xi Jin Ping.

# In[31]:

xijinping = plt.hist(tweets_sentiment['xijinping'], 5, facecolor='green', alpha=0.5)
plt.xlabel('Sentiment Score')
_=plt.xlim([-4,4])


# So Xi Jin Ping also mostly Negative tweets.

# In[32]:

davidcameron = plt.hist(tweets_sentiment['davidcameron'], 5, facecolor='green', alpha=0.5)
plt.xlabel('Sentiment Score')
_=plt.xlim([-4,4])


# The tweets for David Cameron is inclined more towards positive side.
