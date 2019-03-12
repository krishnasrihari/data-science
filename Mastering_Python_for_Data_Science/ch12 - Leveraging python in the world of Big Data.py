# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Leveraging python in the world of Big Data

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hadoopy

# <markdowncell>

# We are generating more and more data day by day. We have generated more data this century than we have in the previous century and we  are currently only 15 years into this century. Big Data is the new buzz word and everyone is talking about it. It brings new possibilities. Google translate is able to translate any language thanks to Big data. We are able to decode our Human Genome. We can predict the failure of a turbine and do the required maintainence, thanks to Big Data.
# 
# There are 3 Vs of Big Data and they are defined as follows.
# 
# 1. Volume - This defines the size of the data. Facebook has petabytes of data about their users
# 2. Velocity - This is the rate at which the data is generated. 
# 3. Variety - Data is not only in table form. There is data from text, images and sound. Data comes in the form of json, xml and other  types
# 
# In this chapter, we'll learn how to use python in the world of Big Data by
# 1. Understanding Hadoop
# 2. Writing a Map Reduce program in Python
# 3. Using Pydoop 
# 4. Understanding Spark
# 5. Writing a spark program

# <headingcell level=1>

# What is Hadoop?

# <markdowncell>

# According to Apache Hadoop website, Hadoop is a framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage. Rather than rely on hardware to deliver high-availability, the library itself is designed to detect and handle failures at the application layer, so delivering a highly-available service on top of a cluster of computers, each of which may be prone to failures.
# 
# <img src="files/images/hadoop_architecture.png">

# <headingcell level=2>

# Programming Model

# <markdowncell>

# Map-Reduce is a programming paradigm that takes a large distributed computation as a sequence of distributed operations on large data sets of key-value pairs. The Map-Reduce framework takes a cluster of machines and executes Map-Reduce jobs across the machines in the cluster. A Map-Reduce job has two phases, a mapping phase and a reduce phase. The input to the Map Reduce is a data set of key/value pairs.
# 
# In the mapping phase,  Hadoop splits the input data set into a large number of fragments and assigns each fragment to a mapping task. Hadoop also distributes the many map tasks across the cluster of machines on which it operates. Each mapping task takes the key-value pairs from its assigned fragment and generates a set of intermediate key-value pairs. For each input key-value pair, the map task invokes a user defined mapping function that transforms the input into a different key-value pair .
# 
# Following the mapping phase the hadoop sorts the intermediate data set by key and produces a set of key value tuples so that all the values associated with a particular key appear together. It also divides the set of tuples into a number of fragments equal to the number of reduce tasks.
# 
# In the reduce phase, each reduce task consumes the fragment of key value tuples assigned to it. For each such tuple, it invokes a  reduce function that transforms the tuple into an output key/value pair. The hadoop framework distributes the many reduce tasks across the cluster of machines and deals with giving the appropriate fragment of intermediate data to each of the reduce task.
# 
# <img src="files/images/MapReduce.png">

# <headingcell level=2>

# Map Reduce Architecture 

# <markdowncell>

# It has a master/slave architecture. It has a single master server - jobtracker and several slave servers - tasktrackers, one per machine/node in the cluster. The jobtracker is the point of communication between users and the framework. Users submit map-reduce jobs to the jobtracker, which puts them in a queue of pending jobs and executes them on a first-come/first-served basis. The jobtracker manages the assignment of map and reduce tasks to the tasktrackers. The tasktrackers execute tasks upon instruction from the jobtracker and also handle data motion between the map and reduce phases.
# 

# <headingcell level=2>

# Hadoop DFS

# <markdowncell>

# Hadoop's Distributed File System is designed to store very large files across machines in a large cluster. It has been inspired by the Google File System. Hadoop DFS stores each file as a sequence of blocks, all blocks in a file except the last block are the same size. Blocks belonging to a file are replicated for fault tolerance. The block size and replication factor are configurable per file. Files in HDFS are "write once" and have strictly one writer at any time.

# <headingcell level=2>

# Hadoop DFS Architecture

# <markdowncell>

# Like Hadoop Map/Reduce, HDFS follows a master/slave architecture. An HDFS installation consists of a single Namenode, a master server that manages the filesystem namespace and regulates access to files by clients. In addition, there are a number of Datanodes, one per node in the cluster, which manage storage attached to the nodes that they run on. The Namenode makes filesystem namespace operations like opening, closing, renaming etc. of files and directories available via an RPC interface. It also determines the mapping of blocks to Datanodes. The Datanodes are responsible for serving read and write requests from filesystem clients, they also perform block creation, deletion, and replication upon instruction from the Namenode.

# <headingcell level=1>

# Python Map-Reduce

# <markdowncell>

# The installation of Hadoop won't be covered in this book but you can install it through the following link
# 
# http://www.cloudera.com/content/cloudera/en/documentation/cdh4/latest/CDH4-Quick-Start/cdh4qs_topic_3_2.html
# 
# We'll be using the Hadoop streaming api for executing our Python Map-Reduce program in Hadoop. The Hadoop Streaming API helps in using any program having standard input and output as a map reduce program.
# 
# We'll be writing two Map Reduce Program with python.
# 
# 1. A Basic Word Count
# 2. Getting the Sentiment Score of each review
# 3. Getting overall sentiment score from all the reviews
# 

# <headingcell level=2>

# Basic word count

# <markdowncell>

# We'll start with the word count map-reduce. Save the following code in a word_mapper.py file

# <codecell>

#!/usr/bin/env python

import sys

for l in sys.stdin:
    
    # Trailing and Leading white space is removed
    l = l.strip()
    
    # words in the line is split
    word_tokens = l.split()
    
    # Key Value pair is outputted
    for w in word_tokens:
        print '%s\t%s' % (w, 1)

# <markdowncell>

# In the above mapper code, each line of the file is stripped of the leading and trailing white spaces. The line is then into tokens of words and then  these tokens of words are outputted as key value pair of <word> 1.
# 
# Save the following code in word_reducer.py file

# <codecell>

#!/usr/bin/env python

from operator import itemgetter
import sys

current_word_token = None
counter = 0
word = None

# STDIN Input
for l in sys.stdin:
    # Trailing and Leading white space is removed
    l = l.strip()

    # input from the mapper is parsed
    word_token, counter = l.split('\t', 1)

    # count is converted to int
    try:
        counter = int(counter)
    except ValueError:
        # if count is not a number then ignore the line
        continue

    #Since hadoop sorts the mapper output by key, the following
    # if else statement works
    if current_word_token == word_token:
        current_counter += counter
    else:
        if current_word_token:
            print '%s\t%s' % (current_word_token, current_counter)
        
        current_counter = counter
        current_word_token = word_token

# The last word is outputed
if current_word_token == word_token:
    print '%s\t%s' % (current_word_token, current_counter)

# <markdowncell>

# In the above code, we use current_word_token to keep track of the current word that is being counted. In the for loop, we use word_token and counter to get the value out of the key value pair. We then convert the counter to int type.
# 
# In the if else statement, if the word_token is same as the previous instance which is current_word_token then we keep counting else  if its new word that has come then we output the word and its count. The last if statement is to output the last word. 
# 
# We can check out if the mapper is working fine by the following command

# <codecell>

%%bash
echo 'dolly dolly max max jack tim max' | ./BigData/word_mapper.py

# <markdowncell>

# Now we can check the reducer is also working fine by piping the reducer to the sorted list of the mapper output.

# <codecell>

%%bash
echo "dolly dolly max max jack tim max" | ./BigData/word_mapper.py | sort -k1,1  | ./BigData/word_reducer.py

# <markdowncell>

# Now, let's try to apply the same on a local file containing the summary of moby dick

# <codecell>

%%bash
cat ./Data/mobydick_summary.txt | ./BigData/word_mapper.py | sort -k1,1  | ./BigData/word_reducer.py

# <headingcell level=2>

# Sentiment Score for each review

# <markdowncell>

# We had written a program in the previous chapter to calculate the sentiment score, We'll extend that to write a map reduce program to determine the sentiment score for each review. Write the following code in senti_mapper.py

# <codecell>

#!/usr/bin/env python

import sys
import re

positive_words = open('positive-words.txt').read().split('\n')
negative_words = open('negative-words.txt').read().split('\n')

def sentiment_score(text, pos_list, neg_list):
    positive_score = 0
    negative_score = 0

    for w in text.split(' '):
        if w in pos_list: positive_score+=1
        if w in neg_list: negative_score+=1

    return positive_score - negative_score


for l in sys.stdin:
    
    # Trailing and Leading white space is removed
    l = l.strip()

    #Convert to lower case
    l = l.lower()

    #Getting the sentiment score	
    score = sentiment_score(l, positive_words, negative_words)
    
    # Key Value pair is outputted
    print '%s\t%s' % (l, score)

# <markdowncell>

# In the above code, we re use the sentiment score function from the previous chapter. For each line, we strip the leading and trailing white spaces and then get the sentiment score for review. Finally, we output the sentence and the score.
# 
# For this program, we don't require a reducer as we are calculating the sentiment in the mapper itself and we just have to output the sentiment score.
# 
# Lets' test out the mapper is working fine locally with a file containing the reviews for Jurassic World.

# <codecell>

%%bash
cat ./Data/jurassic_world_review.txt | ./BigData/senti_mapper.py 

# <markdowncell>

# We can see that our program is able to calculate the sentiment score well.

# <headingcell level=2>

# Overall Sentiment Score

# <markdowncell>

# To calculate the overall sentiment score, we would require the reducer and we'll use the same mapper but with slight modifications.
# 
# The following is the mapper code that we'll use stored in overall_senti_mapper.py.

# <codecell>

import sys
import re
import hashlib

positive_words = open('./Data/positive-words.txt').read().split('\n')
negative_words = open('./Data/negative-words.txt').read().split('\n')

def sentiment_score(text, pos_list, neg_list):
    positive_score = 0
    negative_score = 0

    for w in text.split(' '):
        if w in pos_list: positive_score+=1
        if w in neg_list: negative_score+=1

    return positive_score - negative_score


for l in sys.stdin:
    
    # Trailing and Leading white space is removed
    l = l.strip()

    #Convert to lower case
    l = l.lower()

    #Getting the sentiment score	
    score = sentiment_score(l, positive_words, negative_words)

    #Hashing the review to use it as a string
    hash_object = hashlib.md5(l)
    
    # Key Value pair is outputted
    print '%s\t%s' % (hash_object.hexdigest(), score)

# <markdowncell>

# The mapper code is similar to the previous mapper code and but here we are MD5 hashing the review and then outputting it as the key.
# 
# Following is the reducer coder that utilize to determine the overall sentiment about the movie. Store the following code in overall_senti_reducer.py

# <codecell>

from operator import itemgetter
import sys

total_score = 0

# STDIN Input
for l in sys.stdin:
   
    # input from the mapper is parsed
    key, score = l.split('\t', 1)

    # count is converted to int
    try:
        score = int(score)
    except ValueError:
        # if score is not a number then ignore the line
        continue

    #Updating the total score	
    total_score += score


print '%s' % (total_score,)

# <markdowncell>

# In the above code, we strip out the value containing the score and we then keep adding to the total_score variable. Finally, we output the total_score variable which shows the sentiment of the movie.
# 
# Let's test out locally the overall sentiment on Jurassic World which is a good movie and then test it out on the movie Unfinished Business which was critically poor.

# <codecell>

%%bash
cat ./Data/jurassic_world_review.txt | ./BigData/overall_senti_mapper.py | sort -k1,1  | ./BigData/overall_senti_reducer.py

# <codecell>

%%bash
cat ./Data/unfinished_business_review.txt | ./BigData/overall_senti_mapper.py | sort -k1,1  | ./BigData/overall_senti_reducer.py

# <markdowncell>

# We can see that our code is working well and we also see that Jurassic World has a more positive score which means people liked it a lot and Unfinished Business has negative value which shows that people didn't like it much.

# <headingcell level=2>

# Deploying Map Reduce code on Hadoop

# <markdowncell>

# We'll create a directory Moby Dick, Jurassic World and Unfinished Business Data in HDFS tmp folder.

# <codecell>

%%bash
hadoop fs -mkdir /tmp/moby_dick
hadoop fs -mkdir /tmp/jurassic_world
hadoop fs -mkdir /tmp/unfinished_business

# <markdowncell>

# Let's check if the folders are created.

# <codecell>

%%bash
hadoop fs -ls /tmp/

# <markdowncell>

# Once the folders are created, lets copy the data file to the respective folder.

# <codecell>

%%bash
hadoop fs -copyFromLocal ./Data/mobydick_summary.txt /tmp/moby_dick
hadoop fs -copyFromLocal ./Data/jurassic_world_review.txt /tmp/jurassic_world
hadoop fs -copyFromLocal ./Data/unfinished_business_review.txt /tmp/unfinished_business

# <markdowncell>

# Let's verify that the file is copied

# <codecell>

%%bash
hadoop fs -ls /tmp/moby_dick
hadoop fs -ls /tmp/jurassic_world
hadoop fs -ls /tmp/unfinished_business

# <markdowncell>

# We can see that files have been copied successfully.

# <markdowncell>

# With the following command, we'll execute our mapper and reducers script in hadoop with the following command. In the following command, we define the mapper, reducer, input and output file locations and then use hadoop streaming to execute our scripts.
# 
# Let's execute the word count program first.

# <codecell>

%%bash

hadoop jar /usr/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-*streaming*.jar -file ./BigData/word_mapper.py -mapper word_mapper.py -file ./BigData/word_reducer.py -reducer word_reducer.py -input /tmp/moby_dick/* -output /tmp/moby_output 

# <markdowncell>

# Let's verify that the word count map reduce program is working successfully.

# <codecell>

%%bash

hadoop fs -cat /tmp/moby_output/*

# <markdowncell>

# The program is working as intended. Now, we'll deploy the program of that calculates the sentiment score for each of the review. Do note we are adding the positive and negative dictionary files to the Hadoop streaming.

# <codecell>

%%bash

hadoop jar /usr/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-*streaming*.jar -file ./BigData/senti_mapper.py -mapper senti_mapper.py -file ./BigData/senti_reducer.py -reducer senti_reducer.py -input /tmp/jurassic_world/* -output /tmp/jurassic_output -file ./positive-words.txt -file negative-words.txt  

# <markdowncell>

# Let's check if its score the sentiments of review

# <codecell>

%%bash

hadoop fs -cat /tmp/jurassic_output/*

# <markdowncell>

# This program is also working as intended. Now we'll try out the overall sentiment of a movie.

# <codecell>

%%bash

hadoop jar /usr/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-*streaming*.jar -file ./BigData/overall_senti_mapper.py -mapper overall_senti_mapper.py -file ./BigData/overall_senti_reducer.py -reducer overall_senti_reducer.py -input /tmp/unfinished_business/* -output /tmp/unfinished_business_output -file ./positive-words.txt -file negative-words.txt  

# <markdowncell>

# Let's verify the result.

# <codecell>

%%bash

hadoop fs -cat /tmp/unfinished_business_output/*

# <markdowncell>

# We can see that the overall sentiment score is coming out correctly from Map Reduce. 
# 
# Below is a screenshot of the Jobtracker status page.

# <markdowncell>

# <img src="files/images/jobtracker.png">
# 
# The image shows a portal where the jobs submitted to the jobtracker can be viewed and the status can be seen. This can be seen on port 50070 of the master system. 
# 
# From the image, we can see that there is a job running and the status above the image shows that the job completed successfully.

# <headingcell level=2>

# File Handling with Hadoopy

# <markdowncell>

# Hadoopy is a library in python which provides API to interact with Hadoop to manage the files and perform map reduce on it. Hadoopy can be downloaded from the following location
# 
# http://www.hadoopy.com/en/latest/tutorial.html#installing-hadoopy
# 
# Let's try to put few files in hadoop through hadoopy in a directory created within hdfs called data

# <codecell>

%%bash
hadoop fs -mkdir data

# <markdowncell>

# The following is the code put the data into hdfs with the following code

# <codecell>

#!/usr/bin/env python
import hadoopy
import os

hdfs_path = ''


def read_local_dir(local_path):
    for fn in os.listdir(local_path):
        path = os.path.join(local_path, fn)
        if os.path.isfile(path):
            yield path


def main():
    local_path = './BigData/dummy_data'
    for file in  read_local_dir(local_path):
        hadoopy.put(file, 'data')
        print "The file %s has been put into hdfs" % (file,)

if __name__ == '__main__':
    main()

# <markdowncell>

# In the above code, we list all the files in a directory and then we put each of the file into hadoop using the put method of hadoopy.

# <markdowncell>

# Let's check if all the files have been put into hdfs

# <codecell>

%%bash
hadoop fs -ls data

# <markdowncell>

# So we have successfully been able to put files into hdfs.

# <headingcell level=2>

# Pig

# <markdowncell>

# <img src="files/images/pig-logo.gif">
# 
# Pig is a platform that gives a very expressive language to perform data transformations and querying. The that is written in Pig is in a scripting manner and this gets compiled to Map Reduce Programs which executes on Hadoop. 
# 
# Pig helps in reducing the complexity of the raw level Map-Reduce program and enable the user to perform transformations fast.
# 
# Pig latin can be learned from the this link http://pig.apache.org/docs/r0.7.0/piglatin_ref2.html
# 
# We'll be covering how to perform Top 10 most occuring words with Pig and then we show how you can create your function in python that can be used in Pig.
# 
# Let's start with the Word Count. Following is the Pig Latin Code which you can save it in pig_wordcount.py file.

# <codecell>

data = load '/tmp/moby_dick/';

word_token = foreach data generate flatten(TOKENIZE((chararray)$0)) as word;

group_word_token = group word_token by word;

count_word_token = foreach group_word_token generate COUNT(word_token) as cnt, group;

sort_word_token = ORDER count_word_token by cnt DESC;

top10_word_count = LIMIT sort_word_token 10; 

DUMP top10_word_count;

# <markdowncell>

# In the above code, we load the summary of Moby Dick which is then Tokenized line by line which is basically splitting it into individual elements. The Flatten function converts the Collection of individual word tokens in a line to a row by row form. We then group by the words and then take a count of the words for each word. Finally we sort the counts in descending order and then we limit to the first 10 rows to get the Top 10 most occuring words.
# 
# Let's execute the above pig script.

# <codecell>

%%bash

pig ./BigData/pig_wordcount.pig

# <markdowncell>

# We are able to get our top 10 words. Let's now create a user defined function with Python which will be used in Pig.
# 
# We'll define two user defined function to score the positive and negative sentiment of a sentence. 
# 
# The following is the udf for score the positive sentiment and its available in positive_sentiment.py

# <codecell>

positive_words = ['a+', 'abound', 'abounds', 'abundance', 'abundant', 'accessable', 'accessible', 'acclaim', 'acclaimed', 'acclamation', 'acco$



@outputSchema("pnum:int")
def sentiment_score(text):
    positive_score = 0

    for w in text.split(' '):
        if w in positive_words: positive_score+=1

    return positive_score

# <markdowncell>

# In the above code, we define the positive word list which is used by the sentiment_score function. The function checks for the positive words in a sentence and finally outputs the total count of it. There is a outputSchema decorator which is used to tell Pig what type of data is being outputted out which in our case is int. 
# 
# Following is the code for scoring the negative sentiment and its available in negative_sentiment.py. The code almost similar to the positive sentiment.

# <codecell>

negative_words = ['2-faced', '2-faces', 'abnormal', 'abolish', 'abominable', 'abominably', 'abominate', 'abomination', 'abort', 'aborted', 'ab$


@outputSchema("nnum:int")
def sentiment_score(text):
    negative_score = 0

    for w in text.split(' '):
        if w in negative_words: negative_score-=1

    return  negative_score

# <markdowncell>

# Following is the Pig which scores the sentiments of the Jurassic World review and its available in pig_sentiment.pig

# <codecell>

register 'positive_sentiment.py' using org.apache.pig.scripting.jython.JythonScriptEngine as positive;
register 'negative_sentiment.py' using org.apache.pig.scripting.jython.JythonScriptEngine as negative;

data = load '/tmp/jurassic_world/*';

feedback_sentiments = foreach data generate LOWER((chararray)$0) as feedback, positive.sentiment_score(LOWER((chararray)$0)) as psenti , 
negative.sentiment_score(LOWER((chararray)$0)) as nsenti;

average_sentiments = foreach feedback,feedback_sentiments generate psenti + nsenti;

dump average_sentiments;

# <markdowncell>

# In the above Pig Script, we first register the python udf scripts using the register command and give it an appropriate name. We then load our Jurassic World review. We then convert our reviews to lower case and then score the positive and negative sentiments of a review. Finally, we add the score to get the overall sentiment of a review.
# 
# Let's execute the Pig script and see the result.

# <codecell>

%%bash
pig ./BigData/pig_sentiment.pig

# <markdowncell>

# We have successfully scored the sentiments of Jurassic World Review using Python UDF in Pig.

# <headingcell level=2>

# Python with Apache Spark

# <markdowncell>

# <img src="files/images/spark.png">
# 
# 
# Apache Spark is a computing framework which works on top of HDFS and provides alternative way of computing similar to Map-Reduce. It was developed by AmpLab of UC Berkeley. Spark does its computation mostly in the memory because of which it is much more faster than Map-Reduce and is well suited for Machine Learning as its able to handle Iterative Work Loads really well.
# 
# 
# Spark used the programming abstraction of RDDs (Resilient Distributed Datasets) in which data is logically distributed into partitions and transformations can be performed on top of it.
# 
# Python is one of the language that is used for interacting with Apache Spark and we'll create a program to perform the sentiment scoring for each review of Jurassic Park as well as the overall sentiment.
# 
# You can install by following the instructions in the following link.
# 
# https://spark.apache.org/docs/1.0.1/spark-standalone.html
# 
# The following is the Python code for scoring the sentiment.

# <codecell>

from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

positive_words = open('positive-words.txt').read().split('\n')
negative_words = open('negative-words.txt').read().split('\n')


def sentiment_score(text, pos_list, neg_list):
    positive_score = 0
    negative_score = 0

    for w in text.split(' '):
        if w in pos_list: positive_score+=1
        if w in neg_list: negative_score+=1

    return positive_score - negative_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sentiment <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonSentiment")
    lines = sc.textFile(sys.argv[1], 1)
    scores = lines.map(lambda x: (x, sentiment_score(x.lower(), positive_words, negative_words)))
    output = scores.collect()
    for (key, score) in output:
        print("%s: %i" % (key, score))

    sc.stop()

# <markdowncell>

# In the above code, we define our standard sentiment_score function which we'll be reusing. The if statement checks that the python script and the text file is given. The sc variable is a Spark Context object with the App name "PythonSentiment". The filename in the argument is passed into spark through the textFile method of sc. In the map function of Spark, we define a lambda function where each line of the text file is passed and then we obtain the line and its respective sentiment score. The output variable gets the result and finally we print the result in the screen.
# 
# Let's score the sentiment of each of the review of Jurassic World.

# <codecell>

%%bash
~/spark-1.3.0-bin-cdh4/bin/spark-submit --master spark://samzer:7077 ./BigData/spark_sentiment.py hdfs://localhost:8020/tmp/jurassic_world/*

# <markdowncell>

# We can see that our Spark Program was able to score the sentiment for each of the review. We use the Spark Submit command and we define the Spark master with the python script that needs to be executed along with the location of the Jurassic World review in hdfs.
# 
# Below is a Spark program to score the overall sentiment of all the review.

# <codecell>

from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

positive_words = open('positive-words.txt').read().split('\n')
negative_words = open('negative-words.txt').read().split('\n')


def sentiment_score(text, pos_list, neg_list):
    positive_score = 0
    negative_score = 0

    for w in text.split(' '):
        if w in pos_list: positive_score+=1
        if w in neg_list: negative_score+=1

    return positive_score - negative_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Overall Sentiment <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonOverallSentiment")
    lines = sc.textFile(sys.argv[1], 1)
    scores = lines.map(lambda x: ("Total", sentiment_score(x.lower(), positive_words, negative_words)))\
                  .reduceByKey(add)
    output = scores.collect()
    for (key, score) in output:
        print("%s: %i" % (key, score))

    sc.stop()

# <markdowncell>

# In the above code, we have added a reduceByKey method which reduces the value by adding them and also we have defined the Key as "Total" so that all the scores reduced based on the single key.
# 
# Let's try out the above code to get the overall sentiment of Jurassic World.

# <codecell>

%%bash
~/spark-1.3.0-bin-cdh4/bin/spark-submit --master spark://samzer:7077 ./BigData/spark_overall_sentiment.py hdfs://localhost:8020/tmp/jurassic_world/*

# <markdowncell>

# We can see that Spark gave an overall sentiment score of 19.
# 
# The applications getting executed on Spark can viewed in the browser on the 8080 port of the Spark master. Following is a screenshot of it.
# 
# <img src="files/images/spark_monitor.png">
# 
# 
# We can see that the number of nodes of Spark, applications that are getting executed currently and as well as the applications that completed execution.

