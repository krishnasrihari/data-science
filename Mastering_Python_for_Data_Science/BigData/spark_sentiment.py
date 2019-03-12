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
    scores = lines.map(lambda x: (x, sentiment_score(x, positive_words, negative_words)))
    output = scores.collect()
    for (key, score) in output:
        print("%s: %i" % (key, score))

    sc.stop()

