#!/usr/bin/env python

import sys
import re
import hashlib

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

    #Hashing the review to use it as a string
    hash_object = hashlib.md5(l)
    
    # Key Value pair is outputted
    print '%s\t%s' % (hash_object.hexdigest(), score)
