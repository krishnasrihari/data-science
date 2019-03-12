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
