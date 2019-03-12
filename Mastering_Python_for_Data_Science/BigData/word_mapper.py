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
