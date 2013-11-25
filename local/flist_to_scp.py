#!/usr/bin/python
import sys


for line in sys.stdin:
    print '%s %s' % ('_'.join(line.strip().split('/')[-3:])[:-len('.wav')],
                     line.strip())
