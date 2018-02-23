
import os,sys
import numpy

inpath = sys.argv[1]
outpath = sys.argv[1]+'.st'
f = open(outpath,'w')
i = 0
words = set()
for line in open(inpath,'rU'):
    if i == 0:
        vsize = len(line.strip().split())-1
        print vsize
        print >>f, '\t'.join(['0', '#pad#', ' '.join([str('%.6f'%x) for x in numpy.zeros(vsize)])])
        print >>f, '\t'.join(['1', 'UNK', ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=vsize)])])
    line = line.strip().split()
    word = line[0]
    if len(line) != vsize+1 or word in words:
        continue
    words.add(word)
    line = ' '.join(line[1:])
    print >>f, '\t'.join([str(i+2), word, line])
    i += 1

