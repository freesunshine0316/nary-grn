
import json
import sys
import cPickle
import re
from collections import Counter
import codecs

def update(l, v):
    v.update([x.lower() for x in l])

def update_vocab(path, vocab):
    a = []
    b = 0
    c = 0
    words = []
    data = json.load(codecs.open(path,'rU','utf-8'))
    for inst in data:
        a.append(len(inst['sentences']) > 1)
        b += len(inst['sentences'])
        for sentence in inst['sentences']:
            for node in sentence['nodes']:
                words += node['lemma'].strip().split()
    c += len(words)
    update(words, vocab)
    return a,b,c

def output(d, path):
    f = codecs.open(path,'w',encoding='utf-8')
    for k,v in sorted(d.items(), key=lambda x:-x[1]):
        print >>f, k
    f.close()

##################

aa = []
bb = 0.0
cc = 0.0
vocab = Counter()
for line in open('data_list','rU'):
    a,b,c = update_vocab(line.strip(), vocab)
    aa += a
    bb += b
    cc += c
print len(vocab)
print 1.0*sum(aa)/len(aa), bb/len(aa), cc/len(aa)

#output(vocab, 'vocab_lemma.txt')

