
import json, codecs

arc_set = set()
for line in codecs.open('arcs','rU','utf-8'):
    for token in line.strip().split():
        for arc in token.split(','):
            if arc != '':
                arc = arc.split('::')[0]
                arc = arc.split('_')[0]
                arc = arc.split('(')[0]
                arc_set.add(arc)

print len(arc_set)
json.dump(list(arc_set), codecs.open('arcs_set.json','w','utf-8'))
