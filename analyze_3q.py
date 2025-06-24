#!/usr/bin/env python

import collections as cts
import operator as op
import csv

def main():
    path = './workdir/output_3q.csv'
    handle = open(path, 'r')
    reader = csv.reader(handle)
    data = map(op.itemgetter(1,2,3), reader)
    data = cts.Counter(data)
    handle.close()
    print(data)

    return None

if __name__ == '__main__':
    main()
