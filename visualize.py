#!/usr/bin/env python

import os
import csv
import array
import operator as op

import matplotlib.pyplot as plt


def load_iter():
    file_a = './workdir/output_iter.csv'
    handle = open(file_a, 'r')
    data = csv.DictReader(handle)
    data = tuple(data)
    handle.close()
    return data


def load_quads():
    file_a = './240729_frame_shift_inspection_quads.csv'
    handle = open(file_a, 'r')
    data = csv.DictReader(handle)
    data = tuple(data)
    handle.close()
    return data


def predicate(line):
    # Q I
    AA = op.eq(line['naive'], 'False')
    BB = op.eq(line['ss'], 'True')
    CC = op.and_(AA, BB)
    # IV
    DD = op.eq(line['naive'], 'True')
    EE = op.eq(line['ss'], 'True')
    FF = op.and_(DD, EE)
    return op.or_(CC, FF)



def main():
    iter_data = load_iter()
    quad_data = load_quads()
    q_files = filter(predicate, quad_data)
    q_files = map(op.itemgetter('file'), q_files)
    q_files = tuple(q_files)
    print(len(q_files))
    i_files = map(op.itemgetter('csv_path'), iter_data)
    q_files = set(q_files)
    i_files = tuple(i_files)
    print(len(i_files))
    i_files = set(i_files)
    print(len(i_files))
    q_files = q_files.difference(i_files)
    q_files = tuple(q_files)
    print(len(q_files), q_files)
    breakpoint()




    return None


if __name__ == '__main__':
    main()
