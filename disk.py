#!/usr/bin/env python

import os
import csv
import configparser
import operator as op

import pandas as pd

from correct_accel_gyro_hmm_06_05 import fix_imu

fieldnames_3q = ('csv_path', 'center_stable', 'consistent w expectations',
                 'before/after_visually_consistent',)

def output_data():
    foo = {}
    config = configparser.ConfigParser()
    config.read('config.ini')
    foo['handle'] = open(config['paths']['output'], 'r+')
    bar = csv.DictReader(foo['handle'])
    bar = tuple(bar)
    fieldnames = ('csv_path', 'idx_start', 'duration_samp',
                  'shift_state', 'annotation',)
    output_writer = csv.DictWriter(foo['handle'], fieldnames=fieldnames)
    if os.stat(config['paths']['output']).st_size == 0:
        output_writer.writeheader()
    foo['writer'] = output_writer
    foo['complete'] = set(map(op.itemgetter('csv_path'), bar))

    ###

    foo['handle_3q'] = open(config['paths']['output_3q'], 'r+')
    bar_3q = csv.DictReader(foo['handle_3q'])
    bar_3q = tuple(bar)
    fieldnames_3q = ('csv_path', 'center_stable', 'consistent w expectations',
                     'before/after_visually_consistent',)
    output_writer_3q = csv.DictWriter(foo['handle_3q'],
                                      fieldnames=fieldnames_3q)
    if os.stat(config['paths']['output_3q']).st_size == 0:
        output_writer_3q.writeheader()
    foo['writer_3q'] = output_writer_3q
    foo['complete_3q'] = set(map(op.itemgetter('csv_path'), bar_3q))
    return foo


def load_data(input_file):
    foo = {}
    foo['input_file'] = os.path.basename(input_file)
    dfi = pd.read_csv(input_file)
    foo['dfi'] = dfi
    dfo, shift_state_index, _, _ = fix_imu(dfi, plot=False,
                                           seconds_remove=2,
                                           low_motion_adjust=True)
    foo['dfo'] = dfo
    foo['ssi'] = shift_state_index
    foo['time(s)'] = dfo['Time(ms)']/1000

    return foo


def load_ecf(data):
    config = configparser.ConfigParser()
    config.read('config.ini')
    basename = os.path.basename(data['input_file'])
    subj_tp = '_'.join(basename.split('_')[:2])
    ecf_file = f'{subj_tp}_ECF.csv'
    if os.path.exists(os.path.join(config['paths']['ecfs'], ecf_file)):
        ecf_file = os.path.join(config['paths']['ecfs'], ecf_file)
        handle = open(ecf_file, 'r')
        ecf = tuple(csv.DictReader(handle))
        handle.close()
    else:
        ecf = None
    data['ecf'] = ecf
    return data
