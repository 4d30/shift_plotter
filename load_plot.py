#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# import sys
import csv
import configparser
import operator as op
import itertools as its

import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

import more_itertools as mits

# from correct_accel_gyro_hmm_06_05 import fix_imu
# import pandas as pd
import numpy as np

import disk


def get_sensor_id(data):
    config = configparser.ConfigParser()
    config.read('config.ini')
    basename = os.path.basename(data['input_file'])
    if '_' not in basename:
        data['sensor_id'] = None
        data['location'] = None
        data['age'] = None
        return data
    subj, tp = basename.split('_')[:2]
    sensor_id = basename.split('_')[2]
    spl = config['paths']['spl']
    handle = open(spl, 'r')
    spl = csv.DictReader(handle)
    spl = tuple(spl)

    def filt_a(row):
        AA = op.eq(row['SubjID'], subj)
        BB = op.eq(row['TimePoint'], tp)
        return op.and_(AA, BB)

    row = next(filter(filt_a, spl), None)
    if row is None:
        data['sensor_id'] = None
        data['location'] = None
        data['age'] = None
        return data

    def filt_b(key):
        return op.eq(row[key], sensor_id)

    location = next(filter(filt_b, row.keys()))
    data['sensor_id'] = sensor_id
    data['location'] = location
    data['age'] = tp
    handle.close()
    return data


def process_ssi(data):
    ssi = data['ssi']
    foo = np.zeros_like(ssi)
    is_shifted_idx = np.not_equal(ssi, foo)
    is_shifted_idx = its.compress(range(len(ssi)), is_shifted_idx)
    is_shifted_idx = tuple(is_shifted_idx)
    split_groups = mits.split_when(is_shifted_idx, lambda x, y: op.ne(y-x, 1))
    split_groups = tuple(split_groups)
    acc = []
    for group in split_groups:
        foo = {}
        foo['min'] = min(group)
        foo['max'] = max(group)
        foo['duration'] = foo['max'] - foo['min']
        foo['shift_state'] = ssi[foo['min']]
        acc.append(foo)
    data['ssi_groups'] = acc
    return data


def make_output_row(data, shift_group, char):
    row = {}
    row['csv_path'] = os.path.basename(data['input_file'])
    row['idx_start'] = shift_group[1]['min']
    row['duration_samp'] = shift_group[1]['duration']
    row['shift_state'] = int(shift_group[1]['shift_state'])
    row['annotation'] = char
    return row


def transform_data(data):
    instuments = ['Accl', 'Velo']
    dfs = ['dfi', 'dfo']
    for instrument, df in its.product(instuments, dfs):
        label = f'{df}_{instrument.lower()[0]}'
        cols = [c for c in data[df].columns if instrument in c]
        t_label = f'{label}_t'
        data[t_label] = data[df][cols].values
        m_label = f'{label}_m'
        data[m_label] = np.linalg.norm(data[t_label], axis=1)
    return data


def parse_ecf(data):
    if not data['ecf']:
        data['events'] = None
        return data
    iter_m = its.pairwise(data['ecf'])

    def helper(pair):
        AA = op.ne(pair[0]['Event'], '8')
        BB = op.eq(pair[1]['Event'], '8')
        return op.and_(AA, BB)
    events = filter(helper, iter_m)
    events = tuple(events)
    data['events'] = events
    return data


def parse_event(event):
    foo = {}
    if event:
        foo['event'] = event[0]['Event']
        foo['start'] = float(event[0]['Sensor Time(ms)'])/1000
        foo['end'] = float(event[1]['Sensor Time(ms)'])/1000
        foo['color'] = ''.join(['C', event[0]['Event'].replace(' ', '')])
    return foo


def complicated_tri_plot(data):
    iter_m = enumerate(['dfi', 'dfo'])
    iter_n = enumerate(['a', 'v'])
    if data['events']:
        iter_o = map(parse_event, data['events'])
        iter_o = sorted(iter_o, key=lambda x: x['event'])
        iter_o = tuple(iter_o)
    else:
        iter_o = []
    iter_p = tuple(range(3))

    product = [iter_m, iter_n, iter_p]

    fig, axs = plt.subplots(6, 2, figsize=(32, 9), sharex=True)
    title = f'{data["location"]}, {data["age"]}'
    fig.suptitle(title)
    event_set = set()
    for (ii, df), (jj, instrument), ll in its.product(*product):
        label = f'{df}_{instrument}_t'
        axs[ii*3 + ll, jj].scatter(data['time(s)'], data[label][:, ll],
                                   alpha=0.1, s=1,
                                   zorder=2)
        match instrument:
            case 'a':
                axs[ii*3 + ll, jj, ].set_ylim([-2, 2])
                axs[ii, jj, ].sharey(axs[0, jj])
                if ii*3 + ll == 0:
                    axs[ii*3 + ll, jj].set_title('Accelerometer')
            case _:
                if ii*3 + ll == 0:
                    axs[ii*3 + ll, jj].set_title('Gyroscope')
                axs[ii*3 + ll, jj].set_ylim([-200, 200])
                axs[ii*3 + ll, jj].sharey(axs[0, jj])
        for event in iter_o:
            if event['event'] not in event_set:
                event_set.add(event['event'])
                axs[ii*3 + ll, jj].axvspan(event['start'], event['end'],
                                           color=event['color'], alpha=0.2,
                                           zorder=1, label=event['event'])
            else:
                axs[ii*3 + ll, jj].axvspan(event['start'], event['end'],
                                           color=event['color'], alpha=0.2,
                                           zorder=1)
        for kk, split_group in enumerate(data['ssi_groups']):
            span = slice(split_group['min'], split_group['max'])
            times = data['dfi']['Time(ms)'][span].values/1000
            axs[ii*3 + ll, jj].scatter(times, data[label][:, ll][span],
                                       s=1, zorder=3, c='black', alpha=0.1)
            axs[ii*3 + ll, jj].axvline(times[0], color='black', linestyle='-',)
            axs[ii*3 + ll, jj].axvline(times[-1], color='black',
                                       linestyle='-',)
            axs[ii*3 + ll, jj].annotate(kk, (times[0],
                                             axs[ii*3 + ll, jj].get_ylim()[1]),
                                        color='#CB8600', fontweight='bold',
                                        fontsize=16)

#        axs[ii, jj].legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return None


def complicated_mag_plot(data):
    iter_m = enumerate(['dfi', 'dfo'])
    iter_n = enumerate(['a', 'v'])
    if data['events']:
        iter_o = map(parse_event, data['events'])
        iter_o = sorted(iter_o, key=lambda x: x['event'])
        iter_o = tuple(iter_o)
    else:
        iter_o = []

    product = [iter_m, iter_n]

    fig, axs = plt.subplots(2, 2, figsize=(32, 9), sharex=True)
    title = f'{data["location"]}, {data["age"]}'
    fig.suptitle(title)
    event_set = set()
    for (ii, df), (jj, instrument), in its.product(*product):
        label = f'{df}_{instrument}_m'
        axs[ii, jj].scatter(data['time(s)'], data[label],
                            alpha=0.1, s=1,
                            zorder=2)
        if jj == 0 and ii == 0:
            axs[ii, jj].set_ylabel('Before')
            if ii == 1:
                axs[ii, jj].set_ylabel('After')
        match instrument:
            case 'a':
                axs[ii, jj].set_ylim([0, 2])
                axs[ii, jj].sharey(axs[0, jj])
                if ii == 0:
                    axs[ii, jj].set_title('Accelerometer')
            case _:
                if ii == 0:
                    axs[ii, jj].set_title('Gyroscope')
                axs[ii, jj].set_ylim([0, 200])
                axs[ii, jj].sharey(axs[0, jj])
        for event in iter_o:
            if event['event'] not in event_set:
                event_set.add(event['event'])
                axs[ii, jj].axvspan(event['start'], event['end'],
                                    color=event['color'], alpha=0.2,
                                    zorder=1, label=event['event'])
            else:
                axs[ii, jj].axvspan(event['start'], event['end'],
                                    color=event['color'], alpha=0.2,
                                    zorder=1)
        for kk, split_group in enumerate(data['ssi_groups']):
            span = slice(split_group['min'], split_group['max'])
            times = data['dfi']['Time(ms)'][span].values/1000
            axs[ii, jj].scatter(times, data[label][span], s=1, zorder=3,
                                c='black', alpha=0.1)
            axs[ii, jj].axvline(times[0], color='black', linestyle='-',)
            axs[ii, jj].axvline(times[-1], color='black', linestyle='-',)
            axs[ii, jj].annotate(kk, (times[0], 0),
                                 color='#CB8600',
                                 fontweight='bold', fontsize=16)

#        axs[ii, jj].legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return None


def load_and_transform(input_file):
    data = disk.load_data(input_file)
    data = process_ssi(data)
    data = disk.load_ecf(data)
    data = parse_ecf(data)
    data = get_sensor_id(data)
    data = transform_data(data)
    return data


def read_workdir():
    config = configparser.ConfigParser()
    config.read('config.ini')
    input_files = os.listdir(config['paths']['input'])
    input_files = map(os.path.join,
                      its.repeat(config['paths']['input']),
                      input_files)
    return input_files


def main():
    input_files = read_workdir()
    for input_file in input_files:
        print(input_file)
        data = load_and_transform(input_file)
        complicated_tri_plot(data)
        continue
        complicated_mag_plot(data)

        breakpoint()


if __name__ == '__main__':
    main()
