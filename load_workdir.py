#!/usr/bin/env python

import os
import sys
import csv
import ast
import collections as cts
import shutil
import configparser
import itertools as its
import operator as op

import pandas as pd
import more_itertools as mits

import psycopg2

def load_data(config):
    quad_handle = open(config['paths']['quad_file'], 'r')
    quads = csv.DictReader(quad_handle)
    quads = tuple(quads)
    quad_handle.close()
    return quads


def filter_quadrants(quads, pair):
    aa, bb = pair
    data = filter(lambda x: ast.literal_eval(x['naive']) == aa
                  and ast.literal_eval(x['ss']) == bb, quads)
    data = tuple(data)
    match aa, bb:
        case True, True:
            quad = 'IV'
        case False, True:
            quad = 'I'
        case True, False:
            quad = 'III'
        case False, False:
            quad = 'II'
    return (quad, data,)


def load_quads(config):
    quads = load_data(config)
    foo = its.repeat([True, False], 2)
    foo = its.product(*foo)
    quads = map(filter_quadrants, its.repeat(quads), foo)
    quads = dict(quads)
    return quads


def fetch_ecf_path(con, cur, subj_tp):
    subj, tp = subj_tp
    QQ = """SELECT
            (CASE WHEN ecf_path_updated IS NOT NULL
                THEN ecf_path_updated
                ELSE ecf_path
            END) AS ecf_path
            FROM study_visits
            WHERE subject_id = '{}' AND timepoint = '{}'
            aND part_no = 1
            LIMIT 1
            """.format(subj, tp)
    cur.execute(QQ)
    row = cur.fetchone()
    if row is not None:
        ecf_path = row[0]
    else:
        ecf_path = None
    return ecf_path

def open_db(config):
    con = psycopg2.connect(**config['database'])
    cur = con.cursor()
    return (con, cur)


def find_ecfs(config, con, cur, files):
    subj_tp = map(op.methodcaller('split', '_'), files)
    subj_tp = map(op.itemgetter(0, 1), subj_tp)
    subj_tp = tuple(set((subj_tp)))
    ecf_paths = map(fetch_ecf_path, its.repeat(con), its.repeat(cur), subj_tp)
    ecf_paths = filter(bool, ecf_paths)
    ecf_paths = map(os.path.join, its.repeat(config['paths']['rto']), ecf_paths)
    ecf_paths = tuple(ecf_paths)
    return ecf_paths


def find_sensors(config, con, cur, sensors):
    QQ = """SELECT csv_path from sensor_meta
            WHERE csv_path like '%{}'
            """
    acc = []
    for sensor in sensors:
        cur.execute(QQ.format(sensor))
        row = cur.fetchone()
        if row is not None:
            sensors = filter(lambda x: x != sensor, sensors)
            acc.append(row[0])
    sensors = map(os.path.join, its.repeat(config['paths']['rto']), acc)
    sensors = tuple(sensors)
    return sensors


def old_main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    df = pd.read_csv('foo.csv')
    df = df.set_index('csv_id')
    df = df[df['was_shifted'].notna()]
    df = df[df['was_shifted'].str.contains('t')]
    con, cur = open_db(config)
    src_sensors = df['csv_path'].astype(str).values
    src_sensors = map(os.path.join,
                      its.repeat(config['paths']['rto']),
                      src_sensors)
    src_sensors = tuple(src_sensors)
    src_ecfs = df['ecf_path'].astype(str).values
    src_ecfs = filter(lambda x: 'nan' not in x, src_ecfs)
    src_ecfs = map(os.path.join,
                   its.repeat(config['paths']['rto']),
                   src_ecfs)
    src_ecfs = tuple(src_ecfs)
    proc = map(shutil.copy,
               src_ecfs, its.repeat(config['paths']['ecfs']))
    cts.deque(proc, maxlen=0)
    proc = map(shutil.copy,
               src_sensors, its.repeat(config['paths']['input']))
    cts.deque(proc, maxlen=0)
    print('done')
    return None

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    conn = psycopg2.connect(**config['database'])
    cur = conn.cursor()
    QQ = """SELECT pa.csv_path,
        ecfs.ecf_path
        FROM pending_annotations pa
        JOIN sensor_csvs csvs
        on pa.csv_path = csvs.csv_path
        JOIN ecfs
        ON csvs.subject_id = ecfs.subject_id
        AND csvs.timepoint = ecfs.timepoint
        AND csvs.part_no = ecfs.part_no
        WHERE pa.annotation_type = 'frame_shift'
        """
    cur.execute(QQ)
    desc = cur.description
    desc = map(op.itemgetter(0), desc)
    desc = tuple(desc)
    rows = cur.fetchall()
    data = map(zip, its.repeat(desc), rows)
    data = map(dict, data)
    data = tuple(data)
    csvs = map(op.itemgetter('csv_path'), data)
    csvs = map(os.path.join, its.repeat(config['paths']['rto']), csvs)
    proc = map(shutil.copy,
               csvs, its.repeat(config['paths']['input']))
    cts.deque(proc, maxlen=0)
    ecfs = map(op.itemgetter('ecf_path'), data)
    ecfs = map(os.path.join, its.repeat(config['paths']['rto']), ecfs)
    proc = map(shutil.copy,
               ecfs, its.repeat(config['paths']['ecfs']))
    cts.deque(proc, maxlen=0)
    return None

if __name__ == '__main__':
    main()
