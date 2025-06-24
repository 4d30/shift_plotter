#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import operator as op
import itertools as its
import functools as fts
import multiprocessing as mp

import load_plot as lp
import tui
import disk


def three_questions(data, input_file, proc, plots, output, debug=False):
    answers = []
    for q_n in range(1, 4):
        stdscr = tui.initscr()
        text = '\n'.join([tui.three_questions_text[0],
                          tui.three_questions_text[q_n]])
        text = text.format(input_file)
        tui.display_text(stdscr, text)
        while True:
            char = tui.get_3q_input(stdscr)
            match char:
                case 'q':
                    proc.terminate()
                    proc.join()
                    tui.killscr(stdscr)
                    sys.exit(0)
                case 'h':
                    tui.display_text(stdscr, tui.help_text)
                    tui.get_3q_input(stdscr)
                    continue
                case 'x':
                    proc.terminate()
                    proc.join()
                    proc = mp.Process(target=next(plots), args=(data,))
                    proc.start()
                    tui.killscr(stdscr)
                    continue
                case 'y':
                    answers.append(char)
                    break
                case 'n':
                    answers.append(char)
                    break
    csv_path = os.path.basename(input_file)
    row = [csv_path] + answers
    row = zip(disk.fieldnames_3q, row)
    output['writer_3q'].writerow(dict(row))
    output['handle_3q'].flush()
    return None


def annoate_shift_groups(data, input_file, proc, plots, output, debug=False):
    for shift_group in enumerate(data['ssi_groups']):
        while True:
            if not debug:
                stdscr = tui.initscr()
                text = tui.annotation_text.format(input_file,
                                                  shift_group[0])
                tui.display_text(stdscr, text)
                char = tui.get_input(stdscr)
            else:
                char = input('enter command: ')
            match char:
                case 'q':
                    proc.terminate()
                    proc.join()
                    tui.killscr(stdscr)
                    return None
                case 'n':
                    proc.terminate()
                    proc.join()
                    proc = mp.Process(target=next(plots), args=(data,))
                    proc.start()
                    tui.killscr(stdscr)
                    continue
                case 'h':
                    tui.display_text(stdscr, tui.help_text)
                    tui.get_input(stdscr)
                    continue
                case 's':
                    tui.display_text(stdscr, text)
                    break
                case 'a':
                    tui.display_text(stdscr, text)
                    break
                case 'k':
                    tui.display_text(stdscr, text)
                    break
        row = lp.make_output_row(data, shift_group, char)
        output['writer'].writerow(row)
    if 'stdscr' in locals():
        tui.killscr(stdscr)
    return None


def main():
    debug = False
    if '-d' in sys.argv:
        debug = True
    input_files = lp.read_workdir()
    input_files = list(input_files)
    output = disk.output_data()
    done = output['complete_3q']
    done = map(os.path.join, its.repeat('./workdir/input'), done)
    done = tuple(done)
    predicate = fts.partial(op.contains, done)
    input_files = its.filterfalse(predicate, input_files)
    input_files = list(input_files)
    for input_file in input_files:
        print(input_file)
        plots = its.cycle([lp.complicated_mag_plot, lp.complicated_tri_plot])
        data = lp.load_and_transform(input_file)
        is_shifted = map(op.itemgetter('shift_state'), data['ssi_groups'])
        is_shifted = map(op.eq, its.repeat(0), is_shifted)
        is_shifted = all(is_shifted)
        is_shifted = op.not_(is_shifted)
        if not is_shifted:
            print('continue')
            breakpoint()
            continue
        plot_target = next(plots)
        proc = mp.Process(target=plot_target, args=(data,))
        proc.start()
        three_questions(data, input_file, proc, plots, output, debug)
        if data['ssi_groups']:
            annoate_shift_groups(data, input_file, proc, plots, output, debug)

        output['handle'].flush()
        proc.terminate()
        proc.join()
    return None


if __name__ == "__main__":
    main()
