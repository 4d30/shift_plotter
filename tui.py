#!/usr/bin/env python

import unicurses

three_questions_text = ["""
Press (h) for help or (q) to quit

file: {}
""",
                        """
Are central values of both AFTER signals stable?
                       (y)es
                       (n)o
""",
                        """
Are both AFTER signals consistent with expectations?
                       (y)es
                       (n)o
""",

                        """
Are before and after magnitude signals visually consistent?
                       (y)es
                       (n)o
""",
                        ]




annotation_text = """
Press (h) for help or (q) to quit

file: {}
shift_group: {}
trust:
    (s)ensor
    (a)lgorithm
"""


help_text = """
    (q)uit
    (h)elp
    ne(x)t plot (tri or mag)
    trust (s)ensor
    trust (a)lgorithm

    Press (h) to return
"""


def clearscr(stdscr):
    unicurses.clear()
    unicurses.refresh()
    return stdscr

def get_3q_input(stdscr):
    char = unicurses.getkey()
    if char in 'hqxyn':
        return char
    else:
        return get_3q_input(stdscr)


def get_input(stdscr):
    char = unicurses.getkey()
    if char in 'hqxsak':
        return char
    else:
        return get_input(stdscr)


def initscr():
    stdscr = unicurses.initscr()
    unicurses.noecho()
    unicurses.cbreak()
    unicurses.keypad(stdscr, True)
    return stdscr


def display_text(stdscr, text):
    unicurses.clear()
    unicurses.addstr(text)
    unicurses.refresh()
    return None





def killscr(stdscr):
    unicurses.echo()
    unicurses.nocbreak()
    unicurses.keypad(stdscr, False)
    unicurses.endwin()
    return None
