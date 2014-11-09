#!/usr/bin/python

#
# Creates a JSON object with the name of each document groupas a key and a 
# correspoding random rgb color: as the value.
#

import json
import csv
import random
from colorsys import hsv_to_rgb
from random import randint, uniform

def main():
    csvfile = open('data.csv', 'r')
    jsonfile = open('data_names.json', 'w')
    names_with_color = {}
    r = lambda: random.randint(0,255)

    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0
    first_line = True
    for row in csvreader:
        if first_line:
            first_line = False
            continue 

        h = uniform(0.25, 0.38) # Select random green'ish hue from hue wheel
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)

        r, g, b = hsv_to_rgb(h, s, v)

        # Convert to 0-1 range for HTML output
        r, g, b = [x*255 for x in (r, g, b)]
        names_with_color[row[0]] = [randint(100,300),randint(0,100),randint(0,80)]
        count+= 1

    json.dump(names_with_color,jsonfile)


if __name__ == "__main__":
    main()
