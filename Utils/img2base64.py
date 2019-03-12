# -*- coding: utf-8 -*-
import base64
import os
import sys
import argparse
import time

def main(args):
    start_time = time.time()
    with open(args.input_file,"rb") as f:
        basedata = base64.b64encode(f.read())
    print(basedata)
    print("Transform Complete within {} sec.".format(int(time.time()-start_time)))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",type=str,
            help="input file name.")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
