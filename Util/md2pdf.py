# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time

def main(args):
    start_time = time.time()
    os.system("pandoc {} -o {} --latex-engine=xelatex -V mainfont='WenQuanYi Zen Hei'".format(args.input_file,args.output_file))

    print("Transform Complete within {} sec.".format(int(time.time()-start_time)))
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",type=str,
            help="input file name.")
    parser.add_argument("output_file",type=str,
            help="output file name")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
