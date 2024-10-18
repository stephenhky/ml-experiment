
import argparse
import os
import json
import configparser
from functools import partial

import pandas as pd
from tqdm import tqdm

from .data.adding_features import adding_no_features, convert_label_to_str
from .experiment import add_multiple_features
from .experiment import run_experiment


def get_csv2json_argparser():
    argparser = argparse.ArgumentParser(description='Convert CSV to JSON.')
    argparser.add_argument('csvfile', help='Path of .CSV file.')
    argparser.add_argument('jsonfile', help='Path of output .JSON file.')
    argparser.add_argument('--delimiter', default=None, help='delimiter (default: None (standing for ",")')
    argparser.add_argument('--header', nargs='+', help='Headers (optional)', default=None)
    return argparser


def csv2json():
    args = get_csv2json_argparser().parse_args()

    print('Reading {}...'.format(args.csvfile))
    print('Delimiter = {}'.format(args.delimiter))
    if args.header is None:
        df = pd.read_csv(args.csvfile, delimiter=args.delimiter, engine='python')
    else:
        df = pd.read_csv(args.csvfile, delimiter=args.delimiter, header=None, engine='python')
        df.columns = args.header

    outfile = open(args.jsonfile, 'w')
    nbrows = len(df)
    for rowid in tqdm(range(nbrows)):
        jsonstr = df.iloc[rowid].to_json()
        outfile.write(jsonstr+'\n')

    outfile.close()


def get_runexperiment_argparser():
    argparser = argparse.ArgumentParser(description='ML Experiment Framework')
    argparser.add_argument('configfile', help='Configuration file')
    argparser.add_argument('--overridingh5dir', default=None, type=str,
                           help='Overriding directory for intermediate numerical HDF files.')
    return argparser


def run_experiment():
    args = get_runexperiment_argparser().parse_args()

    # reading config file (.ini or .json)
    file_ext = os.path.splitext(args.configfile)[1]
    if file_ext.lower() == '.json':
        config = json.load(open(args.configfile, 'r'))
    elif file_ext == '.ini':
        config = configparser.ConfigParser()
        config.read(args.configparser)
    else:
        raise ValueError('The configuration file should have an extention ' +
                         'be either ".ini" or ".json", not "{}".')

    # identify label
    label = config['model']['target']

    # defining feature addition (can be configured)
    feature_adder = add_multiple_features([adding_no_features,
                                           partial(convert_label_to_str, label=label)])
    run_experiment(config,
                   feature_adder=feature_adder,
                   overriding_h5dir=args.overridingh5dir)

