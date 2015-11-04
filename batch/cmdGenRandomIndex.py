
"""
    generate train/dev/test idx.pkl for LJ2M dataset
"""

import os
import sys

import argparse
import pickle
import logging
import random

from common import utils

def get_arguments(argv):

    parser = argparse.ArgumentParser(description='generate train/dev/test idx.pkl for LJ2M dataset')

    parser.add_argument('corpus_folder', metavar='corpus_folder', 
                        help='corpus folder which should be structured like LJ2M')  
    parser.add_argument('percent_train', metavar='percent_train', type=int,
                        help='percentage of training data')    
    parser.add_argument('percent_dev', metavar='percent_dev', type=int,
                        help='percentage of development data')   
    parser.add_argument('percent_test', metavar='percent_test', type=int,
                        help='percentage of testing data')  
    parser.add_argument('output_filename', metavar='output_filename', 
                        help='output file name')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')

    args = parser.parse_args(argv)
    return args

def get_and_check_files(corpus_folder):
    '''
        get number of files in each emotion folder
        it should be the same for all emotions
    '''

    nDoc = None
    emotion_dirs = os.listdir(corpus_folder)
    for emotion in emotion_dirs:
        emotion_dir = os.path.join(corpus_folder, emotion)
        files = [name for name in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, name))]
        files = [int(fn) for fn in files if utils.represents_int(fn)]

        is_diff = [i for i, j in zip(sorted(files), range(len(files))) if i != j]

        assert len(is_diff) == 0

        if nDoc is None:
            nDoc = len(files)
        else:
            assert nDoc == len([name for name in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, name))])

    return nDoc

if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)


    # get number of files in each emotion folder
    n_doc = get_and_check_files(args.corpus_folder)

    n_train = n_doc * args.percent_train / 100;
    n_dev = n_doc * args.percent_dev / 100;
    n_test = n_doc * args.percent_test / 100;

    # remaining as training data
    n_train += (n_doc - n_train - n_dev - n_test)

    random_list = range(n_doc)
    random.shuffle(random_list)

    idx_dict = {}
    idx_dict['train'] = sorted(random_list[0:n_train])
    idx_dict['dev'] = sorted(random_list[n_train:n_train+n_dev])
    idx_dict['test'] = sorted(random_list[n_train+n_dev:n_train+n_dev+n_test])


    logger.info("dumping file to %s" % (args.output_filename))
    utils.save_pkl_file(idx_dict, args.output_filename)
    
