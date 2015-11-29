import sys
import argparse
import os
import logging
import cPickle
import numpy as np

from common import utils
from common import filename

emotions = filename.emotions['LJ40K']

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='perform Linear SVM training for LJ40K')
    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('output_file', metavar='output_file', 
                        help='output file name')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    
    args = get_arguments(sys.argv[1:])
    features = utils.get_feature_list(args.feature_list_file)

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    test_file_paths = utils.get_paths_by_re(features[0]['test_dir'], features[0]['test_filename'])

    n_emotions = len(emotions)
    X_test_all = None
    y_test_all = None
    for emotion_id in range(n_emotions):
        emotion_name = emotions[emotion_id]
        X_tmp = cPickle.load(open(test_file_paths[emotion_id]))
        y_tmp = [emotion_name] * X_tmp.shape[0]
        
        if X_test_all is None:
            X_test_all = X_tmp
            y_test_all = y_tmp
        else:
            X_test_all = np.concatenate((X_test_all, X_tmp), axis=0)
            y_test_all = y_test_all + y_tmp

    y_test_all = np.array(y_test_all)
    logger.info("write file: %s" % args.output_file)
    np.savez_compressed(args.output_file, X=X_test_all, y=y_test_all)




