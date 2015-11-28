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
    parser.add_argument('file_prefix', metavar='file_prefix', 
                        help='prefix for each emotion file name')
    parser.add_argument('-t', '--temp_output_dir', metavar='TEMP_DIR', default=None, 
                        help='output intermediate data of each emotion in the specified directory (DEFAULT: not output)')

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

    if args.temp_output_dir is not None and os.path.exists(args.temp_output_dir):
        raise Exception("folder %s is already existed." % (args.temp_output_dir))
    elif args.temp_output_dir is not None and not os.path.exists(args.temp_output_dir):
        logger.info('create output folder %s' % (args.temp_output_dir))
        os.makedirs(args.temp_output_dir)

    train_file_paths = utils.get_paths_by_re(features[0]['train_dir'], features[0]['train_filename'])
    test_file_paths = utils.get_paths_by_re(features[0]['test_dir'], features[0]['test_filename'])

    n_emotions = len(emotions)
    X_train_pos = [None] * n_emotions
    for emotion_id in range(n_emotions):
        X_train_pos[emotion_id] = cPickle.load(open(train_file_paths[emotion_id]))


    # get negative examples
    n_pos = X_train_pos[0].shape[0]
    X_train_neg = [None] * n_emotions
    avg_floor = n_pos/(n_emotions-1)

    for emotion_id in range(n_emotions):

        X_train_neg[emotion_id] = np.zeros(X_train_pos[0].shape)
        neg_emotion_ids = [i for i in range(n_emotions) if i != emotion_id]
        cnt_neg = 0
        for neg_emotion_id in neg_emotion_ids:

            # tricky rules
            if neg_emotion_id == neg_emotion_ids[-1]:
                n_neg = n_pos - cnt_neg
            else:    
                n_neg = avg_floor if (neg_emotion_id%2 == 0) else avg_floor+1

            X_train_neg[emotion_id][range(cnt_neg, cnt_neg+n_neg)] = X_train_pos[neg_emotion_id][range(n_neg)]
            cnt_neg += n_neg

        assert cnt_neg == n_pos

    

    # dump files
    for emotion_id in range(n_emotions):

        X_train = np.concatenate((X_train_pos[emotion_id], X_train_neg[emotion_id]), axis=0)
        y_train = np.concatenate((np.ones(n_pos), np.zeros(n_pos)), axis=1)

        fname = "%s%d.npz" % (args.file_prefix, emotion_id)
        fpath = os.path.join(args.temp_output_dir, fname)
        np.savez_compressed(fpath, X=X_train, y=y_train)
    





