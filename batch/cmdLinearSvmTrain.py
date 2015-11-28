import sys
import argparse
import os
import csv
import operator
import logging

import pickle
import numpy as np
from sklearn.cross_validation import KFold

from common import utils
from common import filename
from feelwords.learners.svm import LinearSvm
from feelwords.features.preprocessing import DataPreprocessor

emotions = filename.emotions['LJ40K']


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='perform Linear SVM training for LJ40K')
    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('-k', '--kfold', metavar='NFOLD', type=int, default=10, 
                        help='k for kfold cross-validtion. If the value less than 2, we skip the cross-validation and choose the first parameter of -c and -g (DEFAULT: 10)')
    parser.add_argument('-o', '--output_file_name', metavar='OUTPUT_NAME', default='out.csv', 
                        help='path to the output file in csv format (DEFAULT: out.csv)')
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-c', metavar='C', type=utils.parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-t', '--temp_output_dir', metavar='TEMP_DIR', default=None, 
                        help='output intermediate data of each emotion in the specified directory (DEFAULT: not output)')
    parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
                        help='do not perform feature scaling (DEFAULT: False)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def collect_results(all_results, emotion, results):
    all_results['emotion'].append(emotion)
    all_results['weighted_score'].append(results['weighted_score'])
    all_results['auc'].append(results['auc'])
    all_results['X_predict_prob'].append(results['X_predict_prob'])
    return all_results

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

    # some pre-checking
    if args.temp_output_dir is not None and os.path.exists(args.temp_output_dir):
        raise Exception("folder %s is already existed." % (args.temp_output_dir))
    elif args.temp_output_dir is not None and not os.path.exists(args.temp_output_dir):
        logger.info('create output folder %s' % (args.temp_output_dir))
        os.makedirs(args.temp_output_dir)

    if os.path.exists(args.output_file_name):
        raise Exception("file %s is already existed." % (args.output_file_name))

    if not utils.test_writable(args.output_file_name): 
        raise Exception("file %s is not writable." % (args.output_file_name))

    train_file_paths = utils.get_paths_by_re(features[0]['train_dir'], features[0]['train_filename'])
    test_file_paths = utils.get_paths_by_re(features[0]['test_dir'], features[0]['test_filename'])
    import pdb; pdb.set_trace()

    # main loop
    collect_best_param = {}   # TODO: remove
    all_results = {'emotion': ['Evals'], 'weighted_score': ['Accuracy Rate'], 'auc': ['AUC'], 'X_predict_prob': []}
    for emotion_id in args.emotion_ids:    
        
        emotion_name = emotions[emotion_id]
        paths = utils.get_paths_by_emotion(features, emotion_name)

        ## prepare data
        preprocessor = DataPreprocessor(loglevel=loglevel)
        preprocessor.loads([f['feature'] for f in features], paths)
        X_train, y_train, feature_name = preprocessor.fuse()

        ## set default gamma for SVM           
        if not args.gamma:
            args.gamma = [1.0/X_train.shape[1]]
                
        learner = LinearSvm(loglevel=loglevel) 
        learner.set(X_train, y_train, feature_name)

        ## setup a kFolder
        if args.kfold > 1:
            kfolder = KFold(n=X_train.shape[0], n_folds=args.kfold, shuffle=True)
        
            ## do kfold with Cs and gammas
            scores = {}
            for svmc in args.c:
                score = learner.kfold(kfolder, prob=False, C=svmc, scaling=(not args.no_scaling))
                scores.update({svmc: score})

            if args.temp_output_dir:
                fpath = os.path.join(args.temp_output_dir, 'scores_%s.csv' % emotion_name)
                utils.dump_dict_to_csv(fpath, scores)

            ## get best parameters
            best_C = max(scores.iteritems(), key=operator.itemgetter(1))[0]

            ## collect misc
            collect_best_param.update({emotion_name: (best_C)}) 

        else:   # we choose first parameters if we do not perfrom cross-validation
            best_C = args.c[0] if args.c else 1.0


        ## ---------------------------------------------------------------------------
        ## train all data
        learner.train(prob=True, C=best_C, scaling=(not args.no_scaling), random_state=np.random.RandomState(0))

        ## prepare testing data
        paths = [f['test_file'] for f in features]
        preprocessor.clear()
        preprocessor.loads([f['feature'] for f in features], paths)
        X_test, y_test, feature_name = preprocessor.fuse()
        
        yb_test = preprocessor.get_binary_y_by_emotion(y_test, emotion_name)
        results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)

        ## collect results
        all_results = collect_results(all_results, emotion_name, results)
        if args.temp_output_dir:
            fpath = os.path.join(args.temp_output_dir, "model_%s_%f.pkl" % (emotion_name, best_C));
            learner.dump_model(fpath);
            if not args.no_scaling:
                fpath = os.path.join(args.temp_output_dir, "scaler_%s.pkl" % (emotion_name));
                learner.dump_scaler(fpath);

    if args.temp_output_dir:
        fpath = os.path.join(args.temp_output_dir, 'best_param.csv')
        utils.dump_dict_to_csv(fpath, collect_best_param)
        fpath = os.path.join(args.temp_output_dir, 'X_predict_prob.csv')    
        utils.dump_list_to_csv(fpath, all_results['X_predict_prob'])

    utils.dump_list_to_csv(args.output_file_name, [all_results['emotion'], all_results['weighted_score'], all_results['auc']])   

