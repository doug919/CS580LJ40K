
import cPickle as pickle
import logging
import numpy as np
import json

def sigmoid(x):
    neg_x = map(lambda v: v*(-1), x)
    return 1 / (1 + np.exp(neg_x))

def save_pkl_file(clz, filename):
    try:
        pickle.dump(clz, open(filename, "w"))
    except ValueError:
        logging.error("failed to dump %s" % (filename))

def load_pkl_file(filename):
    try:
        pkl = pickle.load(open(filename, "r"))
        return pkl
    except ValueError:
        logging.error("failed to load %s" % (filename))

def get_unique_list_diff(a, b):
    return list(set(a) - set(b))

def parse_range(astr):
    result = set()
    for part in astr.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(float(part))
    return sorted(result)

def test_writable(file_path):
    writable = True
    try:
        filehandle = open(file_path, 'w')
    except IOError:
        writable = False
        
    filehandle.close()
    return writable

def read_parameter_file(file_path):
    """
        this is a temporary solution
        ToDo: generalize
    """
    param_dict = {}
    with open(file_path) as f:
        for line in f:

            toks = line.strip().split(',')
            # only work in python 2.x
            toks[1] = float(toks[1].translate(None, '"()'))
            toks[2] = float(toks[2].translate(None, '"()'))

            param_dict[toks[0]] = toks[1], toks[2]

    return param_dict

def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def get_feature_list(feature_list_file):
    fp = open(feature_list_file, 'r')
    feature_list = json.load(fp)
    fp.close()
    return feature_list

def get_file_name_by_emtion(train_dir, emotion, **kwargs):
    '''
    serach the train_dir and get the file name with the specified emotion and extension
    '''
    ext = '.npz' if 'ext' not in kwargs else kwargs['ext']
    files = [fname for fname in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, fname))]

    # target file is the file that contains the emotion string and has the desginated extension
    for fname in files:
        target = None
        if fname.endswith(ext) and fname.find(emotion) != -1:
            target = fname
            break
    return target

def get_paths_by_emotion(features, emotion_name):
    paths = []
    for feature in features:         
        fname = get_file_name_by_emtion(feature['train_dir'], emotion_name, exp='.npz')
        if fname is not None:
            paths.append(os.path.join(feature['train_dir'], fname))
    return paths


def dump_dict_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for key, val in data.items():
        w.writerow([key, val])

def dump_list_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for row in data:
        w.writerow(row)

