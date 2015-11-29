
import logging
import cPickle

class LearnerBase(object):

    def __init__(self, **kwargs):

        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 

        self.X = None
        self.y = None
        self.kfold_results = []
        self.Xs = {}
        self.ys = {}
        self.clf = None
        self.scaler = None

    def set(self, X, y, feature_name):
        self.X = X
        self.y = y
        self.feature_name = feature_name

    def dump(self, fname_model, file_scaler):
        self.dump_model(fname_model)
        self.dump_scaler(file_scaler)

    def dump_model(self, file_name):
        try:
            cPickle.dump(self.clf, open(file_name, "w"))
        except ValueError:
            self.logger.error("failed to dump %s" % (file_name))

    def dump_scaler(self, file_name):
        try:
            if self.scaling:
                cPickle.dump(self.scaler, open(file_name, "w"))
            else:
                self.logger.warning("scaler doesn't exist")
        except ValueError:
            self.logger.error("failed to dump %s" % (file_name))

    def load(self, fname_model, fname_scaler):
        self.load_model(fname_model)
        self.load_scaler(fname_scaler)

    def load_model(self, file_name):
        try:
            self.clf = cPickle.load( open(file_name, "r"))
        except ValueError:
            self.logger.error("failed to load %s" % (file_name))

    def load_scaler(self, file_name):
        try:
            self.scaler = cPickle.load( open(file_name, "r"))
            if self.scaler:
                self.scaling = True
        except ValueError:
            self.logger.error("failed to load %s" % (file_name))

    ##
    # get precision
    #
    # @param y_predict
    #               - predicted labels of testing data
    # @param y_test
    #               - labels of testing data
    #
    # @return None
    #
    @staticmethod
    def precision(y_predict, y_test):
        tp = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]>0]
        fp = [i for i in range(len(y_test)) if y_test[i]<=0 and y_predict[i]>0]

        #assert (len(tp)+len(fp) != 0)
        if len(tp)+len(fp) != 0:
            precision = float(len(tp)) / (len(tp)+len(fp))
        else:
            logging.debug("Both true positives and false positives are zero, so we cannot calculate the precision.")
            precision = -1
        return precision

    ##
    # get recall
    #
    # @param y_predict
    #               - predicted labels of testing data
    # @param y_test
    #               - labels of testing data
    #
    # @return None
    #
    @staticmethod
    def recall(y_predict, y_test):
        tp = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]>0]
        fn = [i for i in range(len(y_test)) if y_test[i]>0 and y_predict[i]<=0]
        
        #assert (len(tp)+len(fn) != 0)

        if len(tp)+len(fn) != 0:
            recall = float(len(tp)) / (len(tp)+len(fn))
        else:
            logging.debug("Both true positives and false negatives are zero, so we cannot calculate the precision.")
            recall = -1

        
        return recall

    ##
    # get F-1 score
    #
    # @param precision
    #               - calculated by precision()
    # @param recall
    #               - calculated by recall()
    #
    # @return None
    #
    @staticmethod
    def f1_score(precision, recall):
        return (2.0 * precision * recall) / (precision + recall)

    # abstract functions
    def train(self, **kwargs):
        pass

    def kfold(self):
        pass

    def predict(self):
        pass

    def score(self):
        pass
