

from sklearn import preprocessing
from sklearn.svm import LinearSVC

from .base import LearnerBase


class LinearSvm(LearnerBase):

    def __init__(self, **kwargs):
        super(LinearSvm, self).__init__(**kwargs)

    def train(self, **kwargs):
        assert self.X is not None
        assert self.y is not None
        self._train(self.X, self.y, **kwargs)

    def _train(self, X_train, y_train, **kwargs):
        with_mean = True if 'with_mean' not in kwargs else kwargs['with_mean']
        with_std = True if 'with_std' not in kwargs else kwargs['with_std']
        self.scaling = True if 'scaling' not in kwargs else kwargs['scaling']
        C = 1.0 if "C" not in kwargs else kwargs["C"]

        self.logger.debug("%d samples x %d features in X_train" % (X_train.shape[0], X_train.shape[1]))
        self.logger.debug("%d samples in y_train" % (y_train.shape[0]))

        if self.scaling:
            self.scaler = preprocessing.StandardScaler(with_mean=with_mean, with_std=with_std)
            self.logger.debug("applying a standard scaling with_mean=%d, with_std=%d" % (with_mean, with_std))
            X_train = self.scaler.fit_transform(X_train)

        self.clf = LinearSVC(C=C)
        self.logger.debug("C=%f" % (C))

        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test):
        if self.scaling:
            X_test = self.scaler.transform(X_test)
        return self.clf.predict(X_test)

    def score(self, X_test, y_test, **kwargs):
        if self.scaling:
            X_test = self.scaler.transform(X_test)

        self.logger.info('y_test = %s', str(y_test.shape))
        y_predict = self.clf.predict(X_test)

        results = {}
        if 'accuracy' in kwargs and kwargs['accuracy'] == True:
            results.update({'accuracy': self.clf.score(X_test, y_test.tolist())})
            self.logger.info('accuracy = %f', results['accuracy'])

        # if 'weighted_score' in kwargs and kwargs['weighted_score'] == True:
        #     results.update({'weighted_score': self._weighted_score(y_test.tolist(), y_predict)})
        #     self.logger.info('weighted_score = %f', results['weighted_score'])

        if 'decision_value' in kwargs and kwargs['decision_value'] == True:
            results.update({'decision_value': self.clf.decision_function(X_test)})
            self.logger.debug('decision_value = %s', str(results['decision_value']))

        if 'f1' in kwargs and kwargs['f1'] == True:
            _prec = LearnerBase.precision(y_predict, y_test)
            _recall = LearnerBase.recall(y_predict, y_test)
            _f1 = LearnerBase.f1_score(_prec, _recall)
            results.update({'precision': _prec})
            self.logger.info('precision = %s', str(results['precision']))
            results.update({'recall': _recall})
            self.logger.info('recall = %s', str(results['recall']))
            results.update({'f1': _f1})
            self.logger.info('f1 = %s', str(results['f1']))

        return results     
    
    # def _weighted_score(self, y_test, y_predict):
    #     # calc weighted score 
    #     n_pos = len([val for val in y_test if val == 1])
    #     n_neg = len([val for val in y_test if val == -1])
        
    #     temp_min = min(n_pos, n_neg)
    #     weight_pos = 1.0/(n_pos/temp_min)
    #     weight_neg = 1.0/(n_neg/temp_min)
        
    #     correct_predict = [i for i, j in zip(y_test, y_predict) if i == j]
    #     weighted_sum = 0.0
    #     for answer in correct_predict:
    #         weighted_sum += weight_pos if answer == 1 else weight_neg
        
    #     wscore = weighted_sum / (n_pos * weight_pos + n_neg * weight_neg)
    #     return wscore
    
    def kfold(self, kfolder, **kwargs):
        metric = 'accuracy' if 'metric' not in kwargs else kwargs['metric']
        self.logger.info("cross-validation metric = %s", metric)

        sum_score = 0.0
        for (i, (train_index, test_index)) in enumerate(kfolder):

            self.logger.info("cross-validation fold %d: train=%d, test=%d" % (i, len(train_index), len(test_index)))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            self._train(X_train, y_train, **kwargs)

            if metric == 'accuracy':
                sc = self.score(X_test, y_test, accuracy=True)['accuracy']
                self.logger.info('accuracy = %.5f' % (sc))
            else:
                # use f1
                sc = self.score(X_test, y_test, f1=True)['f1']
                self.logger.info('f1 = %.5f' % (sc))
            sum_score += sc

        mean_score = sum_score/len(kfolder)
        self.logger.info('*** C = %f, mean_score = %f' % (kwargs['C'], mean_score))
        return mean_score


class KernelSvm(LearnerBase):

    def __init__(self, **kwargs):
        super(LinearSvm, self).__init__(**kwargs)

    # ToDo: implement


    


