import BagLearner as bl, LinRegLearner as lrl, DTLearner as dtl, RTLearner as rtl, numpy as np
class InsaneLearner(object):
    def __init__(self, verbose = False):
        learners = []
        num = 20
        self.verbose = verbose
        for i in range(num):
            learners.append(bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags = 20, verbose = self.verbose))
        self.learners = learners
        self.num = num
    def addEvidence(self, Xtrain, Ytrain):
        for learner in self.learners:
            learner.addEvidence(Xtrain, Ytrain)
    def query(self, Xtest):
        end = np.empty((Xtest.shape[0], self.num))
        for col in range(self.num):
            end[:,col] = self.learners[col].query(Xtest)
        return end.mean(axis = 1)
    def author(self): return 'vbehl3'
if __name__=="__main__": print ("This is an Insane Learner")