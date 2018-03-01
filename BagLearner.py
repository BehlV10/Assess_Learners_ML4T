import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LinRegLearner as lrl, DTLearner as dtl, RTLearner as rtl

class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.verbose = verbose
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        if verbose:
            self.get_learner_info()

    def addEvidence(self, dataX, dataY):
        samples = dataX.shape[0] #Rows
        for learner in self.learners:
            rand_sample = np.random.choice(samples, samples)
            bagX = dataX[rand_sample]
            bagY = dataY[rand_sample]
            learner.addEvidence(bagX, bagY)
        
    def query(self, points):
        labels = np.array([learner.query(points) for learner in self.learners])
        return np.mean(labels, axis=0)

    def get_learner_info(self):
        print ("bags =", self.bags)
        print ("kwargs =", self.kwargs)
        print ("boost =", self.boost)
    
    def author(self):
        return 'vbehl3'

if __name__=="__main__":
    print ("This is a Bag Learner")