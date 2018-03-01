import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
from operator import itemgetter
from copy import deepcopy

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False, tree=None):
        self.leaf_size=leaf_size
        self.tree = deepcopy(tree)
        self.model_coefs = None
        self.residuals = None
        self.rank = None
        self.s = None
        self.verbose = verbose
        if verbose:
            self.get_learner_info()

    def build_tree(self, dataX, dataY):
        samples = dataX.shape[0] #Rows
        features = dataX.shape[1] #Columns
        #Leaf format is leaf, dataY, left child, right child
        leaf = np.array([-1, np.mean(dataY), np.nan, np.nan])
        if (samples <= self.leaf_size): return leaf
        if (len(pd.unique(dataY)) == 1): return leaf #Check if all values are equal

        p = 0;
        while (p < 10):
            rand_sample_i = [np.random.randint(0, samples), np.random.randint(0, samples)];
            rand_feature_i = np.random.randint(0, features);
            p += 1;
            if dataX[rand_sample_i[1], rand_feature_i] != dataX[rand_sample_i[0], rand_feature_i]:
                break

        if dataX[rand_sample_i[1], rand_feature_i] == dataX[rand_sample_i[0], rand_feature_i]:
            return leaf;

        split_val = (dataX[rand_sample_i[0], rand_feature_i] + dataX[rand_sample_i[1], rand_feature_i]) / 2;
        lefttree = self.build_tree(dataX[(dataX[:, rand_feature_i] <= split_val), :], dataY[(dataX[:, rand_feature_i] <= split_val)]);
        righttree = self.build_tree(dataX[(dataX[:, rand_feature_i] > split_val), :], dataY[(dataX[:, rand_feature_i] > split_val)]);

        # Setting the root according to the tree size
        if lefttree.ndim > 1:
            root = np.array([rand_feature_i, split_val, 1, lefttree.shape[0] + 1]);
        elif lefttree.ndim == 1:
            root = np.array([rand_feature_i, split_val, 1, 2]);

        # Return to a tree
        return np.vstack((root, lefttree, righttree));


    def addEvidence(self, dataX, dataY):
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:,0:dataX.shape[1]] = dataX
        self.model_coefs, self.residuals, self.rank, self.s = np.linalg.lstsq(newdataX, dataY)
        self.tree = self.build_tree(dataX, dataY)

    def query(self, points):
        trainY = []
        for num in points:
            trainY.append(self.tree_search(num, row_num=0))
        return np.asarray(trainY)

    def tree_search(self, num, row_num):
        feat, split_val = self.tree[row_num, 0:2]
        if feat == -1: #If leaf, return
            return split_val
        elif num[int(feat)] <= split_val: #If less, go left
            pred = self.tree_search(num, row_num + int(self.tree[row_num, 2]))
        else: #If more, go right
            pred = self.tree_search(num, row_num + int(self.tree[row_num, 3]))
        return pred

    def get_learner_info(self):
        print ("Model coefficient matrix:", self.model_coefs)
        print ("Sums of residuals:", self.residuals)
        print ("")
        
    def author(self):
        return 'vbehl3'
        
if __name__=="__main__":
    print ("This is a RT Learner")
