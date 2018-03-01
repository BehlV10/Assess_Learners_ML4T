import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from operator import itemgetter
from copy import deepcopy

class DTLearner(object): 
    def __init__(self, leaf_size=1, verbose=False, tree=None):
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
        if ((np.all(dataY==dataY[0])) | np.all(dataX==dataX[0,:])): return leaf
        split_features = range(features) #List?
        abs_corr = np.abs(np.corrcoef(dataX, y=dataY, rowvar=False))[:-1,-1]
        max_corr = np.nanargmax(abs_corr)
        split_val = np.median(dataX[:,max_corr])
        smaller_data = dataX[:,max_corr] <= split_val
        
        if ((np.all(smaller_data)) or np.all(~smaller_data)):
            return leaf
        
        lefttree = self.build_tree(dataX[smaller_data,:], dataY[smaller_data])
        righttree = self.build_tree(dataX[~smaller_data,:], dataY[~smaller_data])
        
        if lefttree.ndim == 1:
            righttree_start = 2
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([max_corr, split_val, 1, righttree_start])

        return np.vstack((root, lefttree, righttree))
            
    def addEvidence(self, dataX, dataY):
    
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:,0:dataX.shape[1]] = dataX
        self.model_coefs, self.residuals, self.rank, self.s = np.linalg.lstsq(newdataX, dataY)
        newTree = self.build_tree(dataX, dataY)
        if (self.tree == None):
            self.tree = newTree
        else:
            self.tree = np.vstack((self.tree,newTree))
            
        #Line below added because of indices to small issue
        if (len(self.tree.shape) == 1): #If tree shape is too small, expand
            self.tree = np.expand_dims(self.tree, axis=0)

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
    print ("This is a DT Learner")