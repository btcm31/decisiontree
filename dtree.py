import numpy as np
import pandas as pd
import math
import time

class Node:
    """ 
    - Node in Decision Tree
    - label: the value of feature that its parent node split.
    - attr: best feature that node use to split the dataset.
    - children: list of children nodes of current node.
    - depth: depth of node. 
    """
    def __init__(self, label=None, attr=None, children = [], depth = 0):
        self.attr = attr
        self.label = label
        self.children = children
        self.depth = depth

class DecisionTree:
    """
    - criterion: The function to measure the quality of a split. "gini" for the Gini impurity and "entropy" for the information gain.
    - max_depth: The maximum depth of the tree. Defaule None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    - min_samples_split: The minimum number of samples required to split an internal node.
    """
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2):
        if max_depth and type(max_depth) != int:
            raise TypeError("max_depth must be int, not %s" % (type(max_depth)))
        if type(min_samples_split) != int:
            raise TypeError("min_samples_split must be int, not %s" % (type(min_samples_split)))

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.__root = None #root node of tree
        self.__lstNode = []

    def __infor(self):
        print(f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split})")

    def __gini(self, attrs, label):
        #Calculate Gini Index of a feature.
        uniqueValueofAttr = list(set(attrs))
        gini = 0.0

        for val in uniqueValueofAttr:
            temp = attrs[attrs.loc[:] == val]
            lb_pervalue = label.loc[temp.index]
            gini_pervalue = 1.0

            for row in list(set(lb_pervalue)):
                gini_pervalue -= (list(lb_pervalue).count(row) / len(lb_pervalue)) ** 2
            gini += list(attrs).count(val) / len(attrs) * gini_pervalue

        return gini

    def __entropy_calculate(self, lst):
        #Calculate entropy of a feature.
        s = sum(lst)
        entropy = 0.0
        for i in lst:
            entropy -= i/s * math.log2(i/s)
        return entropy
    
    def __informationGain(self, attrs, label):
        #Measure the decrease of entropy
        uniqueValueofAttr = list(set(attrs))
        uniqueLabel = list(set(label))
        entropy = self.__entropy_calculate([list(label).count(i) for i in uniqueLabel])

        for val in uniqueValueofAttr:
            temp = attrs[attrs.loc[:] == val]
            lb_pervalue = label.loc[temp.index]
            entropy_pervalue = self.__entropy_calculate([list(lb_pervalue).count(i) for i in list(set(lb_pervalue))])

            entropy -= list(attrs).count(val) / len(attrs) * entropy_pervalue

        return entropy

    def __maketree(self, data, target, depth, label):
        lstAttrs = list(data)
        best_attr = None
        children = []
        if len(lstAttrs) == 1:
            return Node(label, attr=lstAttrs[0], children=[Node(i, None ,children=[target[data[lstAttrs[0]].index].values[0]], depth=depth+1) for i in data[lstAttrs[0]].unique()], depth = depth)

        if self.criterion == 'gini':
            bestgini = 1.0
            #Calculate Gini for each feature and choose feature with minimum Gini Index
            for i in lstAttrs:
                gini = self.__gini(data[i],target)
                if gini <= bestgini:
                    best_attr = i
                    bestgini = gini
        else:
            bestIG = 0.0
            #Calculate InformationGain for each feature choose feature with maximum Information Gain
            for i in lstAttrs:
                ig = self.__informationGain(data[i],target)
                if ig >= bestIG:
                    best_attr = i
                    bestIG = ig

        #Split data into subset
        for i in data[best_attr].unique():
            #get subdata corresponding to data[best_attr] = i
            subdata = data[data[best_attr]==i].drop([best_attr], axis=1)
            subtarget = target.loc[subdata.index]

            temp = None
            if (not self.max_depth or depth < self.max_depth) and self.min_samples_split <= subdata.shape[0]:
                temp = self.__maketree(subdata, subtarget, depth+1, i)
            else:
                #if depth == maxdepth, return target with the most frequency
                idx = data[data[best_attr]==i].index
                target_most = np.array(target.loc[idx].values)
                unique, counts = np.unique(target_most, return_counts = True)
                #get target with the most frequent
                temp = unique[np.argmax(counts)]
            children.append(temp if type(temp) == Node else Node(i, None, [temp], depth+1))

        return Node(label, best_attr, children, depth)

    def fit(self, data, target):
        self.attributes = list(data)
        self.labels = target.unique()

        self.__root = self.__maketree(data, target, 1, 'root')
        self.__infor()

    def predict(self, new_data):
        npoints = new_data.count()[0]
        labels = []
        for n in range(npoints):
            x = new_data.iloc[n, :]
            node = self.__root
            while type(node) != str and node.attr: 
                lb = new_data.iloc[n][node.attr]
                lst_child = [i for i in node.children if i.label == lb] 
                node = lst_child[0] if len(lst_child) > 0 else node.children[0]

            labels.append(node if type(node) == str else node.children[0])
            
        return labels


if __name__ == "__main__":
    data = pd.read_csv('weather.csv')
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]

    clf = DecisionTree()
    starttime = time.time()
    clf.fit(X,y)
    print("Training time: %s" % (time.time() - starttime))
    
    newdata = pd.read_csv('predict.csv')
    start_predicttime = time.time()
    ypred = clf.predict(newdata)
    print(ypred)
    print("Predict time: %s" % (time.time() - start_predicttime))


