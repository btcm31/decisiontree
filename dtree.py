import numpy as np
import pandas as pd
import math

class Node:
    #label: The value of feature that its parent node split.
    #attr: Best feature that node use to split the dataset.
    #children: List of children nodes of current node.
    #depth: depth of node.
    def __init__(self, label=None, attr=None, children = [], depth = 0):
        self.attr = attr
        self.label = label
        self.children = children
        self.depth = depth

class DecisionTree:
    #criterion: The function to measure the quality of a split. "gini" for the Gini impurity and "entropy" for the information gain.
    #max_depth: The maximum depth of the tree. Defaule None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    #min_samples_split: The minimum number of samples required to split an internal node.
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None #root node of tree
    def _gini(self, attrs, label):
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

    def _entropy_calculate(self, lst):
        #Calculate entropy of a feature.
        s = sum(lst)
        entropy = 0.0
        for i in lst:
            entropy -= i/s * math.log2(i/s)
        return entropy
    def _informationGain(self, attrs, label):
        #Measure the decrease of entropy
        uniqueValueofAttr = list(set(attrs))
        uniqueLabel = list(set(label))
        entropy = self._entropy_calculate([list(label).count(i) for i in uniqueLabel])

        for val in uniqueValueofAttr:
            temp = attrs[attrs.loc[:] == val]
            lb_pervalue = label.loc[temp.index]
            entropy_pervalue = self._entropy_calculate([list(lb_pervalue).count(i) for i in list(set(lb_pervalue))])

            entropy -= list(attrs).count(val) / len(attrs) * entropy_pervalue

        return entropy
    def _pruning(self):
        pass

    def _maketree(self, data, target, depth, label):
        lstAttrs = list(data)
        best_attr = None
        children = []
        if len(lstAttrs) == 1:
            return Node(label, attr=lstAttrs[0], children=[Node(i, None ,children=[target[data[lstAttrs[0]].index].values[0]], depth=depth+1) for i in data[lstAttrs[0]].unique()], depth = depth)

        if self.criterion == 'gini':
            bestgini = 1.0
            #Calculate Gini for each feature and choose feature with minimum Gini Index
            for i in lstAttrs:
                gini = self._gini(data[i],target)
                if gini <= bestgini:
                    best_attr = i
                    bestgini = gini
            ''' if bestgini == 0.0:
                print(target.unique())
                return target.unique()[0] '''
        else:
            bestIG = 0.0
            #Calculate InformationGain for each feature choose feature with maximum Information Gain
            for i in lstAttrs:
                ig = self._informationGain(data[i],target)
                if ig >= bestIG:
                    best_attr = i
                    bestIG = ig
            ''' if bestIG == 1.0:
                print(target.unique())
                return target.unique()[0] '''

        #Split data into subset
        for i in data[best_attr].unique():
            subdata = data[data[best_attr]==i].drop([best_attr], axis=1)
            subtarget = target.loc[subdata.index]
            temp = self._maketree(subdata, subtarget, depth+1, i)
            children.append(temp if type(temp) == Node else Node(i, None, [temp], depth+1))

        return Node(label, best_attr, children, depth)
    def fit(self, data, target):
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        self.root = self._maketree(data, target, 0, 'root')
    def predict(self, new_data):
        npoints = new_data.count()[0]
        labels = []
        for n in range(npoints):
            x = new_data.iloc[n, :]
            node = self.root
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
    clf.fit(X,y)
    newdata = pd.read_csv('predict.csv')
    print(clf.predict(newdata))


