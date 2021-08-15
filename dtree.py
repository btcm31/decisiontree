import numpy as np
import pandas as pd
import math

class Node:
    def __init__(self, label=None, attr=None, children = [], depth = 0):
        self.attr = attr
        self.label = label
        self.children = children
        self.depth = depth
class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2):
        '''
        criterion: ['gini', 'entropy']
            + gini: CART
            + entropy: ID3 (Iterative Dichotomiser 3)
        '''
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    def _gini(self, attrs, label):
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
        s = sum(lst)
        entropy = 0.0
        for i in lst:
            entropy -= i/s * math.log2(i/s)
        return entropy
    def _informationGain(self, attrs, label):
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
            return Node(label, attr=None, children=[Node(i, None ,children=[target[data[i].index].values[0]], depth=depth+1) for i in lstAttrs], depth = depth)

        if self.criterion == 'gini':
            bestgini = 1.0
            for i in lstAttrs:
                gini = self._gini(data[i],target)
                if gini <= bestgini:
                    best_attr = i
                    bestgini = gini
            if bestgini == 0.0:
                return label.unique()[0]
        else:
            bestentropy = 0.0

            for i in lstAttrs:
                en = self._informationGain(data[i],target)
                if en >= bestentropy:
                    best_attr = i
                    bestentropy = en
            if bestentropy == 1.0:
                return label.unique()[0]
        for i in data[best_attr].unique():
            subdata = data[data[best_attr]==i].drop([best_attr], axis=1)
            subtarget = target.loc[subdata.index]
            children.append(self._maketree(subdata, subtarget, depth+1, i))

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
            x = new_data.iloc[n, :] # one point 
            # start from root and recursively travel if not meet a leaf 
            node = self.root
            while node.attr: 
                lb = new_data.iloc[n][node.attr]
                node = [i for i in node.children if i.label == lb][0]

            labels.append(node.children[0].children[0])
            
        return labels


if __name__ == "__main__":
    data = pd.read_csv('weather.csv')
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    clf = DecisionTree()
    clf.fit(X,y)
    newdata = pd.read_csv('pre.csv')
    print(clf.predict(newdata))


