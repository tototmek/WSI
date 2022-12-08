import statistics
import numpy as np
import pandas as pd
import math

# Klasa reprezentująca węzeł drzewa decyzyjnego
class Node:
    def __init__(self,
                 value=None,
                 root_value=None,
                 decision_param=None,
                 children=[]):

        self.children = children
        self.decision_param = decision_param
        self.root_value = root_value
        self.value = value
    
    # Funkcja zwracająca wartość decyzyjną dla podanego przykładu
    def output(self, x):
        for child in self.children:
            if x[self.decision_param] == child.root_value:
                return child.output(x)
        return self.value
    
    # Funkcja wypisująca fragment logiki drzewa decyzyjnego
    def print(self, depth):
        if depth > 0:
            print( " " * 4*depth + f"case {self.root_value}:")
        if len(self.children) == 0:
            print( " " * (4*depth+2) + self.value)
            return
        print( " " * (4*depth+2) + f"switch ({self.decision_param}):")
        for child in self.children:
            child.print(depth+1)

# Klasa reprezentująca drzewo decyzyjne
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    # Funkcja budująca drzewo decyzyjne
    def fit(self, X, y):
        self.params = list(X.columns)
        self.classes = list(y.unique())
        self.tree = ID3(
                        self.classes,
                        self.params,
                        X,
                        y,
                        self.max_depth
                        )
    
    # Funkcja zwracająca wartość decyzyjną dla podanego przykładu
    def predict(self, X):
        if self.tree is None:
            raise Exception("Call fit() method before predict()!")
        result =[]
        for i in range(len(X)):
            x = X.iloc[i]
            result.append(self.tree.output(x))
        return pd.Series(result)
    
    # Funkcja wypisująca logikę drzewa decyzyjnego
    def print(self):
        self.tree.print(0)

# Implementacja algorytmu ID3 do budowy drzewa decyzyjnego
def ID3(Y, D, X, y, depth, root_value=None):
    for classname in Y:
        if all([y_i == classname for y_i in y]):
            return Node(value=classname, root_value=root_value)
    if len(D) == 0 or depth == 0:
        return Node(value=statistics.mode(y), root_value=root_value)
    infgains = [InfGain(Y, d, X, y) for d in D]
    d = D[np.argmax(infgains)]
    D.remove(d)
    d_values = list(X[d].unique())
    return Node(root_value=root_value, decision_param=d, children=[
        ID3(Y,
            D,
            X.loc[X[d]==value],
            y.loc[X[d]==value],
            depth-1,
            root_value=value)
        for value in d_values])


# Wyznaczanie zdobyczy informacyjnej
def InfGain(Y, d, X, y):
    return I(Y, X, y) - Inf(Y, d, X, y)

def Inf(Y, d, X, y):
    result = 0
    d_values = list(X[d].unique())
    for value in d_values:
       X_i, y_i = X.loc[X[d]==value], y.loc[X[d]==value]
       result += I(Y, X_i, y_i) * len(X_i) / len(X)
    return result

def I(Y, X, y):
    result = 0
    for classname in Y:
        f = len(y.loc[y == classname])
        result -= f * math.log(f+1e-10)
    return result

    