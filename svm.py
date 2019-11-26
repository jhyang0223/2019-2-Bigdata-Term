import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)



    


    #CList = [10, 20,30,40,50,60,70,80,90,100]
    CList = [60]
    bestScore = 0
    c_opt = 0
    for c in CList:
      svm = SVC(kernel = 'linear',C = c, random_state = 0)

      kfold = KFold(n_splits=10)
      scores = cross_val_score(svm, cancer.data, cancer.target,cv=kfold)
      
      score = scores.mean()
      if score > bestScore :
        bestScore = score
        c_opt = c
    
 #       forest.fit(x_train,y_train)

#    svm = SVC(kernel = 'linear',C = c_opt, random_state = 0)
#    svm = SVC(kernel = 'rbf'',C = c_opt, random_state = 0,gamma = "auto")
    svm = SVC(kernel = 'sigmoid',C = c_opt, random_state = 0, gamma = "auto")

    kfold = KFold(n_splits=10)
    scores = cross_val_score(svm, cancer.data, cancer.target,cv=kfold)
    
    print("optimized hyperparamsetting")
    print("c:",c_opt)
    print("best_score_mean:",scores.mean())


