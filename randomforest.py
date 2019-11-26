import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()

    x = cancer.data
    y = cancer.target

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

    nestList = [10,20,30,40,50,60,70,80,90,100]
    maxFeatList = [5,10,15,20,25,30]
    maxDepthList = [10,20,30,40,50,60,70,80]
    bestScore = 0
    n_opt=0
    m_opt=0
    d_opt=0
    for n in nestList:
      for m in maxFeatList:
        for d in maxDepthList:
          forest = RandomForestClassifier(n_estimators=n,max_features=m,max_depth=d)
          kfold = KFold(n_splits=10)
          scores = cross_val_score(forest, cancer.data, cancer.target,cv=kfold)
          score = scores.mean()
          if score > bestScore :
            bestScore = score
            n_opt = n
            m_opt = m
            d_opt = d
 #       forest.fit(x_train,y_train)

    forest = RandomForestClassifier(n_estimators=n_opt,max_features=m_opt)
    kfold = KFold(n_splits=10)
    scores = cross_val_score(forest, cancer.data, cancer.target,cv=kfold)
    
    print("optimized hyperparamsetting")
    print("n_estimators:",n_opt,"max_features:",m_opt,"max_depth:",d_opt)
    print("best_score_mean:",scores.mean())

    forest.fit(x_train,y_train)
    n_feature = cancer.data.shape[1]
    index = np.arange(n_feature)

    plt.barh(index,forest.feature_importances_,align="center")
    plt.yticks(index, cancer.feature_names)
    plt.ylim(-1, n_feature)
    plt.xlabel('feature importance', size=15)
    plt.ylabel('feature', size=15)
    plt.show()

    for importance in forest.feature_importances_ :
      print(importance)
