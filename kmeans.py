import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
#import matplotlib.pyplot as plt
#import seaborn as sns
if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    
    X = cancer.data
    y = cancer.target
    xdf = pd.DataFrame(X)
    ydf = pd.DataFrame(y)
    for n in range(2,11):
      kmeans = KMeans(n_clusters=n,algorithm="auto")
      kmeans.fit(xdf)
      pred = kmeans.predict(xdf)
      predict = pd.DataFrame(pred)
      predict.columns = ['predict']
#      print(type(ydf.values))
#     print(ydf.values.tolist())
#     print(predict.values.tolist())

      print(n,"- Cluster Adjust Random Index : ",adjusted_rand_score(ydf.values.flatten(), predict.values.flatten()))
      print(n,"- Cluster Silhouette Score : ", metrics.silhouette_score (X, pred, metric='euclidean'))
      print("-----------------------------------------------")

