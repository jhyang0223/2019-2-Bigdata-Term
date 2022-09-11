import matplotlib.pyplot as plt

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.manifold import TSNE
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
tsne = tsne.fit_transform(X)
plt.scatter(tsne[:,0],tsne[:,1],  c = y, cmap = "winter", edgecolor = "None", alpha=0.35)
plt.title('t-SNE Scatter Plot')
