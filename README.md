# bigdata
## Introduction
Breast mass data with two classes of benign breast masses (simple cysts, fibroblasts, etc.) and malignant breast masses (breast cancer) of Wisconsin Diagnostic Breast Cancer (WDBC) has 32 data attributes. In this project, the goal is to learn, implement, and test two classification techniques and one clustering technique using these 32 attributes.
## Method
- supervised ML method: random forest, svm
- unsupervised ML method: k-means
* I used scikit library
## Conclusion
I manufactured programs for two classification techniques (Random Forest, SVM) and one cluster techniques (kmeans), and tested programs produced using WDBC's breast cancer diagnostic data set. In the case of classification, since the data is refined example data provided by the sklearn library, there was no separate preprocessing process, such as filling the miss value, and the resulting value acuity was also recorded as very high. On the other hand, in the case of clustering, we believe that the results were not compliant because there were overlapping parts where the data were not completely isolated clusters. This project was a good experience related to machine learning program production. Based on this experience, I will apply machine learning techniques to future research tasks.
