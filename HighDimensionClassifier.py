# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

#read the data
data = pd.read_csv("data.csv")

#split into train and test
train = data[data.train==1]
test = data[data.train!=1]

k = 10
kf = KFold(len(train.index), n_folds=k, shuffle=True)
gnb = GaussianNB()

#make copy of train that also holds predicted probabilities
trainwprob = train.copy()

#loop through each fold
for train_index, test_index in kf:

    trainx = train.iloc[train_index, train.columns!="target_eval"]
    trainy = train.iloc[train_index, train.columns=="target_eval"].values.flatten()
    
    testx = train.iloc[test_index, train.columns!="target_eval"]
    truth = train.iloc[test_index, train.columns=="target_eval"].values.flatten()
    
#    fit model and return pred probabilities for test set
    prob = gnb.fit(trainx, trainy).predict_proba(testx)[:,1]
    
#    update train df
    trainwprob.loc[test_index,"prob"] = prob


from sklearn import metrics

#get metrics
prob = trainwprob.prob
truth = trainwprob.target_eval
fpr, tpr, _ = metrics.roc_curve(truth, prob)
auc = metrics.roc_auc_score(truth, prob)

#plot ROC
import matplotlib.pyplot as plt
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.title("ROC Curve (using " + str(k) + "-fold Cross Validation) \n AUC = " + str(round(auc, 2)))
plt.plot([0,1])
plt.plot(fpr, tpr)
plt.show()


