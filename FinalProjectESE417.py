import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import neighbors
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier


def data_normalization(wine):
    for i in range(8):
        var = np.var(wine[:,i])
        mean = np.sum(wine[:,i])/len(wine[:,i])
        wine[:,i] = (wine[:,i]-mean)/var
    return wine

#LOAD DATA FROM CSV
wine_data = pd.read_csv('winequality-red.csv', sep=",")
#SEPERATE INTO INPUTS (X) AND CLASS LABEL (Y)
X = wine_data.loc[:,"fixed acidity":"alcohol"]
Y = wine_data.loc[:,"quality"]

#SPLIT INTO TRAINING AND TEST DATA
tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.25, random_state=5)

#SELECTING 'K' BEST FEATURES BASED OFF OF CHI2 SCORE
features = SelectKBest(score_func=chi2,k=11)
best_features = features.fit(tr_x,tr_y)
scores = pd.DataFrame(best_features.scores_)
columns = pd.DataFrame(X.columns)
predScores = pd.concat([columns,scores],axis=1)
predScores.columns = ['Predictor','Score']   #naming the dataframe columns
#BEST 8
best = predScores.nlargest(8,'Score')
print(best)

#SETTING X EQUAL TO ONLY 8 BEST PREDICTOR FEATURE COLUMNS
X = wine_data.loc[:,best["Predictor"]]
print(X)
#NEED NUMPY ARRAY FOR NORMALIZATION CALCULATION
x=np.array(X)
X = pd.DataFrame(data_normalization(x))
print(X)

#REINITIALIZE WITH NEW X THAT ONLY HAS 8 BEST PREDICTIVE INPUT FEATURES
tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.25, random_state=5)
score_train = []
score_test = []

#HYPERTUNING PARAMETERS USING GRID SEARCH METHOD AND MLP CLASSIFIER
pr_grid = {'hidden_layer_sizes':[(10,10),(25,25),(50,50),(100,100),(250,250)],
            'activation': ['logistic', 'tanh', 'relu'],
            'learning_rate_init' : [.001,.01,.1]}
ann = MLPClassifier(max_iter = 500, random_state=420)
cv_ann = GridSearchCV(ann,param_grid=pr_grid)
cv_ann.fit(tr_x, tr_y)
#PRINT OUT BEST PARAMETERS AND THE SCORE THEY PROVIDED
print("The best parameters are %s with a score of %0.5f" % (cv_ann.best_params_, cv_ann.best_score_))
#TEST SCORE
print("GRID SCORE FOR TEST DATA: "+str(cv_ann.score(te_x,te_y)*100))


#BEST PARAMETERS FOUND FROM HYPERTUNING
MLP = MLPClassifier(activation = 'logistic',
                   solver = 'adam',
                   hidden_layer_sizes = (50,),
                   max_iter = 500,
                   random_state = 420,
                   learning_rate_init = 0.01)
MLP.fit(tr_x,tr_y)
score_train.append(MLP.score(tr_x,tr_y))
score_test.append(MLP.score(te_x,te_y))

print(score_train)
print(score_test)
plt.figure()
plt.plot(score_train,'.',label = 'train set')
plt.plot(score_test,'-',label = 'test set')
plt.xlabel('logistic :: tanh :: relu')
plt.ylabel('score')
plt.legend()
