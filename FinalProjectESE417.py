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
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.svm import SVC


def data_normalization(wine):
    for i in range(6):
        print(i)
        print(wine)
        var = np.var(wine[:,i])
        mean = np.sum(wine[:,i])/len(wine[:,i])
        wine[:,i] = (wine[:,i]-mean)/var
    return wine

choice = -1
while(True):
    choice = int(input("WHICH CLASSIFIER:\n1)ANN\n2)RF\n3)NBC\n4)SVM\n5)QUIT\n\n"))
    if (choice > 5 or choice < 1):
        print("TRY AGAIN, INVALID SELECTION\n\n")
        continue
    if (choice == 5):
        break

    bin = 0
    while (bin != 1 and bin != 2):
        bin = int(input("DO YOU WANT\n1)BINARY\n2)MULTI-CLASSIFICATION\n\n"))

    bin = True if(bin == 1) else False
    #LOAD DATA FROM CSV
    wine_data = pd.read_csv('winequality-red.csv', sep=",")

    #SEPERATE INTO INPUTS (X) AND CLASS LABEL (Y)
    X = wine_data.loc[:,"fixed acidity":"alcohol"]

    if (bin):
        for i in range(1599):
            print(i)
            if wine_data.iloc[i, -1] > 6:
                wine_data.iloc[i, -1] = 1
            else:
                wine_data.iloc[i,-1] = 0
        #print(wine_data)

    Y = wine_data.loc[:,"quality"]
    #SPLIT INTO TRAINING AND TEST DATA
    tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.25, random_state=5)
    #SELECTING 'K' BEST FEATURES BASED OFF OF CHI2 SCORE
    features = SelectKBest(score_func=chi2,k=11)
    print("FEATURES: \n", features)
    best_features = features.fit(tr_x,tr_y)
    print("BEST FEATURES: ",best_features)
    scores = pd.DataFrame(best_features.scores_)
    print(scores)
    columns = pd.DataFrame(X.columns)
    predScores = pd.concat([columns,scores],axis=1)
    print(predScores)
    predScores.columns = ['Predictor','Score']   #naming the dataframe columns
    #BEST 8
    best = predScores.nlargest(6,'Score')
    print(best)

    #SETTING X EQUAL TO ONLY 8 BEST PREDICTOR FEATURE COLUMNS
    X = wine_data.loc[:,best["Predictor"]]

    #NEED NUMPY ARRAY FOR NORMALIZATION CALCULATION
    x=np.array(X)
    print(x)
    X = pd.DataFrame(data_normalization(x))
    print(X)

    #REINITIALIZE WITH NEW X THAT ONLY HAS 8 BEST PREDICTIVE INPUT FEATURES
    tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.25, random_state=5)
    score_train = []
    score_test = []

    #plt.figure(figsize=(15,10))
    #sns.heatmap(wine_data.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    #plt.show()

    #Plotting histogram of each variable
    plt.figure(1)
    wine_data.hist(alpha=0.5, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    if (choice==1):

        print("ARTIFICIAL NEURAL NETWORK\n\n")

        #HYPERTUNING PARAMETERS USING GRID SEARCH METHOD AND MLP CLASSIFIER
        pr_grid = {'hidden_layer_sizes':[(10,10),(25,25),(50,50),(100,100),(250,250)],
                    'learning_rate_init' : [.001,.01]}
        ann = MLPClassifier(max_iter = 500, random_state=420, early_stopping = True)
        cv_ann = GridSearchCV(ann,param_grid=pr_grid)
        cv_ann.fit(tr_x, tr_y)
        #PRINT OUT BEST PARAMETERS AND THE SCORE THEY PROVIDED
        print("The best parameters are %s with a score of %0.5f" % (cv_ann.best_params_, cv_ann.best_score_))
        #TEST SCORE
        print("GRID SCORE FOR TEST DATA: "+str(cv_ann.score(te_x,te_y)*100))


        #BEST PARAMETERS FOUND FROM HYPERTUNING
        MLP = MLPClassifier(activation = 'tanh',
                           solver = 'adam',
                           hidden_layer_sizes = cv_ann.best_params_['hidden_layer_sizes'],
                           max_iter = 500,
                           random_state = 420,
                           early_stopping = True,
                           learning_rate_init = cv_ann.best_params_['learning_rate_init'])
        MLP.fit(tr_x,tr_y)
        score_train.append(MLP.score(tr_x,tr_y))
        score_test.append(MLP.score(te_x,te_y))
        y_pred = MLP.predict(te_x)

        print("Mean Accuracy Score on Training Data: ",score_train)
        print("Mean Accuracy Score on Test Data: ",score_test)
        print("Accuracy from Metrics Accuracy on Y Predicted vs Y-Test: ",metrics.accuracy_score(te_y,y_pred))

        plt.figure(2)
        plt.plot(score_train,label = 'train set')
        plt.plot(score_test,label = 'test set')
        plt.ylabel('score')
        plt.legend()
        plt.show()

    elif(choice==2):

        print("RANDOM FOREST CLASSIFIER\n\n")

        #HYPERTUNING PARAMETERS USING GRID SEARCH METHOD AND RANDOM FOREST CLASSIFIER
        estimators = [10,25,50,100,250,500,750,1000,2500]
        rf_grid = {'n_estimators': estimators}
        rfr = RandomForestClassifier(random_state=1,criterion='entropy',max_features='log2')
        cv_rfr = GridSearchCV(rfr,param_grid=rf_grid)
        cv_rfr.fit(tr_x, tr_y)
        #PRINT OUT BEST PARAMETERS AND THE SCORE THEY PROVIDED
        print("The best parameters are %s with a score of %0.5f" % (cv_rfr.best_params_, cv_rfr.best_score_))
        #TEST SCORE
        print("GRID SCORE FOR TEST DATA: "+str(cv_rfr.score(te_x,te_y)*100))


        #BEST PARAMETERS FOUND FROM HYPERTUNING
        giggity = 331
        for i in estimators:
            RFC = RandomForestClassifier(n_estimators=i,
                                        random_state=1,
                                        criterion='entropy',
                                        max_features='log2')
            RFC.fit(tr_x,tr_y)
            score_train.append(RFC.score(tr_x,tr_y))
            score_test.append(RFC.score(te_x,te_y))
            y_pred = RFC.predict(te_x)

            print("Mean Accuracy Score on Training Data: ",score_train)
            print("Mean Accuracy Score on Test Data: ",score_test)
            print("Accuracy from Metrics Accuracy on Y Predicted vs Y-Test: ",metrics.accuracy_score(te_y,y_pred))

            plt.figure()
            plt.subplot(giggity)
            plt.plot(score_train,'.',label = 'train set')
            plt.plot(score_test,'-',label = 'test set')
            plt.ylabel('score')
            plt.legend()
            plt.show()

            giggity += 1

    elif(choice==3):

        print("NAIVE BAYES CLASSIFIER\n\n")

        nbc = GaussianNB()
        nbc.fit(tr_x,tr_y)
        score_train.append(nbc.score(tr_x,tr_y))
        score_test.append(nbc.score(te_x,te_y))
        y_pred = nbc.predict(te_x)

        print("Mean Accuracy Score on Training Data: ",score_train)
        print("Mean Accuracy Score on Test Data: ",score_test)
        print("Accuracy from Metrics Accuracy on Y Predicted vs Y-Test: ",metrics.accuracy_score(te_y,y_pred))

    elif(choice==4):
        print("SUPPORT VECTOR MACHINE BABY!\n\n")

        kernels = ['linear','poly','rbf','sigmoid']
        cs = [1, 2, 5, 10, 50, 100]
        #HYPERTUNING PARAMETERS USING GRID SEARCH METHOD AND MLP CLASSIFIER
        svm_grid = {'kernel': ['linear','poly','rbf','sigmoid'],
                    'C': [1, 2, 5, 10, 50, 100]}
        svm = SVC(random_state=420,max_iter=500)
        cv_svm = GridSearchCV(svm,param_grid=svm_grid)
        cv_svm.fit(tr_x, tr_y)
        #PRINT OUT BEST PARAMETERS AND THE SCORE THEY PROVIDED
        print("The best parameters are %s with a score of %0.5f" % (cv_svm.best_params_, cv_svm.best_score_))
        #TEST SCORE
        print("GRID SCORE FOR TEST DATA: "+str(cv_svm.score(te_x,te_y)*100))

        sub_num = 221

        for i in kernels:
            score_train_sub = []
            score_test_sub = []
            for j in cs:
                svm = SVC(random_state=420, max_iter = 500, kernel=i, C=j)
                svm.fit(tr_x,tr_y)
                score_train_sub.append(svm.score(tr_x,tr_y))
                score_test_sub.append(svm.score(te_x,te_y))
                y_pred = svm.predict(te_x)

            score_train.append(score_train_sub)
            score_test.append(score_test_sub)
            plt.figure(1)
            plt.subplot(sub_num)
            plt.plot(cs,score_train_sub,label = 'train set')
            plt.plot(cs,score_test_sub,label = 'test set')
            plt.ylabel('Score for Kernel: '+str(i)+'for C Values')
            plt.legend()
            sub_num += 1
        plt.show()

    else:
        print("SHOULDN'T BE HERE\n\n")
