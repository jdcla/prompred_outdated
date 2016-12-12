import matplotlib
import numpy as np
import math
import itertools
import sklearn
import warnings
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###################################### FUNCTIONS ######################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Query MySQL #######################################################################
'''
def querySQL (database, query):
    config = {
    'user': 'root',
    'password': 'jim',
    'host': 'localhost',
    'database': database,
    'raise_on_warnings': True,
    }
    cnx = myconnect.connect(**config) 
    resultset = pd.read_sql_query(query,cnx)
    
    return resultset
'''
#######################################################################################
### Give scores for subset of parameters ##############################################

def bestSubsetSelection(X_train, X_test, y_train, y_test, parCount, regType, alpha=0, poly=None):
    
    parameters = len(X[0])
    combinations = list(itertools.combinations(range(parameters),parCount))
    combinationsL = len(combinations)
    scores = np.empty([combinationsL])
    for i in range(0, combinationsL):
        
        reg = selectRegression(regType,alpha=alpha, poly=poly)
        reg.fit(X_train[:,combinations[i]], y_train)
        scores[i] = reg.score(X_test[:,combinations[i]], y_test)
        
    #Sort results (descending)
    indexMin = np.argsort(scores)
    sortedCombin =[combinations[u] for u in indexMin[::-1]]
    sortedScores =[scores[u] for u in indexMin[::-1]]
    #plt1 = plt.figure(1)
    #plt.plot(range(len(sortedScores)),sortedScores, 'ro')
    return sortedCombin, sortedScores

#######################################################################################
### Create full dummy dataframe #######################################################

def createFullDummyDF(positions, start35Box, start10Box):
    length = len(positions)
    a = np.empty([length],dtype=np.dtype(str,1))
    c = np.empty([length],dtype=np.dtype(str,1))
    t = np.empty([length],dtype=np.dtype(str,1))
    g = np.empty([length],dtype=np.dtype(str,1))
    dash = np.empty([length],dtype=np.dtype(str,1))
    a.fill('A')
    t.fill('T')
    c.fill('C')
    g.fill('G')
    dash.fill('A')
    dash[:start35Box[0]]='-'
    dash[(start35Box[0]+32):]='-'
    dataframe = pd.DataFrame(np.vstack((a,t,c,g,dash)),columns=positions)
    dummyTable = pd.get_dummies(dataframe)
    
    return dummyTable

######################################################################################
### Create position matrix ###########################################################

def createPositionMatrix(sequences):
    ntArray = np.empty([len(sequences),len(sequences[0])],dtype=np.dtype(str,1))
    sequences=np.array(sequences)
    for i in range(0,len(sequences)):
        ntArray[i] = [sequences[i][u] for u in range(0,len(sequences[i]))]
    
    return ntArray 

######################################################################################
### Create compatible position dataframe #############################################

def createPositionDF(seq, posRange, start35Box, start10Box):
    # Create Position Matrix
    posMatrix = createPositionMatrix(seq)
    # Convert to Dataframe
    posRangeCol = [str(x) for x in range(posRange[0],posRange[1])]
    dfPosMatrix = pd.DataFrame(posMatrix, columns = posRangeCol)

    # Create dummy Matrix
    dfDumPosMatrix =  pd.get_dummies(dfPosMatrix)


    return dfDumPosMatrix

#######################################################################################
### getParameter ######################################################################

def getParameters(parLabel, parRange):
    
    parameters = [np.zeros(parRange[u]) for u in range(len(parRange))]
 
 
    for i in range(len(parLabel)):
        if parLabel[i] is "alpha":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "gamma":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "C":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "coef0":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "epsilon":
            parameters[i][:] = [0+2/parRange[i]*u for u in range(parRange[i])]
        elif parLabel[i] is "max_depth":
            parameters[i][:] = [u+1 for u in range(parRange[i])]
        elif parLabel[i] is 'min_samples':
            parameters[i][:] = [u+1 for u in range(parRange[i])]
        elif parLabel[i] is 'max_features':
            parameters[i][:] = [int(u+2) for u in range(parRange[i])]
        else:
            return print("not a valid parameter")
    
    parEval = {parLabel[u]:parameters[u] for u in range(len(parLabel))}
    return parEval

########################################################################################
### Evaluate Multiple Parameters #######################################################

def evaluateMultiPar(X, y, modelPar, parLabel, parRange):
    
    #data prep
    parEval = getParameters(parLabel,parRange)
    yMin, yMax = np.min(parEval[parLabel[0]]), np.max(parEval[parLabel[0]])
    xMin, xMax = np.min(parEval[parLabel[1]]), np.max(parEval[parLabel[1]])
    
    #model fitting
    reg = selectRegression(**modelPar)
    GS = GridSearchCV(reg, parEval)
    GS.fit(X,y)

    #plotting
    plt.figure("Model accuracy ifv model parameters")
    plt.imshow(GS.cv_results_["mean_test_score"].reshape([parRange[0],parRange[1]]), cmap='hot', interpolation='none', 
              vmin=0, vmax=np.max(GS.cv_results_["mean_test_score"]),  extent=[xMin,xMax,yMin,yMax], aspect="auto", 
              )
    plt.colorbar()
    plt.xlabel(parLabel[1]), plt.xscale('log')
    plt.ylabel(parLabel[0]), plt.yscale('log')

########################################################################################
### Evaluate Single Parameter #################################################################

def evaluateSinglePar(X_train, X_test, y_train, y_test, parModel, parLabel, parRange):

    parEval = getParameters([parLabel], parRange)
    scores = np.zeros(parRange)

    for i in range(len(parEval[parLabel])):
        parEvalIt = {parLabel:parEval[parLabel][i]}
        reg = selectRegression(**parModel,**parEvalIt)
        reg.fit(X_train,y_train)
        scores[i] = reg.score(X_test,y_test)
    
    optimalPar = parEval[parLabel][np.argmax(scores)]
    
    return scores, optimalPar

##########################################################################################
### Evaluate Parameter ################################################################

def evaluateScore(X_train, X_test, y_train, y_test, parModel):

    reg = selectRegression(**parModel)
    reg.fit(X_train,y_train)
    score = reg.score(X_test,y_test)
    y_pred = reg.predict(X_test)
 
    return score, y_pred

##########################################################################################
### Extract ATCG fractions region ########################################################

def extractATCGpercentage(box):
    boxLength = len(box[1])
    
    fractionA= [u.count('A')/boxLength for u in box]
    fractionT= [u.count('T')/boxLength for u in box]
    fractionC= [u.count('C')/boxLength for u in box]
    fractionG= [u.count('G')/boxLength for u in box]
    
    return fractionA, fractionT, fractionC, fractionG

##########################################################################################

def evaluateMutalik(dfDatasetTest, ROI, seqRange, regtype, poly, kernel, treeCount, evalPar = False):
    
    
    dfDatasetTrain = pd.read_csv("data/mut_rand_mod_lib.csv")

    dfDatasetTest['sequence'] = dfDatasetTest['sequence'].str.upper()
    dfDatasetTest = dfDatasetTest.sort(columns='mean_score',ascending=False)

    labelsTrain, positionBoxTrain, spacerTrain = regionSelect(dfDatasetTrain, ROI, seqRange)
    labelsTest, positionBoxTest, spacerTest = regionSelect(dfDatasetTest, ROI, seqRange)

    positionMatrixTrain = positionBoxTrain.values
    positionMatrixTest = positionBoxTest.values


    Xdf = positionBoxTrain
    X = positionMatrixTrain
    y = [math.sqrt(math.sqrt(u)) for u in labelsTrain]


    Xtest = positionMatrixTest
    ytest = labelsTest

    if parModelTweak is None:
        
        

        parModel = {"regType":regType, "poly":poly, "kernel":kernel, "treeCount":treeCount}

        parEval = getParameters(parLabel,parRange)
        reg = selectRegression(**parModel)
        GS = GridSearchCV(reg, parEval)
        GS.fit(X,y)
        reg = GS.best_estimator_
        print(GS.best_estimator_)

    else:
        reg = selectRegression(**parModelTweak)


    reg.fit(X,y)
    rankPredict = reg.predict(Xtest)

    print(np.transpose(np.vstack((dfDatasetTest['sequence'].values,dfDatasetTest['mean_score'].values,rankPredict))))
    print(stats.spearmanr(dfDatasetTest['mean_score'].values,rankPredict))
    plt.plot(dfDatasetTest['mean_score'].values,rankPredict, 'ro')
                    
                    
###############################################################################################                    
### K-fold regression cross validation ###################################################

def KfoldCV(X,y,k,parModel, parLabel ,parRange):
    
    y = np.array(y)
    kf = KFold(n_splits=k)
    scoresPar = np.zeros([k,parRange])
    OutI=np.bool([0])
    OutV=np.empty([0])
    optimalPar = np.zeros([k])
    index=0
    plt.figure(num=None, figsize=(10, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(122)
    plt.plot([0,6],[0,6], ls="--", c=".3")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scoresPar[index,:], optimalPar[index] = evaluateSinglePar(X_train, X_test, y_train, y_test, parModel, parLabel, [parRange])
        score, y_pred = evaluateScore(X_train, X_test, y_train, y_test, {**parModel, parLabel: optimalPar[index]})
        plt.title("pred/true: K-fold")
        plt.plot(y_test,y_pred, 'bo')
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        index+=1
    
    #scores = np.transpose(scoresM)
    plt.subplot(121)
    plt.title("Cross Validation")  
    plt.imshow(scoresPar, cmap='hot', vmin=0 ,interpolation='nearest')
    plt.colorbar()
    #plt.figure("Cross Validation: Normalized")
    #plt.imshow(sklearn.preprocessing.normalize(scoresPar, axis=1),cmap='hot', vmin=0,interpolation='nearest')
    #plt.colorbar()

    return scoresPar, optimalPar

##########################################################################################
### Leave one out accuracy estimation ####################################################

def leaveOneOut(X,y,regType,alpha=1):
    y=np.array(y)
    X=np.array(X)
    loo = LeaveOneOut()
    scores = np.zeros([len(X)])
    index = 0
    for train_index,test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(X_train,y_train, X_test, y_test)
        reg = selectRegression(regType, alpha=alpha)
        reg.fit(X_train,y_train)
        scores[index] = reg.score(X_test,y_test)
        #print(reg.score(X_test,y_test))
        index += 1
    
        return scores

########################################################################################
### Merge multiple regions in one position dataframe####################################

def mergePositionDF(df1,df2):
    
    df3 = pd.concat([df1, df2], axis=1)
    if len(df3)>len(df2):
        print("WARNING: OVERLAPPING POSITIONS")
    else:
        return df3
    

########################################################################################
### K-fold nested cross validation #####################################################

def nestedKfoldCV(X,y,k,kInner, parModel, parLabel,parRange):
    y = np.array(y)
    kf = KFold(n_splits=k)
    kfInner = KFold(n_splits=kInner)
    scoresM = np.zeros([k,kInner,parRange])
    scoresCV = np.zeros([k]) 
    optimalParM = np.zeros([k,kInner])
    index=0

    for train_index, test_index in kf.split(X):     
        indexI=0
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splitLib = list(kfInner.split(X_train))
        for train_indexI, test_indexI in splitLib:
            X_trainI, X_testI = X_train[train_indexI], X_train[test_indexI]
            y_trainI, y_testI = y_train[train_indexI], y_train[test_indexI]
            scoresM[index,indexI,:], optimalParM[index,indexI] = evaluateSinglePar(X_train, X_test, y_train, y_test, parModel, parLabel, [parRange])
            indexI += 1
        
        scoresCV[index], y_pred = evaluateScore(X_train, X_test, y_train, y_test, {**parModel, parLabel:  stats.mode(optimalParM[index])[0][0]})
        fig = plt.figure("pred/true: Nested K-fold")
        plt.plot(y_test,y_pred, 'bo')
        plt.plot([0,6],[0,6],ls="--", c=".3")

        index+=1
        
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.show()
  

    return scoresCV, optimalParM, scoresM

#########################################################################################
### Plot Histograms ######################################################################

def plotHistograms(yData,blocks):
    #yData = np.array(yData)
    yDataSqrt = [math.sqrt(y) for y in yData]
    yDataLog = [math.log(10,y) for y in yData]
    yDataQuad = np.power(yData,2)
    plot = plt.figure("Histograms")

    plt.subplot(221)
    plt.hist(yData,blocks,normed=1)
    plt.title('Histogram of y values')
    plt.subplot(222)
    plt.hist(yDataSqrt,blocks,normed=1)
    plt.title('Histogram of squared y values')
    plt.subplot(223)
    plt.hist(yDataLog,blocks,normed=1)
    plt.title('Histogram of log y values')
    ax4 = plt.subplot(224)
    plt.hist(yDataQuad,blocks,normed=1)
    plt.title('Histogram of quadratic y values')\
    
    wm = plt.get_current_fig_manager()

##########################################################################################
### Plot Position coefficients ###########################################################

def plotPositionCoefficients(linReg, dfDumPosMatrix, positions, results=''):
    posRangeCol = posRangeCol = [str(x) for x in range(positions[0],positions[1])]
    
    normCoef = linReg.coef_/linReg.coef_.max()
    normInt = linReg.intercept_/linReg.coef_.max()
    #normCoef = linReg.coef_
    #normInt = linReg.intercept_
    
    #print(normCoef, dfDumPosMatrix.columns)
    dfResultBox = pd.DataFrame([normCoef],columns=dfDumPosMatrix.columns.values)
    
    dfTemplateDummyBox = createFullDummyDF(posRangeCol)
    dfFinalBox = pd.DataFrame(np.nan, index=[0],columns=dfTemplateDummyBox.columns)

    colNamesResult = dfResultBox.columns

    for i in colNamesResult:
        if i not in  ['spacer','upSeq']:
            dfFinalBox[i] = dfResultBox[i]

    # Extract values per nt
            
    normA = separateNtData('A',dfFinalBox)
    normC = separateNtData('C',dfFinalBox)
    normT = separateNtData('T',dfFinalBox)
    normG = separateNtData('G',dfFinalBox)
    
    
    plt.figure("Position Matrix")
    bar_width = 0.15
    index = np.arange(positions[0],positions[1])
    
    rectsU = plt.bar(positions[0]-0.6, normCoef[-1], bar_width, color='orange', label='Up-Seq')
    rectsS = plt.bar(positions[0]-0.3, normCoef[-2], bar_width, color='purple', label='Spacer')
    rects0 = plt.bar(positions[0]-1, normInt, bar_width, color='black', label="Intercept")
    rects1 = plt.bar(index, normA, bar_width, label='A')
    rects2 = plt.bar(index + bar_width, normG, bar_width, color='r', label='G')
    rects3 = plt.bar(index + 2*bar_width, normC, bar_width, color='g', label='C')
    rects4 = plt.bar(index + 3*bar_width, normT, bar_width, color='y', label='T')
    #plt2 = plt.figure(2)
    plt.xlabel('Position')
    plt.ylabel('Coefficients')
    plt.title('Coefficients of every nucleotide at given position: ')
    plt.axis([positions[0]-1, positions[1]+1, -2, 2])
    plt.legend(prop={'size':15})

##########################################################################################
### Plot Predicted Scores to real scores #################################################

def plotPredictedToReal(yReal,yPred):
    plt.figure(ScatterPvsR)
    plt.plot(yReal,yPred,'ro')
     
        
##########################################################################################
### Region selection #####################################################################

def regionSelect(dfDataset, seqRegions, posRange):
    
    
    

    # Selecting regions
    labels = dfDataset['mean_score'].values
    sequences = dfDataset['sequence'].values 
    start35Box = dfDataset['35boxstart'].values 
    start10Box = dfDataset['10boxstart'].values 
    
    dataset = dfDataset.values
    
    difLength = -35-posRange[0]-start35Box
    sequenceAligned = ["-" *difLength[u]+sequences[u] if difLength[u]>=0 else sequences[u][abs(difLength[u]):] for u in range(np.shape(sequences)[0]) ]
    start35Box = np.array([start35Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])
    start10Box = np.array([start10Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])

    maxLength = max([len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])])
    difLength = [maxLength - len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])]
    sequences = [sequenceAligned[u]+ '-'*difLength[u] for u in range(np.shape(sequenceAligned)[0]) ]
    
    
    reg35 = np.array(seqRegions[0])
    reg10 = np.array(seqRegions[1])
    
    
    box35 = selectRegions(sequences, reg35, start35Box)
    box10 = selectRegions(sequences, reg10, start10Box)
    spacer = start10Box-start35Box-6
    
    spacerM = [u-17 if u-17>0 else 0 for u in spacer]
    spacerL = [abs(u-17) if u-17<0 else 0 for u in spacer]
    
    spacerBox = pd.DataFrame({'spacer_more':spacerM,'spacer_less':spacerL})
   
    # Creating features

    positionBox35 = createPositionDF(box35, reg35-35, start35Box, start10Box)
    positionBox10 = createPositionDF(box10, reg10-12, start35Box, start10Box)

    positionBoxSS = mergePositionDF(positionBox35,positionBox10)
      
    posRangeCol = [str(x) for x in range(posRange[0],posRange[1])]
    dfTemplateDummyBox = createFullDummyDF(posRangeCol, start35Box, start10Box)

    dfFinalBox = pd.DataFrame(0, range(len(dataset)),columns=dfTemplateDummyBox.columns)
    
    colNamesResult = positionBoxSS.columns
    for i in colNamesResult:
        if i in dfFinalBox.columns:
            dfFinalBox[i] = positionBoxSS[i]
    
    positionBox = mergePositionDF(dfFinalBox, spacerBox)

        
    return labels, positionBox, spacer
    

##########################################################################################
### Regression selection #################################################################

def selectRegression(regType, poly=None, kernel=None, alpha=0.0001, gamma=0.1, epsilon=1.95, coef0=1,  
                     fitInt=True, norm=True, max_depth=None, max_features=None, min_samples_split = 2, 
                     treeCount = 100, C=1):
    if kernel:
        if regType is "ridge":
            reg = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel, coef0=coef0)
        if regType is "SVC":
            reg = sklearn.svm.SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, epsilon=epsilon, degree=poly)
        if regType is "SVR":
            reg = sklearn.svm.SVR(C=C, kernel=kernel, gamma=gamma, coef0=coef0, epsilon=epsilon, degree=poly)
                    
    elif poly:
        if regType is "OLS":
            reg = make_pipeline(PolynomialFeatures(poly), LinearRegression(fit_intercept=fitInt, normalize=norm))
        if regType is "ridge":
            reg = make_pipeline(PolynomialFeatures(poly), sklearn.linear_model.Ridge(alpha= alpha, normalize=True))
        if regType is "lasso":
            reg = make_pipeline(PolynomialFeatures(poly), sklearn.linear_model.Lasso(alpha= alpha, normalize=True))
        if regType is "huber":
            reg = make_pipeline(PolynomialFeatures(poly), sklearn.linear_model.HuberRegressor(fit_intercept=fitInt, epsilon=epsilon, alpha=alpha))

    else: 
        if regType is "OLS":
            reg = LinearRegression(fit_intercept=fitInt, normalize=norm)
        if regType is "ridge":
            reg = sklearn.linear_model.Ridge(alpha= alpha, normalize=True)
        if regType is "lasso":
            reg = sklearn.linear_model.Lasso(alpha= alpha, normalize=True)
        if regType is "huber":
            reg = sklearn.linear_model.HuberRegressor(fit_intercept=fitInt, alpha=alpha, epsilon=epsilon)
        if regType is "treeReg":
            reg = tree.DecisionTreeRegressor(max_depth= max_depth, max_features=max_features, min_samples_split = min_samples_split)
        if regType is "treeClass":
            reg = tree.DecisionTreeClassifier(max_depth = max_depth, max_features=max_features, min_samples_split = min_samples_split)
        if regType is "forestReg":
            reg = RandomForestRegressor(n_estimators = treeCount, max_depth = max_depth, max_features= max_features, min_samples_split = min_samples_split)
        if regType is "forestClass":
            reg = RandomForestClassifier(n_estimators = treeCount, max_depth = max_depth, max_features= max_features, min_samples_split = min_samples_split)
    
    return reg

##########################################################################################
### Select regions of interest ###########################################################

def selectRegions(sequences, positions, referencePoint=None):
    if referencePoint.all == None:
        selectedRegion = [sequences[u][positions[0]:positions[1]] for u in range(0,len(sequences))]
    else:
        selectedRegion = [sequences[u][positions[0]+referencePoint[u]:positions[1]+referencePoint[u]] for u in range(0,len(sequences))]
    
    return selectedRegion

##########################################################################################
### Extract nucleotide values from resultset #############################################

def separateNtData(colSubStr, Data):
    Index = [c for c in Data.columns if colSubStr in c]
    subData = np.squeeze(Data[Index].values)
    
    return subData

##########################################################################################
