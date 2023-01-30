import preprocessingFuncts as pp
import numpy as np
import pandas as pd
from models import KmeansCF
from models import CF
from simModel import simiModel
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report

modelname = ["KmeansCF('user')","CF('user')","CF('item')","simiModel('user')","CategoricalNB()"]
dicpath=['ml-100k\\u1.base','ml-100k\\u2.base','ml-100k\\u3.base','ml-100k\\u4.base','ml-100k\\u5.base']
dicpathtest=['ml-100k\\u1.test','ml-100k\\u2.test','ml-100k\\u3.test','ml-100k\\u4.test','ml-100k\\u5.test']
arravgtrain=[]
arravgtest=[]

for i in range(4,len(dicpath)):
    print(f'i is now at : {i}')
    index=0
    mydatatest=pp.readRatingData(dicpathtest[i])
    testX, testY = mydatatest.loc[:, ['user_id',
                                    'item_id']], mydatatest.loc[:, 'rating']
    # print(dicpath[i])
    # model = [KmeansCF('user'),CF('user'),CF('item'),simiModel('user')]  
    # for j in model:
    #     print(f'Model: {modelname[index]}')
    #     tempmodel = j
    #     tempmodel.fit(dicpath[i])
    #     predY = tempmodel.predict(testX)
    #     print(classification_report(testY,predY))
    #     index+=1
    
    print("going here")
    print(f'Model: {modelname[index]}')
    tempmodel = CategoricalNB()
    print(dicpath[i])
    ratingData = pp.readRatingData(dicpath[i])
    print(ratingData)
    xTrain, yTrain = ratingData.loc[:,['user_id','item_id']], ratingData.loc[:, 'rating']
    tempmodel.fit(xTrain, yTrain)
    print(tempmodel.class_count_)
    predY = tempmodel.predict(testX)
    print(classification_report(testY, predY))
    exit()
