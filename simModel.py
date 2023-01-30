import pandas as pd
import numpy as np
import preprocessingFuncts as pp
from sklearn import neighbors


'''''''''
TODO 
- take simi for rating,occup,age,gender
- weighted parameter [made rating most important ,second to occup and age,gender atleast signi]
- 

'''''''''
class  simiModel:
    def __init__(self,base,k=5,threshold=30,weight=[5,3,3,1]) -> None:
      self.k = k
      self.base = 'user_id' if base == 'user' else 'item_id'
      self.not_base = 'user_id' if base != 'user' else 'item_id'
      self.threshold = threshold
      self.dataDF = 0
      self.dataMatrix = 0
      self.simMatrix = 0
      self.combMatrix=0
      self.weight= weight
      self.simRating = None
      if base=="user":
          self.length= "user_id"
      else:
          self.length= "item_id"

    def predict(self, x):
      y = []
      # print(x.columns)
      # print()
      for _x in x.values:
        # print(_x)
        # print(self.base)
        # raise("debug lol")
        notBaseID, baseID = 0,0
        if(self.base == 'item_id'):
          notBaseID, baseID = _x[0], _x[1]
        else:
          notBaseID, baseID = _x[1], _x[0]

        try:
          simItemIds = self.simMatrix.loc[:,baseID].sort_values(ascending=False)
        except Exception as e:
          # TODO - figure out how to get unrated movie's sim
          # This try except is made because movie 1582 has never been rated before
          # print(e)
          y.append(0)
          continue
        simItemIds = simItemIds.drop(baseID).to_frame().dropna().reset_index().set_axis([self.base,'corr'],axis='columns')
        if (len(simItemIds.index)==0):
          y.append(0)
          continue
    
        _y,a,b = 0,0,0
        _k = self.k
        _idx = 0

        while _k>0 and _idx<len(simItemIds.index):
          try:
            # Because of missing movie 1582
            if(not pd.isna(self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]])):
              tempA = simItemIds.loc[_idx,'corr']*self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]]
              tempB = simItemIds.loc[_idx,'corr']
              a+=tempA
              b+=tempB
              _k-=1
            _idx += 1
          except:
            break
      
        try:
          #TODO - fix bug. for some reason some movies' ratings are -10,
          # some others are 6. So bad loll
          # also need to handle the datas that are 0.
          _y = round(a/b) if round(a/b)>0 else 0
          _y = 5 if _y>5 else _y
        except Exception as e:
          # print(e)
          _y = 0
        y.append(_y)
      # print(y)
      return y

    def fit(self,path):
        ratingData = pp.readRatingData(path)
        userData = pp.readUserData()
        if(self.dataDF!= 0):
            if(self.dataDF.columns != ratingData.columns):
                raise Exception(f"Columns of inputted data {ratingData.columns} does not match the pre existing data {self.dataDF.columns}")
            self.dataDF.append(ratingData)
        else:
            self.dataDF = ratingData
        
        self.dataDF = pd.merge(self.dataDF,userData)
        self.dataDF = self.dataDF.drop(['age','zip_code'],axis=1)
        simocc = self.weight[1]*pp.getSimOccup()
        simage = self.weight[2]*pp.getSimAgeCategory()
        simgen = self.weight[3]*pp.getSimGender()


        userData = userData.set_index('user_id')
        #TODO - do this, but to user_id and item_id, not just for rating
        tempDataMatrix = self.dataDF.pivot_table(
            index=self.not_base, columns=self.base, values='rating')
        tempDataMatrix = tempDataMatrix.set_axis(
        [int(x) for x in tempDataMatrix.columns], axis='columns', inplace=False)
        tempDataMatrix = tempDataMatrix.set_axis(
        [int(x) for x in tempDataMatrix.index], axis='index', inplace=False)
        #print(tempDataMatrix)
        self.dataMatrix = tempDataMatrix
        self.simMatrix = self.dataMatrix.corr(min_periods=self.threshold)
        self.simRating = self.weight[0]*self.simMatrix
        endresult = pd.DataFrame(np.zeros((943, 943)),columns=range(1,944),index=range(1,944))
        # print(userData.loc[0,'age_category'])
        self.simRating = self.simRating.loc[userData.index,userData.index] #Probably wrong but i'll try it when the code finishes

        simage = simage.loc[userData.loc[userData.index,'age_category'],userData.loc[userData.index,'age_category']]
        simage.index = range(1,944)
        simage.columns = range(1,944)
        
        simocc = simocc.loc[userData.loc[userData.index,'occupation'],userData.loc[userData.index,'occupation']]
        simocc.index = range(1,944)
        simocc.columns = range(1,944)

        simgen = simgen.loc[userData.loc[userData.index,'gender'],userData.loc[userData.index,'gender']]
        simgen.index = range(1,944)
        simgen.columns = range(1,944)
        
        pembagian = endresult.copy()
        tempb = (self.simRating.notna()*5) + (simage.notna()*3) + (simocc.notna()*3) +(simgen.notna()*1)
        # ((not self.simRating.isna())*5).add((not simage.isna())*3).add((not simocc.isna())*3).add((not simgen.isna())*3)
        # pembagian.loc[self.simRating.loc[:,:].isna(),:] = 1
        endresult = (self.simRating.fillna(0) + simage + simocc + simgen) / tempb
        self.simMatrix = endresult
        return
                
# TEA = simiModel('user')
# TEA.fit('ml-100k\\ua.base')

# testData = pp.readRatingData('ml-100k\\ua.test')
# testX, testY =  testData.loc[:,['user_id','item_id']],testData.loc[:,'rating']

# predY = TEA.predict(testX)

# from sklearn.metrics import classification_report

# print(classification_report(testY, predY))
