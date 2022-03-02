#thoughtful machine learning in python

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr
import random
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import sys

sys.setrecursionlimit(10000)#since KDTree will recurse and throw an error otherwise

class Regression:
	def __init__(self, csv_file=None, data=None, values=None):
		if(data is None and csv_file is not None):
			df=pd.read_csv(csv_file)
			self.values=df['AppraisedValue']
			df=df.drop('AppraisedValue',1)
			df=((df-df.mean())/df.max()-df.min())
			self.df=df
			self.df=self.df['lat','long','SqFtLot']
		elif (data is not None and values is not None):
			self.df=data
			self.values=values
		else:
			raise ValueError("Must have either csv_file or data set")
			
		self.n=len(self.df)
		self.kdtree=KDTree(self.df)
		self.metric=np.mean
		self.k=5
#here we are normalizing data. this makes all of data similar.
#we are only selecting latitude, longitude and sqftlot because this is proof of concept	

	def regress(self,query_point):
		distances, indexes=self.kdtree.query(query_point,self.k)
		m=self.metric(self.values.iloc[indexes])
		if np.isnan(m):
			zomg
		else:
			return m
#here we are querying KDTree to find closes K houses.
#we use mean metric to calculate regression value.

#we need test to make sure our data is working properly.
#we use cross-validation to calculate performance. The generalized algorithm is 
#take training set and split it into two categories: testing and training
#use training data to train model
#use testing data to test how well model performs
	def error_rate(self,folds):
		holdout=1/float(folds)
		errors=[]
		for fold in range(folds):
			y_hat,y_true=self.__validation_data(holdout)
			errors.append(mean_absolute_error(y_true,y_hat))
			
		return errors
		
	def __validation_data(self,holdout):
		test_rows=random.sample(self.df.index, int(round(len(self.df)*holdout)))
		train_rows=set(range(len(self.df)))-set(test_rows)
		df_test=self.df.ix[test_rows]
		df_train=self.df.drop(test_rows)
		test_values=self.values.ix[test_rows]
		train_values=self.values.ix[train_rows]
		kd=Regression(data=df_train,values=train_values)
		
		y_hat=[]
		y_actual=[]
		
		for idx, row in def_test.iterrows():
			y_hat.append(kd.regress(row))
			y_actual.append(self.values[idx])
		return (y_hat,y_actual)
		
#folds are geneally how many times you wish to split the data. for eg for 3 folds we would hold 2/3 of data for training and 1/3 of for testing
	def plot_error_rates(self):
		folds=range(2,11)
		errors=pd.DataFrame({'max':0,'min':0},index=folds)
		for f in folds:
			error_rates=r.error_rate(f)
			errors['max'][f]=max(error_rates)
			errors['min'][f]=min(error_rates)
		errors.plot(title="Mean absolute error of KNN over different folds")
		plt.show()
		

