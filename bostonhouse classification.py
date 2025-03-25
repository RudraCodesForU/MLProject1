## Importing libraries for this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn as sk
## LOADING THE DATASET
columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO",
         "B","LSTAT","MEDV"]
## Load the data
boston=pd.read_csv('BostonHousing.csv')
boston.head()
## Visualization
boston.describe()
## Visualization of the houses with its pricing
sns.regplot(x='RM',y='MEDV',data=boston,fit_reg=True)
plt.title("Relationship between rooms and price")
plt.show()
## Visualization of the pricing with population
sns.regplot(x='LSTAT',y='MEDV',data=boston,fit_reg=True)
plt.title("Relationship between No of People residing and price")
plt.show()
## Prediction of the pricing over the standard variables
boston_selected_var_df=boston.iloc[:,[0,4,5,7,10,12]]
boston_selected_var_df.corr()
boston_selected_var_df['PRICE']=boston.MEDV
from statsmodels.formula.api import ols
model=ols('PRICE ~ CRIM + NOX + RM + DIS + PTRATIO + LSTAT',boston_selected_var_df).fit()
print(model.summary())
## Split the data to train and test the pricing
X=boston.drop(columns=["MEDV","RAD"],axis=1)
Y=boston['MEDV']
## Model Training
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
def train(model,X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42)
    model.fit(X_train,Y_train)
    pred=model.predict(X_test)
    cv_score=cross_val_score(model,X,Y,scoring='neg_mean_squared_error',cv=5)
    cv_score=np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE:",mean_squared_error(Y_test,pred)) 
    print("CV Score:",cv_score)
## Model Testing
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
train(model,X,Y)
coef=pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True)
coef.plot(kind='bar',title='Listing of Houses according the Features')
