#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#Loading the House Price Dataset from sklearn datasets
house_price_dataset = sklearn.datasets.fetch_california_housing()

#Loading the dataset to pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data,columns=house_price_dataset.feature_names)

#Print the first 5 rows of the dataset
# print(house_price_dataframe.head(5))

#Add the target column into the dataframe
house_price_dataframe['Price'] = house_price_dataset.target

#Print the first 5 rows of the dataset
# print(house_price_dataframe.head(5))

#Number of rows and columns
# print(house_price_dataframe.shape)

#Check missing values
# print(house_price_dataframe.isnull().sum())

#Statistical Measures 
# print(house_price_dataframe.describe())

#Understanding the Correlation between various features in the dataset
# 1. Positive Correlation
# 2. Negative Correlation
correlation = house_price_dataframe.corr()

#Constructing a heatmap to understand the correlation
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()

#Dropping the columns and Splitting the data/target
X=house_price_dataframe.drop(['Price','Population','AveBedrms','AveOccup'],axis=1)
Y=house_price_dataframe['Price']

#Splitting the dataset into training and test dataset
#In regression problemns, dont use stratify
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

#Loading the model
xgboost_REG = XGBRegressor()

#Model Training
xgboost_REG.fit(x_train,y_train)

#Model Prediction
y_train_pred = xgboost_REG.predict(x_train)
y_test_pred = xgboost_REG.predict(x_test)

#Model Prediction
#R Squared Error
score_train_1 = metrics.r2_score(y_train, y_train_pred)
score_test_1 = metrics.r2_score(y_test, y_test_pred)

#Mean Absolute Error
score_train_2 = metrics.mean_absolute_error(y_train, y_train_pred)
score_test_2 = metrics.mean_absolute_error(y_test, y_test_pred)

print(f'TRAIN SET::: RSqaured: {score_train_1}, MAE: {score_train_2}')
print(f'TEST SET::: RSqaured: {score_test_1}, MAE: {score_test_2}')

#Visualizing the actual price and predicted price
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Vs. Predicted Price')
plt.show()