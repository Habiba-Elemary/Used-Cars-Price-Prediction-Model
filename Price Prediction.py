import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import  Pipeline


#Loading the dataset, checking null values
data= pd.read_csv('car data.csv')
print(data.head())
print(data.columns)
print(data.info())
print(data.describe())
nulls= data.isnull().sum()
print(nulls)


#Graph 1: Distribution of selling price
sns.set_style(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(data['Selling_Price'], kde=True, bins=30)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()


#Graph 2: Car age vs. selling price

#calculating car age from the year column
data['Car_Age']= 2025 - data['Year']
plt.figure(figsize=(10,6))
sns.scatterplot(x='Car_Age', y='Selling_Price', data=data)
plt.title('Car Age vs. Selling Price')
plt.xlabel('Car Age in Years')
plt.ylabel('Selling Price')
plt.show()


#Graph 3: Present prics vs selling price
plt.figure(figsize=(10,6))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data)
plt.title('Present Price vs. Selling Price')
plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.show()


#Graph 4: Driven Kms vs selling price
plt.figure(figsize=(10,6))
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=data)
plt.title('Kilometers Driven vs. Selling Price')
plt.xlabel('Driven kms')
plt.ylabel('Selling Price')
plt.show()


#Graph 5: Top 20 car models
n=20
topmodels= data['Car_Name'].value_counts().head(n)
plt.figure(figsize=(10,6))
sns.barplot(x=topmodels.values, y=topmodels.index, palette='viridis')
plt.title(f'Top {n} Car Models by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Car Model')
plt.show()


#Graph 6: Top 20 car models by avg price
avg= data.groupby('Car_Name')['Present_Price'].mean().sort_values(ascending=False)
topmodels= avg.head(n)
plt.figure(figsize=(10,6))
sns.barplot(x=topmodels.values, y=topmodels.index, palette='magma')
plt.title(f'Top {n} Car Models by Average Price')
plt.xlabel('Average Price')
plt.ylabel('Car Model')
plt.show()


#Separating features from target variables
x= data.drop(columns=['Selling_Price']) #feature
y=data['Selling_Price'] #target variable


#identifying categorical and numerical columns
categorical= [col for col in x.columns if x[col].dtype == 'object']
numerical= [col for col in x.columns if x[col].dtype in ['int64', 'float64']]

#training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)


#defining the model
# noinspection PyCallingNonCallable
preprocessor= ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])


#defining models to test
models= {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', LinearRegression())]),
                                  'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                                                   ('regressor',
                                                                    RandomForestRegressor(n_estimators=100, random_state=42))])
}


#training the model
results={}
for model_name, model in models.items():
    model.fit(xtrain, ytrain)


#predicting the model
ypredict= model.predict(xtest)


#evaluating the model
mae= mean_absolute_error(ytest, ypredict)
mse= mean_squared_error(ytest, ypredict)
rmse= np.sqrt(mse)
r2= r2_score(ytest, ypredict)


#storing results
results[model_name]= {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
print(f'Results for {model_name}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}\n')


#saving the model
joblib.dump(model, 'car data.csv')