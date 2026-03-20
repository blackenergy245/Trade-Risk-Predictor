from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import joblib as jb

data = pd.read_csv('trade.csv')
df = pd.DataFrame(data)

X = df[['Origin_GDP_Billions' ,'Sanction_Status' , 'Commodity_Risk_Score' ,'Historical_Violation_Rate','Days_in_Transit']]
y = df['Risk_Score']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=32)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly',PolynomialFeatures()),
    ('model', Ridge())
])

grid_params = [{
    'model':[Lasso(),Ridge()],
    'model__alpha':[0.1,1,5,10],
    'poly__degree':[1,2,3,4]
},
{
    'model':[ElasticNet()],
    'model__alpha':[0.1,1,5,10],
    'model__l1_ratio':[0.2,0.5,0.9],
    'poly__degree':[1,2,3,4]
},
{
    'model':[KNeighborsRegressor()],
    'model__n_neighbors':[1,2,3,4],
    'model__weights':['uniform','distance'],
    'poly__degree':[1]
},
{
    'model':[DecisionTreeRegressor(random_state=54)],
    'model__min_samples_split':[2,3,4,5,6],
    'model__min_samples_leaf':[2,3,4,5,6],
    'model__max_depth':[5,10,15,20],
    'poly__degree':[1]
},
{
    'model':[RandomForestRegressor(random_state=43)],
    'model__min_samples_split':[2,3,4,5,6],
    'model__min_samples_leaf':[2,3,4,5,6],
    'model__n_estimators':[100,200,300],
    'poly__degree':[1]
}]

grid = GridSearchCV(pipe,grid_params,cv=5,scoring='r2')
grid.fit(X_train,y_train)

model = grid.best_estimator_
y_pred = model.predict(X_test)
print("Predictded score:",y_pred)
print("Actual:",y_test)
plt.scatter(x=y_test,y=y_pred,color = "Darkblue")
plt.xlabel("Actual score")
plt.ylabel("Predicted score")
plt.title("Comparision")
plt.show()

r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2:",r2)

print("--Best settings--")
print("Best parameters:",grid.best_params_)

jb.dump(model,'Trade predictor.pkl')
a = jb.load('Trade predictor.pkl')
test_data = {
    'Origin_GDP_Billions': [4500.0, 15.5, 850.0, 120.0, 3000.0],
    'Sanction_Status': [0, 1, 0, 1, 0], 
    'Commodity_Risk_Score': [1.5, 9.5, 5.0, 2.0, 8.5],
    'Historical_Violation_Rate': [0.01, 0.18, 0.05, 0.02, 0.03],
    'Days_in_Transit': [5, 45, 20, 15, 10]
}
new = pd.DataFrame(test_data)
new_predictions = a.predict(new)
print("New predictions:",new_predictions)