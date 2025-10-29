#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Hello! This is the final version of the code dealing with the Tree-Based Algorithms. 
Do not forget to change the working directory by choosing the preferred os.chdir().

Additionally, I clean the data using the functions clean and clean2 defined in the scripts DS_ML_Data 
and DS_ML_Data2. 

This code should take a long time to compile (probably few hours). 

If you have any questions, please do ask me.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

from hajibeygi import armin

import random
random.seed(42)

import matplotlib.pyplot as plt 
import seaborn as sns


import pickle
import os

os.chdir('/Users/hiro/DS-ML')
from DS_ML_Data import clean
#from DS_ML_Data_2 import clean2

#dd = clean2()
d, train_df, test_df = clean()

X = d[['x', 'y', 'p']]
y = d['ce']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
rf = RandomForestRegressor()

# Hyperparameter tuning (optional but recommended)
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

#"""
grid_search_rf = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)


best_rf = grid_search_rf.best_estimator_


pickle.dump(best_rf, open('best_rf.sav', 'wb'))
pickle.dump(grid_search_rf, open('grid_search_rf.sav', 'wb')) 
#"""

best_rf = pickle.load(open('best_rf.sav', 'rb'))
grid_search_rf = pickle.load(open('grid_search_rf.sav', 'rb'))


# Predictions
y_pred = best_rf.predict(X_test)

cv_results_rf = pd.DataFrame(grid_search_rf.cv_results_)

# View results

cv_results_rf = cv_results_rf[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score', 'std_test_score', 'param_min_samples_leaf']]
print(cv_results_rf[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score', 'std_test_score', 'param_min_samples_leaf']])


#[100, 10, 5]

print(grid_search_rf.best_estimator_)

# Evaluation
mse_rf = mean_squared_error(y_test, y_pred)
mae_rf = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse_rf}")
print(f"Mean Absolute Error: {mae_rf}")

cv_results_rf = pd.DataFrame(grid_search_rf.cv_results_)
print(cv_results_rf[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score', 'param_min_samples_leaf']])

sorted_cv_results_rf = cv_results_rf.sort_values(by='mean_test_score', ascending=False)
print(sorted_cv_results_rf.head())



plt.figure(figsize=(10, 6))
sns.lineplot(x='param_n_estimators', y='mean_test_score', data=cv_results_rf)
plt.fill_between(cv_results_rf['param_n_estimators'],
                 cv_results_rf['mean_test_score'] - cv_results_rf['std_test_score'],
                 cv_results_rf['mean_test_score'] + cv_results_rf['std_test_score'],
                 alpha=0.2)
plt.title('Effect of n_estimators on Model Performance')
plt.xlabel('n_estimators')
plt.ylabel('Mean Test Score (Negative MSE)')
plt.show()


# Pivot the DataFrame to create a heatmap-friendly format
#heatmap_data = cv_results_rf.pivot('param_max_depth', 'param_n_estimators', 'mean_test_score', 'param_min_samples_leaf')
#heatmap_data = cv_results_rf.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

heatmap_data = cv_results_rf.pivot_table(
    values='mean_test_score', 
    index=['param_max_depth', 'param_min_samples_leaf'], 
    columns='param_n_estimators', 
    aggfunc='mean', 
    fill_value=0
)

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4g", cmap="coolwarm")
plt.title('Hyperparameter Heatmap of Mean Test Score (Negative MSE)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='param_max_depth', y='mean_test_score', data=cv_results_rf)
plt.title('Boxplot of Mean Test Scores for Different max_depth')
plt.xlabel('max_depth')
plt.ylabel('Mean Test Score (Negative MSE)')
plt.show()

##########################################################################################################
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 6, 9, 12],
    'subsample': [0.6, 0.8, 1.0]
}

# Perform Grid Search with Cross-Validation to find the best parameters


grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

#"""
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_

pickle.dump(best_xgb, open('best_xgb.sav', 'wb'))
pickle.dump(grid_search_xgb, open('grid_search_xgb.sav', 'wb'))
#"""


best_xgb = pickle.load(open('best_xgb.sav', 'rb'))
grid_search_xgb = pickle.load(open('grid_search_xgb.sav', 'rb'))




# Make predictions
y_pred_xgb = best_xgb.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error: {mse_xgb:.4f}")
print(f"XGBoost Mean Absolute Error: {mae_xgb:.4f}")


cv_results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)

# View results
print(cv_results_xgb[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']])

print(grid_search_xgb.best_estimator_)
############################################################################################################

from catboost import CatBoostRegressor


catboost_model = CatBoostRegressor(verbose=0, random_state=42)

#"""
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
pickle.dump(catboost_model, open('catboost.sav', 'wb'))
#"""

catboost_model = pickle.load(open('catboost.sav', 'rb'))
y_pred_catboost = catboost_model.predict(X_test)

mse_catboost = mean_squared_error(y_test, y_pred_catboost)
mae_catboost = mean_absolute_error(y_test, y_pred_catboost)

print(f"CatBoost Mean Squared Error: {mse_catboost:.4f}")
print(f"CatBoost Mean Absolute Error: {mae_catboost:.4f}")


print(catboost_model.get_params())

####################################################################################

import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(random_state=42)
param_grid_lgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'num_leaves': [20, 31, 50],
    'max_depth': [-1, 10, 20]
}


#"""
grid_search_lgb = GridSearchCV(lgb_model, param_grid_lgb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_lgb.fit(X_train, y_train)




best_lgb = grid_search_lgb.best_estimator_



pickle.dump(best_lgb, open('best_lgb.sav', 'wb'))
pickle.dump(grid_search_lgb, open('grid_search_lgb.sav', 'wb'))


#"""
best_lgb = pickle.load(open('best_lgb.sav', 'rb'))
grid_search_lgb = pickle.load(open('grid_search_lgb.sav', 'rb'))



y_pred_lgb = best_lgb.predict(X_test)

mse_lgb = mean_squared_error(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)

print(f"LightGBM Mean Squared Error: {mse_lgb:.4f}")
print(f"LightGBM Mean Absolute Error: {mae_lgb:.4f}")


####################################################################################

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9]
}


#"""
grid_search_gb = GridSearchCV(gb_model, param_grid_gb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_gb = grid_search_gb.best_estimator_


pickle.dump(best_gb, open('best_gb.sav', 'wb'))
pickle.dump(grid_search_gb, open('grid_search_gb.sav', 'wb'))


#"""

best_gb = pickle.load(open('best_gb.sav', 'rb'))
grid_search_gb = pickle.load(open('grid_search_gb.sav', 'rb'))


y_pred_gb = best_gb.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print(f"Gradient Boosting Mean Squared Error: {mse_gb:.4f}")
print(f"Gradient Boosting Mean Absolute Error: {mae_gb:.4f}")


print("Best Parameters for Gradient Boosting:", grid_search_gb.best_params_)

###################################################

print(f"XGBoost Mean Squared Error: {mse_xgb:.4f}")
print(f"XGBoost Mean Absolute Error: {mae_xgb:.4f}")
print(f"CatBoost Mean Squared Error: {mse_catboost:.4f}")
print(f"CatBoost Mean Absolute Error: {mae_catboost:.4f}")
print(f"LightGBM Mean Squared Error: {mse_lgb:.4f}")
print(f"LightGBM Mean Absolute Error: {mae_lgb:.4f}")
print(f"Gradient Boosting Mean Squared Error: {mse_gb:.4f}")
print(f"Gradient Boosting Mean Absolute Error: {mae_gb:.4f}")




####################################################3



"""
cv_results_gb = pd.DataFrame(grid_search_gb.cv_results_)
print(cv_results_gb[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']])

cv_results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
print(cv_results_xgb[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']])

cv_results_lgb = pd.DataFrame(grid_search_lgb.cv_results_)
print(cv_results_lgb[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']])

sorted_cv_results = cv_results_rf.sort_values(by='mean_test_score', ascending=False)
print(sorted_cv_results.head())


plt.figure(figsize=(10, 6))
sns.lineplot(x='param_n_estimators', y='mean_test_score', data=cv_results)
plt.fill_between(cv_results['param_n_estimators'],
                 cv_results['mean_test_score'] - cv_results['std_test_score'],
                 cv_results['mean_test_score'] + cv_results['std_test_score'],
                 alpha=0.2)
plt.title('Effect of n_estimators on Model Performance')
plt.xlabel('n_estimators')
plt.ylabel('Mean Test Score (Negative MSE)')
plt.show()

"""




# Create a dictionary to store the results for different models
model_results = {
    'Random Forest': mse_rf,
    'XGBoost': mse_xgb,
    'CatBoost': mse_catboost,
    'LightGBM': mse_lgb,
    'Gradient Boosting': mse_gb
}

# Convert to a DataFrame
results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'MSE'])

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title('Comparison of Model Performance (MSE)')
plt.ylabel('Mean Squared Error')
plt.show()

residuals_rf = y_test - y_pred
residuals_xgb = y_test - y_pred_xgb

plt.figure(figsize=(8, 6))
sns.histplot(residuals_rf, kde=True, label='Random Forest', color='blue')
sns.histplot(residuals_xgb, kde=True, label='XGBoost', color='red')
plt.title('Residuals Distribution for Random Forest and XGBoost')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Saving the plot
plt.savefig('residuals_rf_vs_xgb.pdf')

# Showing the plot
plt.show()

importances = best_rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance - Random Forest')
plt.show()

y_train_pred_rf = best_rf.predict(X_train)
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)

print(f"Train Mean Squared Error (Random Forest): {mse_train_rf:.4f}")
print(f"Test Mean Squared Error (Random Forest): {mse_rf:.4f}")




predictions_comparison_df = pd.DataFrame({
    'Actual': y_test,
    'RandomForest': y_pred,
    'XGBoost': y_pred_xgb,
    'CatBoost': y_pred_catboost,
    'LightGBM': y_pred_lgb,
    'GradientBoosting': y_pred_gb
})



pred_diff_rf_xgb = predictions_comparison_df['RandomForest'] - predictions_comparison_df['XGBoost']

# Plot the difference
plt.figure(figsize=(10, 6))
sns.histplot(pred_diff_rf_xgb, kde=True, color="blue", bins=30)
plt.title("Difference Between Random Forest and XGBoost Predictions")
plt.xlabel("Difference (Random Forest - XGBoost)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()












############
#%%
"""

from sklearn.feature_extraction import FeatureHasher

X_2 = d[['x', 'y', 'p', 'subject_global']]
y_2 = d['ce']

# Split the data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

X_train_2_dict = X_train_2[['subject_global']].astype(str).to_dict(orient='records')
X_test_2_dict = X_test_2[['subject_global']].astype(str).to_dict(orient='records')

hasher = FeatureHasher(n_features=10, input_type='string')

# Transform the 'subject_global' column
hashed_train_features = hasher.transform(X_train_2_dict)
hashed_test_features = hasher.transform(X_test_2_dict)

# Convert to dense matrix and combine with other features
X_train_2_hashed = pd.DataFrame(hashed_train_features.toarray())
X_test_2_hashed = pd.DataFrame(hashed_test_features.toarray())

# Add the other features (x, y, p)
X_train_2_final = pd.concat([X_train_2_hashed.reset_index(drop=True), X_train_2[['x', 'y', 'p']].reset_index(drop=True)], axis=1)
X_test_2_final = pd.concat([X_test_2_hashed.reset_index(drop=True), X_test_2[['x', 'y', 'p']].reset_index(drop=True)], axis=1)


##############################################################
# Model initialization
rf_2 = RandomForestRegressor()

# Hyperparameter tuning (optional but recommended)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf_2, param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_2, y_train_2)

# Use the best estimator
best_rf_2 = grid_search.best_estimator_

# Predictions
y_pred = best_rf_2.predict(X_test_2_final)

# Evaluation
mse_rf_2 = mean_squared_error(y_test, y_pred)
mae_rf_2 = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse_rf_2}")
print(f"Mean Absolute Error: {mae_rf_2}")



##########################################################################################################
import xgboost as xgb

xgb_model_2 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
}

# Perform Grid Search with Cross-Validation to find the best parameters
grid_search_xgb_2 = GridSearchCV(xgb_model_2, param_grid_xgb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_xgb_2.fit(X_train_2, y_train_2)

# Use the best estimator from the grid search
best_xgb_2 = grid_search_xgb.best_estimator_

# Make predictions
y_pred_xgb_2 = best_xgb.predict(X_test_2)

# Evaluate the model
mse_xgb_2 = mean_squared_error(y_test_2, y_pred_xgb_2)
mae_xgb_2 = mean_absolute_error(y_test_2, y_pred_xgb_2)

print(f"XGBoost Mean Squared Error: {mse_xgb_2:.4f}")
print(f"XGBoost Mean Absolute Error: {mae_xgb_2:.4f}")


############################################################################################################

from catboost import CatBoostRegressor

catboost_model_2 = CatBoostRegressor(verbose=0, random_state=42)
catboost_model_2.fit(X_train_2, y_train_2)
y_pred_catboost_2 = catboost_model_2.predict(X_test_2)

mse_catboost_2 = mean_squared_error(y_test_2, y_pred_catboost_2)
mae_catboost_2 = mean_absolute_error(y_test_2, y_pred_catboost_2)

print(f"CatBoost Mean Squared Error: {mse_catboost_2:.4f}")
print(f"CatBoost Mean Absolute Error: {mae_catboost_2:.4f}")



####################################################################################

import lightgbm as lgb

lgb_model_2 = lgb.LGBMRegressor(random_state=42)
param_grid_lgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [20, 31, 50],
    'max_depth': [-1, 10, 20]
}

grid_search_lgb_2 = GridSearchCV(lgb_model, param_grid_lgb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_lgb_2.fit(X_train_2, y_train_2)

best_lgb_2 = grid_search_lgb.best_estimator_
y_pred_lgb_2 = best_lgb.predict(X_test_2)

mse_lgb_2 = mean_squared_error(y_test_2, y_pred_lgb_2)
mae_lgb_2 = mean_absolute_error(y_test_2, y_pred_lgb_2)

print(f"LightGBM Mean Squared Error: {mse_lgb_2:.4f}")
print(f"LightGBM Mean Absolute Error: {mae_lgb_2:.4f}")


####################################################################################

from sklearn.ensemble import GradientBoostingRegressor

gb_model_2 = GradientBoostingRegressor(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search_gb_2 = GridSearchCV(gb_model_2, param_grid_gb, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_gb_2.fit(X_train, y_train)

best_gb_2 = grid_search_gb_2.best_estimator_
y_pred_gb_2 = best_gb.predict(X_test_2)

mse_gb_2 = mean_squared_error(y_test_2, y_pred_gb_2)
mae_gb_2 = mean_absolute_error(y_test_2, y_pred_gb_2)

print(f"Gradient Boosting Mean Squared Error: {mse_gb_2:.4f}")
print(f"Gradient Boosting Mean Absolute Error: {mae_gb_2:.4f}")




###################################################

print(f"XGBoost Mean Squared Error: {mse_xgb_2:.4f}")
print(f"XGBoost Mean Absolute Error: {mae_xgb_2:.4f}")
print(f"CatBoost Mean Squared Error: {mse_catboost_2:.4f}")
print(f"CatBoost Mean Absolute Error: {mae_catboost_2:.4f}")
print(f"LightGBM Mean Squared Error: {mse_lgb_2:.4f}")
print(f"LightGBM Mean Absolute Error: {mae_lgb_2:.4f}")
print(f"Gradient Boosting Mean Squared Error: {mse_gb_2:.4f}")
print(f"Gradient Boosting Mean Absolute Error: {mae_gb_2:.4f}")


"""
