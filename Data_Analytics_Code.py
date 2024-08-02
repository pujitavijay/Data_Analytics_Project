#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import r2_score


# In[121]:


data = pd.read_csv("train.csv")


# In[128]:


x= data.drop(['id'], axis=1)


# In[129]:


train=x.iloc[:7000,:]
test=x.iloc[7000:,:]
X_train = train.drop(['Hardness'],axis = 1)
y_train= train['Hardness']
X_test = test.drop(['Hardness'],axis = 1)
y_test= test['Hardness']


# In[130]:


pr = PolynomialFeatures(degree = 2)
X_poly = pr.fit_transform(X_test)
lr_2 = LinearRegression()
lr_2.fit(X_poly, y_test)


# In[29]:


y_pred_poly = lr_2.predict(X_poly) 


# In[32]:


mae = mean_absolute_error(y_test,y_pred_poly)


# In[33]:


mae


# In[11]:


X_test


# In[12]:


y_test


# In[ ]:





# In[13]:


X_train


# In[14]:


X_test


# In[15]:


X_test


# In[16]:


y_train


# In[17]:


y_test


# In[18]:


# Plotting scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, label='Actual vs Predicted', marker='o')

# Adding a straight line representing perfect prediction
min_val = min(np.min(y_test), np.min(y_test))
max_val = max(np.max(y_test), np.max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[137]:


y_pred_poly.size


# In[67]:


rfc_baseline_model = RandomForestRegressor(random_state=1, n_estimators=50).fit(X_train, y_train)
preds_baseline_rfr = rfc_baseline_model.predict(X_test)


# In[68]:


# Plotting scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test,preds_baseline_rfr , label='Actual vs Predicted', marker='o')

# Adding a straight line representing perfect prediction
min_val = min(np.min(y_test), np.min(preds_baseline_rfr))
max_val = max(np.max(y_test), np.max(preds_baseline_rfr))
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[69]:


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test,y_pred_poly , label='Actual vs Predicted', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()

# Show the plot
plt.show()



# In[77]:


mae = mean_absolute_error(y_test,y_pred )
print(f"Mean Absolute Error (MAE): {mae}")

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")




# Accuracy
accuracy = accuracy_score(y_test,preds_baseline_rfr)
print(f"Accuracy: {accuracy}")


# In[71]:


preds_baseline_rfr


# In[90]:


model =RandomForestRegressor()



# In[91]:


for name, model in algorithms.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    


# In[40]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test,y_pred , label='Actual vs Predicted', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[43]:


numerical_columns = ['allelectrons_Total', 'density_Total', 'allelectrons_Average', 'val_e_Average',
                      'atomicweight_Average', 'ionenergy_Average', 'el_neg_chi_Average',
                      'R_vdw_element_Average', 'R_cov_element_Average', 'zaratio_Average',
                      'density_Average']

from scipy import stats

z_scores = np.abs(stats.zscore(X_train[numerical_columns]))

threshold = 3

outlire_indices = np.where(z_scores>threshold)[0]

X_train = X_train.drop(X_train.index[outlire_indices])
y_train = y_train.drop(y_train.index[outlire_indices])


# In[44]:


X_train


# In[48]:


y_test


# In[62]:


python_list = y_pred_poly.tolist()
y_test = y_test.tolist()


# In[63]:


accuracy = accuracy_score(y_test,python_list )
print(f"Accuracy: {accuracy}")


# In[64]:


transpose_matrix


# In[56]:


y_pred_poly


# In[61]:


python_list


# In[95]:


r2 = r2_score(y_test, y_test)


# In[96]:


r2


# In[97]:


mae = mean_absolute_error(y_test,y_test)
print(f"Mean Absolute Error (MAE): {mae}")

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_test)
print(f"Mean Squared Error (MSE): {mse}")


# In[111]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate the correlation matrix
correlation_matrix = x.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn_r', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# In[125]:


# Extract correlation values for each feature with the target
correlation_with_target = correlation_matrix['Hardness']

colors = ['red' if corr < 0 else 'green' for corr in correlation_with_target]
correlation_with_target.drop('Hardness').plot(kind='bar', color=colors)
plt.title("Correlation with Target (Hardness)")
plt.ylabel("Correlation Coefficient")
plt.show()


# In[131]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances.sort_values(ascending=False, inplace=True)
feature_importances.plot(kind='bar')


# In[132]:


among these which has less error in prediction Forest Regression
KNN(n=5)
Polynomial
Linear Regression
Ideal


# In[133]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_train and y_train are your training data and labels
# You may need to split your dataset into training and testing sets

# Define a range of k values to test
k_values = list(range(1, 21))  # You can adjust this range based on your problem

# Create an empty list to store cross-validation scores
cv_scores = []

# Perform k-fold cross-validation for each k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Using 10-fold cross-validation, you can adjust the number of folds
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(np.mean(scores))

# Plot the cross-validation scores for each k
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal k for KNN')
plt.show()

# Find the optimal k value with the highest cross-validation score
optimal_k = k_values[np.argmax(cv_scores)]
print(f'Optimal k value: {optimal_k}')


# In[139]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Assuming X and y are your feature matrix and target variable
# You may need to split your data into training and testing sets first

# Create a range of k values to test
k_values = np.arange(1, 21)

# Initialize an empty list to store mean cross-validation scores
mean_scores = []

# Perform k-fold cross-validation for each k
for k in k_values:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    # Use 5-fold cross-validation (you can adjust as needed)
    scores = cross_val_score(knn_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_scores.append(-np.mean(scores))  # Convert to positive mean squared error

# Find the optimal k value that minimizes mean squared error
optimal_k = k_values[np.argmin(mean_scores)]

print(f"The optimal k value is: {optimal_k}")


# In[140]:


import matplotlib.pyplot as plt

# Plot cross-validation scores for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation Scores for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (Cross-Validation Score)')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[141]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

# Assuming X and y are your feature matrix and target variable
# You may need to split your data into training and testing sets first

# Choose a range of k values
k_values = np.arange(1, 21)

# Initialize an empty list to store mean cross-validation scores
mean_scores = []

# Perform k-fold cross-validation for each k
for k in k_values:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    # Use 5-fold cross-validation (you can adjust as needed)
    scores = cross_val_score(knn_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_scores.append(-np.mean(scores))  # Convert to positive mean squared error

# Find the optimal k value that maximizes the cross-validation score
optimal_k = k_values[np.argmin(mean_scores)]

print(f"The optimal k value is: {optimal_k}")

# Plot cross-validation scores for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation Scores for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (Cross-Validation Score)')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[ ]:




