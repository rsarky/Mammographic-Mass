---
layout: page
title: Report
permalink: /report/
---

```python
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

cleanData = data.replace('?', np.nan).dropna().reset_index(drop=True)
cleanData = cleanData.astype('float64')



cleanData.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BIRADS</th>
      <th>Age</th>
      <th>Shape</th>
      <th>Margin</th>
      <th>Density</th>
      <th>Severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>830.000000</td>
      <td>830.000000</td>
      <td>830.000000</td>
      <td>830.000000</td>
      <td>830.000000</td>
      <td>830.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.393976</td>
      <td>55.781928</td>
      <td>2.781928</td>
      <td>2.813253</td>
      <td>2.915663</td>
      <td>0.485542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.888371</td>
      <td>14.671782</td>
      <td>1.242361</td>
      <td>1.567175</td>
      <td>0.350936</td>
      <td>0.500092</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>46.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>57.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>66.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.000000</td>
      <td>96.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop row containing BIRADS value as 55 which doesnt make sense.
toDrop = pd.Index(cleanData['BIRADS']).get_loc(55)
cleanData = cleanData.drop(toDrop).reset_index(drop=True)


```


```python
import matplotlib as plt
import seaborn as sns
sns.pairplot(cleanData, hue='Severity')

```




    <seaborn.axisgrid.PairGrid at 0x7f3cf7c3dbe0>




![png](output_2_1.png)


## Handle Categorical Data

### Attribute Information
> 6 Attributes in total (1 goal field, 1 non-predictive, 4 predictive attributes)
    1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!) 
    2. Age: patient's age in years (integer) 
    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal) 
    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal) 
    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal) 
    6. Severity: benign=0 or malignant=1 (binominal, goal field!) 
    
As we can see BI-RADS, Density (ordinal) and Shape, Margin(nominal) are all categorical in nature.
Thus we need to handle this type of data appropriately.


```python
print(cleanData.head())
```

       BIRADS   Age  Shape  Margin  Density  Severity
    0     5.0  67.0    3.0     5.0      3.0       1.0
    1     5.0  58.0    4.0     5.0      3.0       1.0
    2     4.0  28.0    1.0     1.0      3.0       0.0
    3     5.0  57.0    1.0     5.0      3.0       1.0
    4     5.0  76.0    1.0     4.0      3.0       1.0


From the data we can observe that all the categorical features have been transformed into ordinal features.
For attribues such as shape and margin which are nominal ordering does not make sense. Thus we one hot encode these attributes.

### Using One Hot Encoding to Handle Categorical data


```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
shapeFeatureArr = enc.fit_transform(cleanData[['Shape']])
shapeFeatureLabels = ['round', 'oval', 'lobular', 'irregular']
shapeFeature = pd.DataFrame(shapeFeatureArr, columns=shapeFeatureLabels)
shapeFeature

marginFeatureArr = enc.fit_transform(cleanData[['Margin']])
marginFeatureLabels = ['circumscribed', 'microlobulated', 'obscured', 'ill-defined', 'spiculated']
marginFeature = pd.DataFrame(marginFeatureArr, columns=marginFeatureLabels)
marginFeature

dfOHE = pd.concat([cleanData[['BIRADS', 'Age']], shapeFeature, marginFeature, cleanData[['Density','Severity']]],axis=1)
print('Nominal features are one hot encoded and ordinal features are left as is.')
dfOHE.head()
```

    Nominal features are one hot encoded and ordinal features are left as is.


    /home/rohit/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/_encoders.py:368: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)
    /home/rohit/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/_encoders.py:368: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BIRADS</th>
      <th>Age</th>
      <th>round</th>
      <th>oval</th>
      <th>lobular</th>
      <th>irregular</th>
      <th>circumscribed</th>
      <th>microlobulated</th>
      <th>obscured</th>
      <th>ill-defined</th>
      <th>spiculated</th>
      <th>Density</th>
      <th>Severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>





## Data processing

### Feature Normalisation
We can observe that the range of the `Age` feature differs from the other categorical features by a large margin. So we normalise our data. After normalising all features are standard normal with mean 0 and unit variance. The features normalisation will be added to `sci-kit learn` pipeline. It is incorrect to normalise the entire data beforehand as we are using knowledge about the test set to normalise. Therefore information about the test set will leak into the model which is not acceptable.

### Removing outliers
No significant outliers can be seen in the data.

### Splitting the Dataset
The dataset needs to be partitioned into training, testing and validation. The training set is used to train the model, the validation set is to optimise model parameters, the testing set is used to evaluate model performance on unseen data. Care needs to be taken so that no bias is introduced in the data.

Since the number of samples are limited (829) k-fold nested cross validation is chosen to be the method to chose an optimal model.

<img src="{{"/grid_search_cross_validation.png" | relative_url }}" alt="Drawing" style="width: 400px;" />

## Model Evaluation Metric

Models can be evaluated using a number of metrics like accuracy, precision, recall etc.
<img src="{{"/precision_recall.png" | relative_url }}" alt="Drawing" style="width: 600px;"/>

Since we do not want to falsely classify a malignant tumour as benign at any rate, or in other words we want to minimise the number of false negatives we should `recall` as our Model Evaluation Metric.



```python
# Evaluation metric is recall
metric = 'recall'

# Get Inputs and outputs.
X = pd.DataFrame(dfOHE.drop(['Severity'],axis=1))
y = dfOHE['Severity']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# StandardScaler Object to normalise our inputs.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

## Classifiers 
We will be considering three classifiers
- Logistic Regression
- Artificial Neural Network
- Support Vector Machine


## Logistic Regression


```python
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

# First we make a pipeline containing our StandardScaler object and our estimator ie LogisticRegression
clf = make_pipeline(scaler, LogisticRegression(random_state=0,solver='lbfgs'))

# Logistic Regression uses a regularisation hyperparameter 'C'. We find the optimal parameter
# using Cross Validation.
cparams = [ 10**i for i in range(-4,5) ]
params = [{'logisticregression__C': cparams}]
gridLR = GridSearchCV(clf, params, scoring=metric, cv=3)
gridLR.fit(X_train, y_train)
print('Best parameters and Best Score')
print(gridLR.best_params_, gridLR.best_score_)

print(classification_report(y_test, gridLR.predict(X_test)))
# # Nested cross validation
# results = cross_validate(grid, X, y, scoring=['recall', 'accuracy', 'precision'], cv=3)
# results_ = []
# for result in [results['test_accuracy'], results['test_precision'], results['test_recall']]:
#     mean = result.mean()
#     std = result.std()
#     results_.append((mean, std))
# for i,s in enumerate(['Accuracy', 'Precision', 'Recall']):
#     print("%s : %0.3f (+/-%0.03f)" % (s, results_[i][0], results_[i][1] * 2))
```

    Best parameters and Best Score
    {'logisticregression__C': 0.01} 0.871079476709014
                  precision    recall  f1-score   support
    
             0.0       0.82      0.76      0.79       134
             1.0       0.74      0.81      0.78       115
    
       micro avg       0.78      0.78      0.78       249
       macro avg       0.78      0.78      0.78       249
    weighted avg       0.79      0.78      0.78       249
    


## Neural Network


```python
from sklearn.neural_network import MLPClassifier
# Parameters to use to find optimal parameters using cross validation
params = {
    'mlpclassifier__hidden_layer_sizes': [(i,j) for i in range(1,10) for j in range(1,10)],
    'mlpclassifier__alpha': [i**10 for i in range(-4,3)]
}
clf = make_pipeline(scaler, MLPClassifier(solver='lbfgs',random_state=0))
gridNN = GridSearchCV(clf, parameter_space,scoring=metric,cv=3)

gridNN.fit(X_train, y_train)
print('Best parameters and Best Score')
print(gridNN.best_params_, gridNN.best_score_)

print(classification_report(y_test, gridNN.predict(X_test)))
# results = cross_validate(grid, X, y, scoring=['recall', 'accuracy', 'precision'], cv=3)
# results_ = []
# for result in [results['test_accuracy'], results['test_precision'], results['test_recall']]:
#     mean = result.mean()
#     std = result.std()
#     results_.append((mean, std))
# for i,s in enumerate(['Accuracy', 'Precision', 'Recall']):
#     print("%s : %0.3f (+/-%0.03f)" % (s, results_[i][0], results_[i][1] * 2))
```

    Best parameters and Best Score
    {'mlpclassifier__alpha': 0, 'mlpclassifier__hidden_layer_sizes': (3, 2)} 0.8885015880217785
                  precision    recall  f1-score   support
    
             0.0       0.80      0.80      0.80       134
             1.0       0.77      0.77      0.77       115
    
       micro avg       0.79      0.79      0.79       249
       macro avg       0.79      0.79      0.79       249
    weighted avg       0.79      0.79      0.79       249
    


    /home/rohit/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


## Support Vector Machine


```python
from sklearn.svm import SVC

clf = make_pipeline(scaler, SVC())
params = [{'svc__C':[10**i for i in range(-4,4)], 'svc__kernel':['linear', 'poly', 'rbf']}]

gridSVM = GridSearchCV(clf, params, scoring=metric,cv=3)
gridSVM.fit(X_train, y_train)
print('Best parameters and best score')
print(gridSVM.best_params_, gridSVM.best_score_)

print(classification_report(y_test, gridSVM.predict(X_test)))
# results = cross_validate(grid, X, y, scoring=['recall', 'accuracy', 'precision'], cv=3)
# results_ = []
# for result in [results['test_accuracy'], results['test_precision'], results['test_recall']]:
#     mean = result.mean()
#     std = result.std()
#     results_.append((mean, std))
# for i,s in enumerate(['Accuracy', 'Precision', 'Recall']):
#     print("%s : %0.3f (+/-%0.03f)" % (s, results_[i][0], results_[i][1] * 2))

```

    Best parameters and best score
    {'svc__C': 0.001, 'svc__kernel': 'linear'} 0.8885012099213553
                  precision    recall  f1-score   support
    
             0.0       0.84      0.73      0.78       134
             1.0       0.73      0.83      0.78       115
    
       micro avg       0.78      0.78      0.78       249
       macro avg       0.78      0.78      0.78       249
    weighted avg       0.79      0.78      0.78       249
    


    /home/rohit/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


We will choose the **neural network** as our model because it has higher recall than the other models.


```python
# Training the model on the entire data.
gridNN.fit(X,y)
```




    0.8408890409232487




```python
# Serializing the model to deploy it.
import pickle
pickle.dump(gridNN, open("modelNN.pkl", "wb"))
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BIRADS</th>
      <th>Age</th>
      <th>round</th>
      <th>oval</th>
      <th>lobular</th>
      <th>irregular</th>
      <th>circumscribed</th>
      <th>microlobulated</th>
      <th>obscured</th>
      <th>ill-defined</th>
      <th>spiculated</th>
      <th>Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



## 80% chance that if shape is 4 the cancer is malignant


```python
# sns.distplot(cleanData[['Shape']].loc[cleanData['Severity']==1])
# sns.distplot(cleanData['Shape'], rug=True)
# cleanData[['Shape']].plot()
for i in range(1,5):
    print(cleanData[['Shape']].loc[cleanData['Shape']==i].size)
print(cleanData[['Shape']].loc[(cleanData['Shape']==i) & (cleanData['Severity']==1)].size/378)

```

    190
    180
    81
    378
    0.7857142857142857


## No significant correlation observed between features.

Only Shape and Margin appear to be correlated with coefficient 0.7


```python
sns.heatmap(cleanData.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3d077a9b70>




![png](output_26_1.png)



```python
sns.distplot(cleanData['Age'])

```

    /home/rohit/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x7f3d04e4cd30>




![png](output_27_2.png)

