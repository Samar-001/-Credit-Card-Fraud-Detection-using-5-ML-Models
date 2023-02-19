# SAMPLING: Credit Card Fraud Detection
## Using 5 sampling methods to predict the result with the help of 5 ML models
### _Sampling Methods used: Random, Systematic, Stratified, Clustered, Weighted_
### _ML Models used: Logistic Regression, Decision Trees, Random Forest, Naive Bayes, K-Nearest Neighbor_

<br>

Done By: **Samarjot Singh  102003242**

***


### Importing libraries & dataset

```Python
import numpy as np
import pandas as pd
from sklearn import datasets
```
```Python
rdf = pd.read_csv('./Creditcard_data.csv')
```

<br>

### Checking for imbalanced data

When examining our dataset, it becomes apparent that there is a significant class imbalance issue, where credit card fraud transactions only make up 1.2% of the data while genuine transactions constitute 98.8%. If we do not account for this imbalance during model training, the model will prioritize genuine transactions due to their larger representation in the data, which can result in high accuracy but poor fraud detection performance.

![Raw Data Pie Chart](https://raw.githubusercontent.com/Samar-001/-Credit-Card-Fraud-Detection-using-5-ML-Models/main/images/raw_pie_chart.png "Raw Data Pie Chart")

One way to address imbalanced datasets is to oversample the minority class. We can generate new instances by replicating existing ones. The Synthetic Minority Oversampling Technique, also known as SMOTE, is a widely-used data augmentation technique for the minority class.

We can utilize the SMOTE implementation in the imblearn package to resample our data and address the class imbalance issue.

```Python
from imblearn.over_sampling import SMOTE
su = SMOTE(random_state=42)
X_su, Y_su = su.fit_resample(x, y)
```

After resampling, let's verify if the data is balanced:

![Balanced Data Pie Chart](https://raw.githubusercontent.com/Samar-001/-Credit-Card-Fraud-Detection-using-5-ML-Models/main/images/balanced_pie_chart.png.png "Balanced Data Pie Chart")

<br>

### Scaling `Amount` & `Time` columns

Scaling is a crucial preprocessing step, particularly for the 'Amount' and 'Time' columns. These two features have different scales and units, which can result in biased results during model training. Scaling techniques such as min-max scaling and standardization can be used to normalize or standardize these features and ensure they contribute equally to the model's decision-making process. The Amount column has outliers, that's why we chose RobustScaler() as it's robust to outliers.

```Python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(df[["Time", "Amount"]])
df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
```

<br>

### Applying different sampling methods

#### Random Sampling
```Python
random_df = df.sample(n=382, random_state=42)
```

#### Systematic Sampling
```Python
indexes = np.arange(0, len(df), step=4)
systematic_df = df.iloc[indexes]
```

#### Stratified Sampling
```Python
stratified_df = df.groupby('AmountGroup', group_keys=False).apply(lambda x: x.sample(frac=0.25))
```

#### Cluster Sampling
```Python
def clustered_Sample(df, rows_per_cluster, no_of_clusters):
    K = int(len(df)/rows_per_cluster)
    cl_data = None
    for k in range(K):
        sample_cl = df.sample(rows_per_cluster)
        sample_cl["Cluster"] = np.repeat(k,len(sample_cl))
        df = df.drop(index = sample_cl.index)
        cl_data = pd.concat([cl_data,sample_cl],axis = 0)

    random_clusters = np.random.randint(0,K,size = no_of_clusters)
    samples = cl_data[cl_data.Cluster.isin(random_clusters)]
    return(samples)

clustered_df = clustered_Sample(df = df, rows_per_cluster = 96, no_of_clusters = 4)
```

<br>

#### Weighted Sampling
```Python
df['weights'] = df['AmountGroup'].map({0: 20, 1: 60, 2: 20})
weighted_df = df.sample(n=382, weights='weights')
```

<br>

### Applying different machine learning algorithms

#### Creating dictionary containing five ML models
```Python
models = {}

from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()
```

<br>

### Applying all ML models on a sample
```Python
sample_accuracy, sample_precision, sample_recall = {}, {}, {}

for key in models.keys():
    models[key].fit(sample_X_train, sample_Y_train)
    sample_predictions = models[key].predict(X_test)
    
    sample_accuracy[key] = accuracy_score(sample_predictions, Y_test)
```

<br>

### Combining accuracies from all the samples & models in form of a table
```Python
accuracy_table = pd.DataFrame()
accuracy_table['Random Sampling'] = random_accuracy
accuracy_table['Systematic Sampling'] = systematic_accuracy
accuracy_table['Stratified Sampling'] = stratified_accuracy
accuracy_table['Clustered Sampling'] = clustered_accuracy
accuracy_table['Weighted Sampling'] = weighted_accuracy
```

We recieved the following table comparing accuracies of all five ML models applied on all five samples created using various sampling methods.

Model\Sample        | Random Sampling | Systematic Sampling | Stratified Sampling | Clustered Sampling | Weighted Sampling
------------------- | --------------- | ------------------- | ------------------- | ------------------ | -----------------
Logistic Regression |  0.916776	      |  0.932503           |  0.925950           |  0.918087          |  0.893840
Decision Trees      |  0.967235	      |  0.963958           |  0.971166           |  0.967235          |  0.945609
Random Forest       |**0.993447**     |  0.991481           |  0.992136           |  0.990826          |  0.992136
Naive Bayes         |  0.872870	      |  0.779817           |  0.816514           |  0.828309          |  0.838794
K-Nearest Neighbor  |  0.899083	      |  0.904325           |  0.918087           |  0.917431          |  0.908257

From the table, it is evident that highest accuracy has been achieved using `Random Forest Classifier` on `Random Sample`. Although the results may vary everytime you run the code.
