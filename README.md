# EXNO:4-Feature Scaling and Selection
## Name: POZHILAN V D
## Register NO: 212223240118
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:

##### STEP 1:Read the given Data.
##### STEP 2:Clean the Data Set using Data Cleaning Process.
##### STEP 3:Apply Feature Scaling for the feature in the data set.
##### STEP 4:Apply Feature Selection for the feature in the data set.
##### STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/576db0f0-b3af-40dd-8b87-ae3a3467c39f)

```python
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/2e68c12f-c709-47fe-a0f0-c7db5533b2fa)

```python
df.dropna()
```
![image](https://github.com/user-attachments/assets/a68a469a-bed2-4059-b7b6-a8ad994ebce8)

```python
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/a6569c54-df5a-4da1-9b42-f4dd2548c55a)

```python
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/7f397797-7518-49f6-ba39-83621b2c1f70)

```python
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/1138b202-48df-41f5-bc56-65582a854a90)

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/bd2dce7a-2b1e-4625-a5b4-e084f161df42)

```python
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()

df3=pd.read_csv("bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/8be79e3a-8bb6-4f72-8d33-23187a7a4acc)

```python
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/943b2792-5c97-49ba-8738-67116841280c)

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/4160e71f-083b-430f-941a-1c66573c0bf1)

```python
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/42239c52-2e24-4890-8d5d-cda5470f3a9b)

```python
import pandas as pd
df=pd.read_csv("income.csv")
df.info()
```
![image](https://github.com/user-attachments/assets/898734bc-6e8d-4e52-ba3d-dacc5f8fe72a)

```python
df
```
![image](https://github.com/user-attachments/assets/47cf2608-a6a1-40bb-af03-a50b4e5f25d5)

```python
df.info()
```
![image](https://github.com/user-attachments/assets/393f78ad-f744-43d0-8610-462fd89d566b)

```python
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/b95ed42f-9f8b-4a49-a046-a38725f829f9)

```python
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/fc92262a-9110-4615-a53b-b4e8210fade5)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/98a00600-f8f7-4bae-85ce-3966e3c4656c)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/74c9d8c7-008f-48cf-8449-3fa4021d4a45)

```python
y_pred = rf.predict(X_test)
```

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/4c96a34c-61e0-4092-8ae7-4991090716f5)

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/8fadcfcb-f31c-4e99-92e4-a0a615aab080)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/7d32af8a-2058-4d08-98fa-f19e70766e77)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/0078e112-95e1-4d52-951b-1508c105f394)

```python
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/d68baaf3-6650-4ca9-83b8-6bd899bd8c0c)

```python
y_pred = rf.predict(X_test)
```
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/5df95cbc-ca3c-47ba-86d2-e7cde0a91b2a)

```python
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/c4e67730-b6a4-4081-b19e-712db41adfea)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/431123a0-ea56-4263-b99c-ba4169c97d0f)

```python
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/9b5544eb-05c9-4d96-b3cc-fb6734065b40)


```python
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/46fee88e-24e1-4d9f-abda-c5229b593080)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/21506abb-ca51-4824-a4d5-e40d8b1491e5)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![Screenshot 2025-05-07 204713](https://github.com/user-attachments/assets/49cd84cb-1ff4-4792-9949-3c32dd473380)

```python
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/400ebf1d-eb36-4cc5-a657-be374167a7b8)

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/a5e4b613-1b7b-4fd6-8504-dafbb007f464)




# RESULT:
Thus, Feature selection and Feature scaling has been used on the given dataset.
