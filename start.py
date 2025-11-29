import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('Placement.csv')


# Query: top 5 Sci&Tech students placed based on salary
print("\nTop 5 placed Sci&Tech students by salary:")
print(
    data[(data['degree_t'] == "Sci&Tech") & (data['status'] == "Placed")]
    .sort_values(by="salary", ascending=False)
    .head()
)

print("\nData info BEFORE preprocessing:")

# -------------------------
# 8. Data Preprocessing
# -------------------------

# Remove unused columns
data = data.drop(['sl_no', 'salary'], axis=1)

# Encode categorical columns
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['ssc_b'] = data['ssc_b'].map({'Central': 1, 'Others': 0})
data['hsc_b'] = data['hsc_b'].map({'Central': 1, 'Others': 0})
data['hsc_s'] = data['hsc_s'].map({'Science': 2, 'Commerce': 1, 'Arts': 0})
data['degree_t'] = data['degree_t'].map({'Sci&Tech': 2, 'Comm&Mgmt': 1, 'Others': 0})
data['specialisation'] = data['specialisation'].map({'Mkt&HR': 1, 'Mkt&Fin': 0})
data['workex'] = data['workex'].map({'Yes': 1, 'No': 0})
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Split features & target
X = data.drop('status', axis=1)
y = data['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -------------------------
# Model Training
# -------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

lr = LogisticRegression()
svm_model = svm.SVC()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

lr.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------

y_pred1 = lr.predict(X_test)
y_pred2 = svm_model.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = gb.predict(X_test)

# -------------------------
# Accuracy Scores
# -------------------------

from sklearn.metrics import accuracy_score

score1 = accuracy_score(y_test, y_pred1)
score2 = accuracy_score(y_test, y_pred2)
score3 = accuracy_score(y_test, y_pred3)
score4 = accuracy_score(y_test, y_pred4)
score5 = accuracy_score(y_test, y_pred5)
score6 = accuracy_score(y_test, y_pred6)

print("\n==================== ACCURACY SCORES ====================")
print("Logistic Regression:", score1)
print("SVM:", score2)
print("KNN:", score3)
print("Decision Tree:", score4)
print("Random Forest:", score5)
print("Gradient Boosting:", score6)
print("=========================================================")

final_data = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'],
            'ACC':[score1*100,
                  score2*100,
                  score3*100,
                  score4*100,
                  score5*100,score6*100]})
print(final_data)


new_data = pd.DataFrame({
    'gender':0,
    'ssc_p':67.0,
    'ssc_b':0,
    'hsc_p':91.0,
    'hsc_b':0,
    'hsc_s':1,
    'degree_p':58.0,
    'degree_t':2,
    'workex':0,
    'etest_p':55.0,
     'specialisation':1,
    'mba_p':58.8,
},index=[0])

import joblib
joblib.dump(lr,'model_campus_placement')
['model_campus_placement']
model = joblib.load('model_campus_placement')
print(model.predict(new_data))