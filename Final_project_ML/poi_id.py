#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import pandas as pd
import tester
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale, Imputer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
var_financeiro = ['salary','bonus','long_term_incentive','deferred_income',
                 'deferral_payments','loan_advances','other',
                 'expenses','director_fees','total_payments',
                 'exercised_stock_options','restricted_stock',
                 'restricted_stock_deferred','total_stock_value']

var_email =  ['to_messages','from_messages','from_poi_to_this_person',
              'from_this_person_to_poi','shared_receipt_with_poi']

features_list = ['poi'] + var_financeiro + var_email

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    my_dataset = pickle.load(data_file)
df = pd.DataFrame.from_dict(my_dataset, orient='index')
df = df.replace('NaN',np.nan)
### Task 2: Remove outliers
df = df.drop(['TOTAL'],axis = 0)
df  = df.drop(['FREVERT MARK A','LAVORATO JOHN J',
               'WHALLEY LAWRENCE G','BHATNAGAR SANJAY'],axis =0)
### Task 3: Create new feature(s)
new_features =[]
df['p_bonus'] = df['bonus']/df['total_payments']
df['p_salary'] = df['salary']/df['total_payments']
new_features.append('p_bonus')
new_features.append('p_salary')
df['p_to_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['p_shared_poi'] = df['shared_receipt_with_poi'] / df['to_messages']
new_features.append('p_to_poi')
new_features.append('p_shared_poi')
df = df.replace('NaN',np.nan)
df = df.fillna(0.0)

### Store to my_dataset for easy export below.
my_dataset = df.T.to_dict()

### Extract features and labels from dataset for local testing
total_features = []
total_features += features_list + new_features

data = featureFormat(my_dataset,total_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import svm
svm = Pipeline([('scaler',StandardScaler()),('selector',SelectKBest()),('svm',svm.SVC())])
param_grid = ([{'svm__C': [1,50,100,1000],
                'svm__gamma': [0.5, 0.1],
                'svm__degree':[1,2],
                'svm__kernel': ['rbf','poly'],
                'selector__k':range(1,len(total_features))}])


svm_clf = GridSearchCV(svm, param_grid, scoring='recall', cv = 5).fit(features, labels).best_estimator_
tester.test_classifier(svm_clf, my_dataset, total_features)

best_features = []
for i, feature in enumerate(svm_clf.get_params()['selector'].scores_):
        best_features.append([total_features[1:][i],feature])
pd_feature= pd.DataFrame(best_features,index = np.arange(1,len(best_features)+1),columns  = ['Feature','Score'])

best_features = ['poi'] + pd_feature.nlargest(13,'Score')['Feature'].tolist()
pd_feature.nlargest(13,'Score')
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, best_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn import svm
svm = Pipeline([('scaler',StandardScaler()), ('svm',svm.SVC())])

param_grid = ([{'svm__C': [1000],
                'svm__gamma': [0.1],
                'svm__degree':[2],
                'svm__kernel': ['poly']}])

clf = GridSearchCV(svm, param_grid, scoring='recall',cv = 5).fit(features, labels).best_estimator_


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
tester.test_classifier(clf, my_dataset,best_features)
# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
