import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from datetime import datetime
startTime = datetime.now()

def income_experiment(n_samples:int, A_effect:float, me_prob:float, sl_prob:float, sp_prob:float, clf_type:str):
  """
  This experiment adds biases (measurement error, selection on label,
  selection on predictors) to the Adult income dataset and tests
  counterfactually fair (CF) predictors trained on each to show
  generalization via Theorem 1 and to validate Corollary 2.1.

  Args:
  n_samples: number of sampled distributions to test
  A_effect: the frequency of A affecting X
  me_prob: the probability a sample vulnerable to measurement error is measured incorrectly
  sl_prob: the probability a sample vulnerable to selection on label is selected out of the data
  sp_prob: the probability a sample vulnerable to selection on predictors is selected out of the data
  clf_type: classifier type
  """

  n_samples = 10
  A_effect = 0.8
  me_prob = 0.8
  sl_prob = 0.5
  sp_prob = 0.8
  clf_type = 'xg'

  np.random.seed(1)

  # Load the dataset from a local file
  data = pd.read_csv('adult.csv')

  # Or, uncomment to load the dataset from the UCI Machine Learning Repository
  column_names = ['age','workclass','fnlwgt','education','education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country', 'income'
                  ]
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  data = pd.read_csv(url, names=column_names, sep=r'\s*,\s*', engine='python')

  # Clean up data
  data.loc[data['income'] == '>50k', 'income'] = '>50K'
  data = data.drop(columns=['fnlwgt','education'])

  # Track results
  results1 = pd.DataFrame(columns=['graph','acc_naive','acc_ftu','acc_cf',
                                  'acc_naive_target','acc_ftu_target','acc_cf_target','acc_target_target'])
  results2 = pd.DataFrame(columns=['graph','dp_naive','eo_naive','c_naive',
                                  'dp_ftu','eo_ftu','c_ftu','dp_cf','eo_cf','c_cf'])

  for graph in ['me','sl','sp']:
      """
      Iterate through the graphs of interest.
      me: measurement error => (CF <=> demographic parity)
      sl: selection on label => (CF <=> equalized odds)
      sp: selection on predictors => (CF <=> calibration)
      """

      # Initialize tracking lists
      acc_naive_list, acc_ftu_list, acc_cf_list, acc_naive_target_list, acc_ftu_target_list, acc_cf_target_list, acc_target_target_list, \
      dp_naive_list, eo_naive_list, c_naive_list, dp_ftu_list, eo_ftu_list, c_ftu_list, dp_cf_list, eo_cf_list, c_cf_list = ([] for i in range(16))

      for i in range(n_samples):
          
          print(f'Running sample {i+1} of {n_samples} for graph \'{graph}\'.')

          # Create a copy of the data for manipulation
          d = data.copy()

          # Randomly simulate a new protected class that affects some X variables
          d['A'] = np.random.choice([0, 1], size=len(d))
          d.loc[d['A'] == 1 & (np.random.rand(len(d)) < A_effect), 'race'] = 'Other'

          # Convert to dummies for classification
          d = pd.get_dummies(d, drop_first=True)

          # Create a copy of the data as the unbiased target (i.e., no relationship between A and Y)
          target_data = d.copy()
          X_target = target_data.drop(columns=['income_>50K'])
          y_target = target_data['income_>50K']
          X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(X_target, y_target, test_size=0.2)

          # Generate bias
          if graph == 'me':
              y_marginal = (d['income_>50K'] == 1).mean()
              d.loc[(d['A'] == 1) & (np.random.rand(len(d)) < me_prob), 'income_>50K'] = 0
              d.loc[(d['A'] == 0) & (np.random.rand(len(d)) < (me_prob*y_marginal)/(1-y_marginal)), 'income_>50K'] = 1
          elif graph == 'sl':
              for index, row in d.iterrows():
                  if (row['income_>50K'] == 0) and (row['A'] == 1):
                      if np.random.rand() < sl_prob:
                          d.drop(index, inplace=True)
                  elif (row['income_>50K'] == 1) and (row['A'] == 0):
                      if np.random.rand() < sl_prob:
                          d.drop(index, inplace=True)
          elif graph == 'sp':
              median_age = np.median(data['age'])
              for index, row in d.iterrows():
                  if (row['age'] < median_age) and (row['A'] == 1):
                      if np.random.rand() < sp_prob:
                          d.drop(index, inplace=True)
                  elif (row['age'] > median_age) and (row['A'] == 0):
                      if np.random.rand() < sp_prob:
                          d.drop(index, inplace=True)

          # Create datasets for training and testing
          X = d.drop(columns=['income_>50K'])
          y = d['income_>50K']
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

          # Train FTU classifier
          clf_types = {'lm': LogisticRegression(max_iter=10000),
                      'rf': RandomForestClassifier(),
                      'xg': GradientBoostingClassifier(),
                      'sv': SVC(probability=True),
                      'mlp': MLPClassifier()}
          clf_ftu = clf_types[clf_type]
          clf_ftu.fit(X_train.drop(columns=['A']), y_train)

          # Train naive classifier on biased data
          clf_types = {'lm': LogisticRegression(max_iter=10000),
                      'rf': RandomForestClassifier(),
                      'xg': GradientBoostingClassifier(),
                      'sv': SVC(probability=True),
                      'mlp': MLPClassifier()}
          clf_naive = clf_types[clf_type]
          clf_naive.fit(X_train, y_train)

          # Train classifier on target data
          clf_types = {'lm': LogisticRegression(max_iter=10000),
                      'rf': RandomForestClassifier(),
                      'xg': GradientBoostingClassifier(),
                      'sv': SVC(probability=True),
                      'mlp': MLPClassifier()}
          clf_target = clf_types[clf_type]
          clf_target.fit(X_target_train, y_target_train)

          # Naive classification
          y_pred_naive = clf_naive.predict(X_test)
          y_pred_target_naive = clf_naive.predict(X_target_test)

          # FTU classification
          y_pred_ftu = clf_ftu.predict(X_test.drop(columns=['A']))
          y_pred_target_ftu = clf_ftu.predict(X_target_test.drop(columns=['A']))

          # CF classification
          p = (X_test['A'] == 1).mean()
          y_pred_cf = np.round(p*clf_naive.predict_proba(X_test.assign(A=1))[:,1] + (1-p)*clf_naive.predict_proba(X_test.assign(A=0))[:,1])
          p_target = (X_target_test['A'] == 1).mean()
          y_pred_target_cf = np.round(p_target*clf_naive.predict_proba(X_target_test.assign(A=1))[:,1] + (1-p_target)*clf_naive.predict_proba(X_target_test.assign(A=0))[:,1])

          # Target-trained classification of the target data
          y_pred_target = clf_target.predict(X_test)
          y_pred_target_target = clf_target.predict(X_target_test)

          # Test accuracies
          acc_naive_run = accuracy_score(y_test, y_pred_naive)
          acc_ftu_run = accuracy_score(y_test, y_pred_ftu)
          acc_cf_run = accuracy_score(y_test, y_pred_cf)
          acc_naive_target_run = accuracy_score(y_target_test, y_pred_target_naive)
          acc_ftu_target_run = accuracy_score(y_target_test, y_pred_target_ftu)
          acc_cf_target_run = accuracy_score(y_target_test, y_pred_target_cf)
          acc_target_target_run = accuracy_score(y_target_test, y_pred_target_target)

          # Test DP/EO/C
          dp_naive_run = (y_pred_naive[X_test['A'] == 1] == 1).mean() - (y_pred_naive[X_test['A'] == 0] == 1).mean()
          eo_naive_run = (y_pred_naive[(X_test['A'] == 1) & (y_test == 1)] == 1).mean() - (y_pred_naive[(X_test['A'] == 0) & (y_test == 1)] == 1).mean()
          c_naive_run = (y_test[(X_test['A'] == 1) & (y_pred_naive == 1)]).mean() - (y_test[(X_test['A'] == 0) & (y_pred_naive == 1)]).mean()
          dp_ftu_run = (y_pred_ftu[X_test['A'] == 1] == 1).mean() - (y_pred_ftu[X_test['A'] == 0] == 1).mean()
          eo_ftu_run = (y_pred_ftu[(X_test['A'] == 1) & (y_test == 1)] == 1).mean() - (y_pred_ftu[(X_test['A'] == 0) & (y_test == 1)] == 1).mean()
          c_ftu_run = (y_test[(X_test['A'] == 1) & (y_pred_ftu == 1)]).mean() - (y_test[(X_test['A'] == 0) & (y_pred_ftu == 1)]).mean()
          dp_cf_run = (y_pred_cf[X_test['A'] == 1] == 1).mean() - (y_pred_cf[X_test['A'] == 0] == 1).mean()
          eo_cf_run = (y_pred_cf[(X_test['A'] == 1) & (y_test == 1)] == 1).mean() - (y_pred_cf[(X_test['A'] == 0) & (y_test == 1)] == 1).mean()
          c_cf_run = (y_test[(X_test['A'] == 1) & (y_pred_cf == 1)]).mean() - (y_test[(X_test['A'] == 0) & (y_pred_cf == 1)]).mean()

          # Track accuracies and DP/EO/C
          acc_naive_list.append(acc_naive_run)
          acc_ftu_list.append(acc_ftu_run)
          acc_cf_list.append(acc_cf_run)
          acc_naive_target_list.append(acc_naive_target_run)
          acc_ftu_target_list.append(acc_ftu_target_run)
          acc_cf_target_list.append(acc_cf_target_run)
          acc_target_target_list.append(acc_target_target_run)
          dp_naive_list.append(dp_naive_run)
          eo_naive_list.append(eo_naive_run)
          c_naive_list.append(c_naive_run)
          dp_ftu_list.append(dp_ftu_run)
          eo_ftu_list.append(eo_ftu_run)
          c_ftu_list.append(c_ftu_run)
          dp_cf_list.append(dp_cf_run)
          eo_cf_list.append(eo_cf_run)
          c_cf_list.append(c_cf_run)

      # Calculate mean metrics
      acc_naive = np.nanmean(acc_naive_list)
      acc_ftu = np.nanmean(acc_ftu_list)
      acc_cf = np.nanmean(acc_cf_list)
      acc_naive_target = np.nanmean(acc_naive_target_list)
      acc_ftu_target = np.nanmean(acc_ftu_target_list)
      acc_cf_target = np.nanmean(acc_cf_target_list)
      acc_target_target = np.nanmean(acc_target_target_list)
      dp_naive = np.nanmean(dp_naive_list)
      eo_naive = np.nanmean(eo_naive_list)
      c_naive = np.nanmean(c_naive_list)
      dp_ftu = np.nanmean(dp_ftu_list)
      eo_ftu = np.nanmean(eo_ftu_list)
      c_ftu = np.nanmean(c_ftu_list)
      dp_cf = np.nanmean(dp_cf_list)
      eo_cf = np.nanmean(eo_cf_list)
      c_cf = np.nanmean(c_cf_list)

      results1.loc[len(results1.index)] = [graph,acc_naive,acc_ftu,acc_cf,
                                          acc_naive_target,acc_ftu_target,acc_cf_target,acc_target_target]
      results2.loc[len(results2.index)] = [graph,dp_naive,eo_naive,c_naive,dp_ftu,eo_ftu,c_ftu,dp_cf,eo_cf,c_cf]

  print("""Validations:
  1. The CF classifier is more accurate on the target data than the training data.
  2. The CF classifier approximately achieves DP in the ME graph, EO in the SL graph, and C in the SP graph.\n""")


  print(f'Inputs: n_samples = {n_samples}, A_effect = {A_effect}, me_prob = {me_prob}, sl_prob = {sl_prob}, sp_prob = {sp_prob} , clf_type = {clf_type}')
  print(results1)
  print(results2)

"""
Cell output (runtime ~30min):

graph	acc_naive	acc_ftu	acc_cf	acc_naive_target	acc_ftu_target	acc_cf_target	acc_target_target
0	me	0.828389	0.795609	0.769476	0.803101	0.811377	0.820405	0.867373
1	sl	0.877139	0.868589	0.861015	0.856579	0.865162	0.866344	0.865515
2	sp	0.865867	0.865615	0.865791	0.869906	0.869753	0.869922	0.868018

graph	dp_naive	eo_naive	c_naive	dp_ftu	eo_ftu	c_ftu	dp_cf	eo_cf	c_cf
0	me	-0.295976	-0.497652	-0.227772	-0.116520	-0.168690	-0.676656	-0.000473	0.090555	-0.815756
1	sl	0.289482	0.301609	0.017033	0.188254	0.097015	0.170343	0.132071	-0.002128	0.222536
2	sp	0.143528	0.080142	0.002516	0.143331	0.079882	0.003227	0.142779	0.078918	0.003984
"""


income_experiment(n_samples = 10, A_effect = 0.8, me_prob = 0.8, sl_prob = 0.5, sp_prob = 0.8, clf_type = 'xg')

print(f'Runtime: {datetime.now() - startTime}')