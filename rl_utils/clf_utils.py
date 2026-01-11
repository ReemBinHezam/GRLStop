# import
import pandas as pd
import numpy as np

#modeling
from sklearn.linear_model import LogisticRegression

#evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#feature
from sklearn.feature_extraction.text import TfidfVectorizer



import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import vstack


def run_classification_model_tfidf(query_id, n_samp_docs, n_docs, labels, features, clf_name, imbalance_handle):

    features = vstack(features) # combine sparse rows into single sparse matrix


    #split train & test sets
    train_x = features[0:n_samp_docs]
    train_y = labels[0:n_samp_docs]
    valid_x = features[n_samp_docs:n_docs]
    valid_y = labels[n_samp_docs:n_docs]


    # calculate relv, non-relv
    relv_cnt = sum(train_y)
    non_relv_cnt = len(train_y) - relv_cnt


    if imbalance_handle == 'cost_sensitive_manual':
      # manually assign majority and minority to either 0 or 1 based on sample
      if relv_cnt >= non_relv_cnt:
        majority_class = 1
        minority_class = 0
        IR = non_relv_cnt/relv_cnt
        class_weight={majority_class:IR, minority_class:1}
      else:
        majority_class = 0
        minority_class = 1
        IR = relv_cnt/non_relv_cnt
        class_weight={majority_class:IR, minority_class:1}

      clf = LogisticRegression(solver=solver, random_state=0, C=1.0, max_iter=10000, class_weight = class_weight)


    accuracy, f1, predictions = train_model_save_threshold(query_id,clf_name, clf, train_x, train_y, valid_x, valid_y)

    predictions = predictions.astype(int)

    return accuracy, f1, predictions



def train_model_save_threshold(topic_id, clf_name, classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
      # fit the training dataset on the classifier
      classifier.fit(feature_vector_train, label)

      #set threshold optimised F1 (models default)
      model_threshold = 0.5

      # get clf labels
      predictions = (classifier.predict_proba(feature_vector_valid)[:,1] >= model_threshold).astype(bool) # set threshold to threshold_list[i]


      acc = metrics.accuracy_score(valid_y, predictions) * 100
      f1 = metrics.f1_score(valid_y, predictions, average='macro') * 100

      return round(acc,3), round(f1,3), predictions
