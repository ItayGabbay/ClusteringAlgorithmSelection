import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import numpy as np
pd.options.mode.chained_assignment = None

def get_baseline(baseline, rankings: pd.DataFrame):
    SRC = []
    MRR = []
    raw_ranks = rankings.drop('dataset', axis=1)
    for index in range(len(raw_ranks)):
        SRC.append(np.abs(spearmanr(baseline, raw_ranks.iloc[index])[0]))

        if len(np.where(raw_ranks.iloc[index]==baseline[0])[0]) == 0:
                mrr = 1/(len(raw_ranks.iloc[index]) / 2)
        else:
                mrr = 1/(np.where(raw_ranks.iloc[index]==baseline[0])[0][0] + 1)

        MRR.append(mrr)
    
    return np.array([np.mean(SRC), np.std(SRC)]), np.array([np.mean(MRR), np.std(MRR)])

def evaluate(meta_features: pd.DataFrame, rankings: pd.DataFrame, k=6):
    cv = KFold(n_splits=10)
    X = meta_features
    y = meta_features['dataset']
    final_results = []
    final_MRR = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_test['original_dataset'] = X_test['dataset'].apply(lambda x: x.split('_')[0])
        X_train['original_dataset'] = X_train['dataset'].apply(lambda x: x.split('_')[0])

        # Removing from train the duplicates of test
        idx = X_train.loc[X_train['original_dataset'].isin(X_test['original_dataset'])].index
        X_train = X_train.drop(idx, axis=0)
        y_train = y_train.drop(idx, axis=0)

        model = NearestNeighbors(n_neighbors=k)
        model.fit(X_train.drop(['dataset', 'original_dataset'], axis=1), y_train)
        prediction = model.kneighbors(X_test.drop(['dataset', 'original_dataset'], axis=1), return_distance=True)
        # print(prediction[0])
        predicted = []
        actual = []
        for index, sample in enumerate(prediction[1]):
            # print(X.iloc[sample]['dataset'])
            # print(rankings[rankings['dataset'].isin(X.iloc[sample]['dataset'])])
            predicted.append(rankings[rankings['dataset'].isin(X.iloc[sample]['dataset'])].drop(['dataset'], axis=1).sum().rank(method='average').values)
            pred_actual = rankings[rankings['dataset'] == X_test['dataset'].values[index]].drop(['dataset'], axis=1).values

            if pred_actual.shape[0] == 0:
                print(index, X_test.iloc[index]['dataset'])
            actual.append(pred_actual)
        
        predicted = np.array(predicted)
        actual = np.array(actual)
        actual = actual.reshape(predicted.shape)
        MRR = []
        SRCs = []
        for index in range(len(predicted)):
            if len(np.where(predicted[index]==actual[index][0])[0]) == 0:
                mrr = 1/(len(predicted[index]) / 2)
            else:
                mrr = 1/(np.where(predicted[index]==actual[index][0])[0][0] + 1)

            # if np.abs(spearmanr(predicted[index], actual[index])[0]) < 0.1:
            #     print(X_test.iloc[index]['dataset'], np.abs(spearmanr(predicted[index], actual[index])[0]))
            SRCs.append(np.abs(spearmanr(predicted[index], actual[index])[0]))
            MRR.append(mrr)
        final_results.append([np.mean(SRCs), np.std(SRCs)])
        final_MRR.append([np.mean(MRR), np.std(MRR)])
        # print()
    return np.mean(np.array(final_results), axis=0), np.mean(np.array(final_MRR), axis=0)