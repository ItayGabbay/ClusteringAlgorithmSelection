import pandas as pd
import numpy as np
from scipy.stats import entropy

def get_statlog_metafeatures(dataset_name, df: pd.DataFrame):
    record = {'dataset': dataset_name.split('.')[0]}
    record['MA1'] = np.log2(df.shape[0])
    record['MA2'] = np.log2(df.shape[1])
    
    discrete_cols = []
    continuous_cols = []
    discrete_entropies = []
    discrete_concentration = []
    outlier_limits = dict()
    for col in df.columns:
        if df[col].dtype == 'float64':
            continuous_cols.append(col)
        elif df[col].nunique() <= df.shape[0] * 0.3:
            discrete_cols.append(col)
            value,counts = np.unique(df[col].astype('str'), return_counts=True)
            discrete_entropies.append(entropy(counts))
            discrete_concentration.append(((df[col].value_counts() / df[col].count()) ** 2).sum())
        else:
            continuous_cols.append(col)
     
#      for col in continuous_cols:
#         quarter1 = (df[col].astype('float64').quantile(0.25))
#         quarter3 = df[col].astype('float64').quantile(0.75)
#         iqr = quarter3 - quarter1

    record['MA3'] = len(discrete_cols) / df.shape[1] * 100
    
    record['MA5'] = np.mean(discrete_entropies)
    record['MA6'] = np.mean(discrete_concentration)
    
    record['MA7'] = (np.mean(np.mean(np.abs(df[continuous_cols].corr()[df[continuous_cols].corr()!=1]))))
    record['MA8'] = np.mean(df[continuous_cols].skew(axis=0))
    record['MA9'] = np.mean(df[continuous_cols].kurtosis(axis=0))
    
    return record
    