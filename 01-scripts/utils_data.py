import pickle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr,spearmanr
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score, accuracy_score, f1_score
from sklearn.cluster import SpectralClustering
from glob import glob
from tqdm import tqdm

def save_pickle(filename,data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def hamming_distance(array1, array2):
    return np.sum(array1 != array2)

def min_hamming_distance(target_seq, seq_list):
    min_distance = float('inf')
    for seq in seq_list:
        distance = hamming_distance(target_seq, seq)
        if distance < min_distance:
            min_distance = distance
    return min_distance

def calculate_silhouette(X, max_clusters):
    silhouette_scores = []
    for n_clusters in tqdm(range(2, max_clusters+1)):
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        cluster_labels = spectral.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def select_maximally_different_arrays(arrays, num_selected):
    selected_arrays = []
    selected_indices = []
    first_array_index = np.argmax(np.sum(arrays, axis=1))
    first_array = arrays[first_array_index]
    selected_arrays.append(first_array)
    selected_indices.append(first_array_index)
    for _ in range(num_selected - 1):
        dissimilarities = [np.sum([hamming_distance(array, selected_array) for selected_array in selected_arrays]) for array in arrays]
        selected_array_index = np.argmax(dissimilarities)
        while selected_array_index in selected_indices:
            dissimilarities[selected_array_index] = 0 
            selected_array_index = np.argmax(dissimilarities)
        selected_arrays.append(arrays[selected_array_index])
        selected_indices.append(selected_array_index)
    return selected_arrays, selected_indices

def bin_and_subsample(df, col1, col2, bins1=50, bins2=50, max_per_bin=25, labels=False, random_state=0):
    df = df.copy()
    df['bin_col1'] = pd.cut(df[col1], bins=bins1, labels=labels)
    df['bin_col2'] = pd.cut(df[col2], bins=bins2, labels=labels)
    df['bin_2d'] = list(zip(df['bin_col1'], df['bin_col2']))

    grouped = df.groupby('bin_2d',group_keys=False)
    subsampled = grouped.apply(
        lambda g: g.sample(n=min(len(g), max_per_bin), random_state=random_state)
    )

    return subsampled.reset_index(drop=True)

def calculate_metrics_from_df(test_df):
    pearson_corr_df = test_df.groupby('params').apply(lambda row: pearsonr(row['y_test_hat'],row['y_test'])[0]).reset_index().rename(columns={0:'pearson_corr'})
    spearman_corr_df = test_df.groupby('params').apply(lambda row: spearmanr(row['y_test_hat'],row['y_test'])[0]).reset_index().rename(columns={0:'spearman_corr'})
    corr_df = pearson_corr_df.merge(spearman_corr_df,on='params')
    r2_df = test_df.groupby('params').apply(lambda row: r2_score(row['y_test_hat'],row['y_test'])).reset_index().rename(columns={0:'R2'})
    rmse_df = test_df.groupby('params').apply(lambda row: mean_squared_error(row['y_test_hat'],row['y_test'],squared=False)).reset_index().rename(columns={0:'RMSE'})
    return corr_df.merge(r2_df,on='params').merge(rmse_df,on='params')

def calculate_metrics(y_test,y_test_hat):
    assert len(y_test) == len(y_test_hat)
    corr = pearsonr(y_test,y_test_hat)[0]
    rmse = np.sqrt(mean_squared_error(y_test,y_test_hat))
    r2 = r2_score(y_test,y_test_hat)
    return corr, rmse, r2

def calculate_performance(df, y_test_col, y_test_hat_col):
    assert len(df[y_test_col].to_numpy()) == len(df[y_test_hat_col].to_numpy())
    pearson_corr, _ = pearsonr(df[y_test_col], df[y_test_hat_col])
    correlation_matrix = np.corrcoef(df[y_test_col], df[y_test_hat_col])
    R2 = correlation_matrix[0, 1] ** 2
    rmse = np.sqrt(mean_squared_error(df[y_test_col], df[y_test_hat_col]))
    return pearson_corr, R2, rmse

def parse_results(results_dir):
    train_df = None
    for train_pkl in glob(f'{results_dir}/*_train_df.pkl'):
        tmp_df = pd.read_pickle(train_pkl)
        tmp_df['params'] = train_pkl.split('/')[-1].replace('_train_df.pkl','')
        train_df = pd.concat([train_df,tmp_df])

    test_df = None
    for test_pkl in glob(f'{results_dir}/*_test_df.pkl'):
        tmp_df = pd.read_pickle(test_pkl)
        tmp_df['params'] = test_pkl.split('/')[-1].replace('_test_df.pkl','')
        test_df = pd.concat([test_df,tmp_df])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def harmonize_dataset(initial_data,
                      new_data,
                      overlap,
                      curve_fn,
                      params,
                      harmonized_col_prefix=''):
    initial_dataset = pd.read_csv(initial_data)
    initial_scaler = preprocessing.MinMaxScaler()
    initial_dataset['old_activity_scaled'] = initial_scaler.fit_transform(initial_dataset['old_activity'].to_numpy().reshape(-1,1))
    initial_dataset['dataset'] = 'initial'

    initial_data = initial_data.copy()
    initial_data[['harmonized_activity','harmonized_activity_scaled']] = initial_data[['old_activity','old_activity_scaled']] 
    initial_data.loc[initial_data['AAseq'].isin(overlap['AAseq']),'dataset'] = 'initial_overlap'

    new_df = new_data.copy()
    new_dist_scaled = new_df['new_activity_scaled'].to_numpy().reshape(-1, 1)
    y_pred = curve_fn(new_dist_scaled, *params)
    gaussian_noise = np.random.normal(0, 0.05, size=len(new_df))
    new_df['harmonized_activity_scaled'] = y_pred + gaussian_noise.reshape(-1,1)
    new_df.loc[new_df['AAseq'].isin(overlap['AAseq']),'dataset'] = 'updated_overlap'
    new_df = new_df[['dataset','AAseq','harmonized_activity_scaled','new_activity', 'new_activity_scaled']]

    df = pd.concat([new_df,initial_data]).reset_index(drop=True)
    df['harmonized_activity_scaled'] = df['harmonized_activity_scaled'].apply(lambda x: 0.0 if x < 0.0 else x)
    df['harmonized_activity_scaled'] = df['harmonized_activity_scaled'].apply(lambda x: 1.0 if x > 1.0 else x)
    df['harmonized_activity'] = initial_scaler.inverse_transform(df['harmonized_activity_scaled'].to_numpy().reshape(-1,1)).flatten()
    df = df.rename(columns={'harmonized_activity_scaled':harmonized_col_prefix+'harmonized_activity_scaled','harmonized_activity':harmonized_col_prefix+'harmonized_activity'})
    
    return df

def polynomial(x,a,b,c):
    return a * x**2 + b * x + c

def linear(x,a,b):
    return a * x + b

def sigmoid(x,L,x0,k,b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def classification_metrics(y_test, y_test_hat):
    accuracy = accuracy_score(y_test, y_test_hat)
    f1 = f1_score(y_test,y_test_hat,average='weighted')
    metrics = {"Accuracy": accuracy,"F1 Score": f1}
    return metrics