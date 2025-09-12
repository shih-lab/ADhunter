import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr,spearmanr
from utils_data import hamming_distance

def plot_train_val(df, ax, color1, color2):
    g = sns.lineplot(data=df,x='epoch',y='train_pearsonr',label='Training',ax=ax,color=color1,linewidth=3,alpha=1)
    sns.lineplot(data=df,x='epoch',y='val_pearsonr',label='Validation',ax=ax,color=color2,linewidth=3,alpha=1)
    g.legend(loc='lower right',frameon=False)
    g.set_ylim(0.3,1.0)
    g.set(ylabel=f'Pearson r',xlabel='Epoch')
    
def plot_test(df, ax, color1):
    y_test_hat, y_test = np.array(df['y_test_hat']), np.array(df['y_test'])
    ax.scatter(y_test_hat,y_test,alpha=0.4,s=5,linewidths=0,edgecolor=None,color=color1)
    pearson_corr = pearsonr(y_test_hat, y_test)[0]
    sns.kdeplot(data=df,x='y_test_hat',y='y_test',thresh=0.2,levels=5,ax=ax,color='black',linewidths=0.5)
    ax.set(xlabel='Predicted activity', ylabel='Empirical activity')
    return pearson_corr

def plot_hamming_distances(arrays,labels):
    num_arrays = len(arrays)
    distances = np.zeros((num_arrays, num_arrays))
    for i in range(num_arrays):
        for j in range(num_arrays):
            distances[i, j] = hamming_distance(arrays[i], arrays[j])
    mask = np.triu(np.ones_like(distances, dtype=bool))
    colors = ["white", '#bb4927']
    cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=256)

    if num_arrays > 20:
        g = sns.heatmap(distances,cbar_kws={'label':'Hamming Distance','shrink':0.6}, cmap='gray_r',mask=mask)
    else:
        g = sns.heatmap(distances,cbar_kws={'label': 'Hamming Distance','shrink':0.6}, annot_kws={"fontsize":8}, cmap=cmap,annot=True, fmt=".0f", mask=mask)
    g.set(title='Hamming distances between models on test data',xlabel='Model',ylabel='Model')
    g.set_xticklabels(labels,rotation=90)
    g.set_yticklabels(labels,rotation=0)

    return g   