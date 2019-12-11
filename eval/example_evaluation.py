import scipy.io as scio
mat = scio.loadmat(file_name='connectome_features_100.mat')
connectome_features = mat['connectome_features'] #matrix with a connectome in each row
task_index = mat['task_index'] #name of the task being considered
task_label = mat['task_label'] #label corresponding to the task of each connectome

def eval_metric(features,task_labels):
    from sklearn.metrics import silhouette_score
    silhouette_metric = silhouette_score(X=features,labels=task_labels.ravel(),metric='euclidean')
    return silhouette_metric

def plot_tsne(features,task_labels):
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2).fit_transform(features)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 

    targets = np.unique(task_labels)
    for target in targets:
        indicesToKeep = task_labels == target
        indicesToKeep = np.squeeze(indicesToKeep.T)
        ax.scatter(X_embedded[indicesToKeep.T,0]
                   , X_embedded[indicesToKeep.T,1]
                   , c = np.random.rand(3,))
    ax.grid()
