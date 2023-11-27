import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, OPTICS
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
#from yellowbrick.cluster import SilhouetteVisualizer
#import umap
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import sys

results = {
    "KMeans": {
        "normal": {},
        "umap": {},
        "pca": {},
        "outlier_free": {}
    },
    "DBSCAN": {
        "normal": {},
        "umap": {},
        "pca": {}
    },
    "OPTICS": {
        "normal": {},
        "umap": {},
        "pca": {}
    }
}



def data_preprocessing(df):
    print(df.isna().mean()*100)
    df.loc[(df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].mean()
    df.loc[(df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=df['CREDIT_LIMIT'].mean()
    df = df.drop(columns=['CUST_ID'])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)


def Kmeans(df): 
    k_values = range(1, 11)  
    wcss = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
        if k > 1:  
            silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
        else:
            silhouette_scores.append(0)  

    fig, ax1 = plt.subplots(figsize=(12, 7))


    ax1.set_xlabel('Küme Sayısı (k)')
    ax1.set_ylabel('WCSS', color='tab:blue')
    ax1.plot(k_values, wcss, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Silhouette Skoru', color='tab:orange')
    ax2.plot(k_values, silhouette_scores, 'o-', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Dirsek Yöntemi ve Silhouette Skoru')
    plt.show()
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, color in enumerate(colors):
        ax.scatter(df_scaled[labels == i, 0], df_scaled[labels == i, 1], df_scaled[labels == i, 2], 
                c=color, label=f'Cluster {i+1}', s=50)

    ax.set_title("3D Scatter Plot of Clusters")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()
    plt.show()

    silhouette_vals = silhouette_samples(df_scaled, labels)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    y_lower = 10

    for i in range(4):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / 4)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        
        y_lower = y_upper + 10  

    ax.set_title("Silhouette Plot for the Clusters")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.set_yticks([])  
    ax.axvline(x=silhouette_score(df_scaled, labels), color="red", linestyle="--")  
    plt.show()

    cluster_counts = np.bincount(labels)
    total_count = len(labels)
    percentages = (cluster_counts / total_count) * 100

    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=[f'Cluster {i+1}' for i in range(4)], colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title("Percentage Distribution of Clusters")
    plt.show()   

    kmeans = KMeans(n_clusters=3)  
    labels = kmeans.fit_predict(df_scaled)  


    results["KMeans"]["normal"]["Silhouette Coefficient"] = silhouette_score(df_scaled, labels)
    results["KMeans"]["normal"]["Calinski-Harabasz Index"] = calinski_harabasz_score(df_scaled, labels)
    results["KMeans"]["normal"]["Davies-Bouldin Index"] = davies_bouldin_score(df_scaled, labels)

    for metric, value in results["KMeans"]["normal"].items():
        print(f"{metric}: {value:.2f}") 


def epsilon(X):
    
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    
    distances_1 = distances[:, 1]
    plt.plot(distances_1, color='#5829A7')
    plt.xlabel('Total')
    plt.ylabel('Distance')
        
    for spine in plt.gca().spines.values():
        spine.set_color('None')
        
    plt.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    plt.grid(axis='x', alpha=0)
    
    plt.title('DBSCAN Epsilon Value for Scaled Data')
    plt.tight_layout()
    plt.show()

def dbscan(df):
    dbscan = DBSCAN(eps=2.2, min_samples=5)
    labels = dbscan.fit_predict(df_scaled)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  
            col = [0.6, 0.6, 0.6, 1]
        class_member_mask = (labels == k)
        xy = df_scaled[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')  
    plt.grid(True)
    plt.show()

    dbscan_normal = DBSCAN(eps=2.2, min_samples=5)
    labels_normal = dbscan_normal.fit_predict(df_scaled)

    results["DBSCAN"]["normal"]["Silhouette Coefficient"] = silhouette_score(df_scaled, labels_normal) if len(np.unique(labels_normal)) > 1 else 0
    results["DBSCAN"]["normal"]["Calinski-Harabasz Index"] = calinski_harabasz_score(df_scaled, labels_normal)
    results["DBSCAN"]["normal"]["Davies-Bouldin Index"] = davies_bouldin_score(df_scaled, labels_normal)

    for metric, value in results["DBSCAN"]["normal"].items():
        print(f"{metric}: {value:.2f}")



def optics(df):
    optics = OPTICS(min_samples=5, max_eps=2.2)
    labels = optics.fit_predict(df_scaled)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  
            col = [0.6, 0.6, 0.6, 1]
        class_member_mask = (labels == k)
        xy = df_scaled[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)  

    plt.title('OPTICS Clustering')
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')  
    plt.grid(True)
    plt.show()

    optics_normal = OPTICS(min_samples=9)
    labels_normal = optics_normal.fit_predict(df_scaled)

    results["OPTICS"]["normal"]["Silhouette Coefficient"] = silhouette_score(df_scaled, labels_normal) if len(np.unique(labels_normal)) > 1 else 0
    results["OPTICS"]["normal"]["Calinski-Harabasz Index"] = calinski_harabasz_score(df_scaled, labels_normal)
    results["OPTICS"]["normal"]["Davies-Bouldin Index"] = davies_bouldin_score(df_scaled, labels_normal)

    for metric, value in results["OPTICS"]["normal"].items():
        print(f"{metric}: {value:.2f}")






def run(csv: str = './CC GENERAL.csv'):
        data = pd.read_csv(csv)
        print(data)
        print(data.info())        
        print(data.nunique())
        print(data.describe(include='all'))

        #missed values 
        print(data.isnull().sum())
        # Parcourir le dictionnaire
        for algorithm, algorithm_results in results.items():
            print(f"Algorithm: {algorithm}")
            
            for technique, technique_results in algorithm_results.items():
                print(f"  Technique: {technique}")

                print("    Processing...")
                


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(folder=sys.argv[1])
    else:
        run()        

