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
# Importing required libraries for clustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#model
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def cluster_analyzer(df,n):
    Scaler = StandardScaler()
    df_scaled = Scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    

    kmeans = KMeans(n_clusters=n, random_state=0)
    scaled_labesl = kmeans.fit_predict(df_scaled)
    df["Clusters"] = scaled_labesl
    df["Clusters"] = df["Clusters"].astype('category')

    plt.figure(figsize=(20,35))
    for col in df.columns:
        grid = sns.FacetGrid(df, col='Clusters')
        grid.map(plt.hist, col)
        plt.show()

def data_viz(df):
        cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
        """""
        # Boxplot for each numerical variable
        df.boxplot(figsize=(12, 6))
        plt.show()

        # Countplot for categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            sns.countplot(x=col, data=df)
            plt.show()
                
            
        for col in cols: 
             
            p = sns.histplot(df[col], color="green", kde=True, bins=50, alpha=1, fill=True, edgecolor="black")
            p.axes.lines[0].set_color("#101B15")
            p.axes.set_title("\n Age Distribution\n", fontsize=25)
            plt.ylabel("Count", fontsize=20)
            plt.xlabel(col, fontsize=20)
            sns.despine(left=True, bottom=True)
            plt.show()
        """

        fig = plt.figure(figsize = (15,15))
        sns.heatmap(df.corr(), cmap = 'Blues', square = True, annot = True, linewidths = 0.5)    


def Kmeans(df, data): 
    k_values = range(1, 11)  
    wcss = []
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        if k > 1:  
            silhouette_scores.append(silhouette_score(df, kmeans.labels_))
        else:
            silhouette_scores.append(0)  

    fig, ax1 = plt.subplots(figsize=(12, 7))


    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('WCSS', color='tab:blue')
    ax1.plot(k_values, wcss, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Silhouette Score', color='tab:orange')
    ax2.plot(k_values, silhouette_scores, 'o-', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score')
    plt.show()

    if True : 
            kmeans = KMeans(n_clusters=2,  random_state=23)
            kmeans.fit(df)
            y_means = kmeans.fit_predict(df)
            labels = kmeans.labels_

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            #colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Adjust colors for 3 clusters
            colors = ['#1f77b4', '#ff7f0e']  # Adjust colors for 3 clusters

            # Plotting points for each cluster
            for i, color in enumerate(colors):
                ax.scatter(df[labels == i]['PC1'], df[labels == i]['PC2'], 
                        c=color, label=f'Cluster {i+1}', s=50)
                

            ax.set_title("3D Scatter Plot of Clusters")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.legend()
            plt.show()
            
            print(df)

            from yellowbrick.cluster import InterclusterDistance
            visualizer4 = InterclusterDistance(kmeans)
            visualizer4.fit(df)
            visualizer4.show()
            plt.show()

            centroid = kmeans.cluster_centers_

            y_means = pd.DataFrame(index = y_means)
            #y_means = y_means.rename(index = {0:'Cluster 1',1:'Cluster 2',2:'Cluster 3'})
            y_means = y_means.rename(index = {0:'Cluster 1',1:'Cluster 2'})
            y_means.reset_index(level = 0,inplace = True)
            y_means = y_means.rename(columns = {'index':'Labels'})

            centroid = kmeans.cluster_centers_

            # Plotting using seaborn and matplotlib
            sns.scatterplot(data=df, x='PC1', y='PC2', hue=labels, palette='viridis')
            plt.scatter(x=centroid[:, 0], y=centroid[:, 1], c='red', s=250, marker='*')
            plt.show()

            silhouette_vals = silhouette_samples(df, labels)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            y_lower = 10

            for i in range(3):
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
            ax.axvline(x=silhouette_score(df, labels), color="red", linestyle="--")  
            plt.show()

            cluster_counts = np.bincount(labels)
            total_count = len(labels)
            percentages = (cluster_counts / total_count) * 100

            plt.figure(figsize=(8, 8))
            plt.pie(percentages, labels=[f'Cluster {i+1}' for i in range(2)], colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=140)
            plt.title("Percentage Distribution of Clusters")
            plt.show()   

    kmeans = KMeans(n_clusters=2 , random_state=23)  
    labels = kmeans.fit_predict(df)  


    print('silhouette_score',silhouette_score(df, labels))
    print('calinski_harabasz_score',calinski_harabasz_score(df, labels))
    print('davies_bouldin_score',davies_bouldin_score(df, labels))
   

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
    labels = dbscan.fit_predict(df)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  
            col = [0.6, 0.6, 0.6, 1]
        class_member_mask = (labels == k)
        xy = df[class_member_mask]
        plt.plot(xy['PC1'], xy['PC2'], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)


    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')  
    plt.grid(True)
    plt.show()

    dbscan_normal = DBSCAN(eps=2.2, min_samples=5)
    labels = dbscan_normal.fit_predict(df)

    print('silhouette_score',silhouette_score(df, labels))
    print('calinski_harabasz_score',calinski_harabasz_score(df, labels))
    print('davies_bouldin_score',davies_bouldin_score(df, labels))


def optics(df):

    
        optics = OPTICS(min_samples=5, max_eps=0.2)
        labels = optics.fit_predict(df)

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:  
                col = [0.6, 0.6, 0.6, 1]
            class_member_mask = (labels == k)
            xy = df[class_member_mask]
            plt.plot(xy['PC1'], xy['PC2'], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6,  label=f'Cluster {k}')

        plt.title('OPTICS Clustering')
        plt.xlabel('Feature 1')  
        plt.ylabel('Feature 2')  
        plt.grid(True)
        plt.show()

        optics_normal = OPTICS(min_samples=9)
        labels = optics_normal.fit_predict(df)

        print('silhouette_score',silhouette_score(df, labels))
        print('calinski_harabasz_score',calinski_harabasz_score(df, labels))
        print('davies_bouldin_score',davies_bouldin_score(df, labels))






def run(csv: str = './CC GENERAL.csv'):

  
        df = pd.read_csv(csv)
         
             
        print(df)
        print(df.info())        
        print(df.nunique())
        print(df.describe(include='all'))
        #data_viz(df)
        print(df.isna().mean()*100)
        df.drop(['CUST_ID'], axis=1, inplace=True)
        df.dropna(subset=['CREDIT_LIMIT'], inplace=True)
        df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
        
        
        cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
        for col in cols:
            df[col] = np.log(1 + df[col])


        sns.heatmap(df.corr(), cmap = 'Blues', square = True, annot = True, linewidths = 0.5)  
        plt.show()

        data_viz(df)    
        data = df



        if False:
            from sklearn.decomposition import PCA
        
            pca = PCA()
            #X_red = pca.fit_transform(df) 
            pca = PCA(n_components=2)
            pca_fit = pca.fit_transform(df)
            X_red = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
            from sklearn.cluster import KMeans
    
        if False:
            Kmeans(X_red ,data)
            cluster_analyzer(data,3)
            #optics(X_red)
            #dbscan(X_red)


        

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(folder=sys.argv[1])
    else:
        run()        

