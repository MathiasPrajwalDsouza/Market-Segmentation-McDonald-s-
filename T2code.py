import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

st.title('Market Segmentation Analysis')


st.header("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(data.head())


    def transform_data(data):
        binary_cols = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
        data[binary_cols] = data[binary_cols].applymap(lambda x: 1 if x == 'Yes' else 0)
        return data[binary_cols]

    st.header("Step 2: Data Transformation")
    transformed_data = transform_data(data)
    st.write("### Transformed Data")
    st.write(transformed_data.head())


    def perform_pca(data):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_


        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(principal_components[:, 0], principal_components[:, 1], c='grey', alpha=0.5)
        ax.set_title('Perceptual Map of McDonald’s Attributes')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True)
        
        st.write("### Perceptual Map")
        st.pyplot(fig)

        st.write(f"Explained Variance by Components: {explained_variance}")
        return principal_components

    st.header("Step 3: PCA Analysis")
    pca_data = perform_pca(transformed_data)


    def perform_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=123)
        data['Segment'] = kmeans.fit_predict(data)


        wcss = []
        for i in range(2, 9):
            kmeans = KMeans(n_clusters=i, random_state=123)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(range(2, 9), wcss, marker='o')
        ax.set_title('Scree Plot of Cluster Numbers')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Within-Cluster Sum of Squares')
        ax.grid(True)

        st.write("### Scree Plot")
        st.pyplot(fig)
        return data


    st.header("Step 4: KMeans Segmentation")
    n_clusters = st.slider('Select the number of clusters (segments)', 2, 8, 4)
    segmented_data = perform_kmeans(transformed_data.copy(), n_clusters)


    def generate_segmented_perceptual_map(pca_data, segment_data):
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=segment_data['Segment'], cmap='rainbow', alpha=0.6)
        legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
        ax.add_artist(legend1)
        ax.set_title('Segmented Perceptual Map')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True)

        st.write("### Segmented Perceptual Map")
        st.pyplot(fig)

    generate_segmented_perceptual_map(pca_data, segmented_data)

    def calculate_slsa(data):
        
        ari_matrix = np.zeros((7, 7))

        
        for i in range(2, 9):
            kmeans_i = KMeans(n_clusters=i, random_state=123).fit(data)
            labels_i = kmeans_i.labels_

            for j in range(2, 9):
                if i != j:
                    kmeans_j = KMeans(n_clusters=j, random_state=123).fit(data)
                    labels_j = kmeans_j.labels_

                    # Calculate ARI between two sets of labels
                    ari_matrix[i-2, j-2] = adjusted_rand_score(labels_i, labels_j)

        return ari_matrix

    def plot_slsa(ari_matrix):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(ari_matrix, annot=True, cmap='coolwarm', xticklabels=range(2, 9), yticklabels=range(2, 9), cbar=True)
        ax.set_title('Segment Level Stability Across Solutions (SLSA)')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Number of Clusters')
        st.write("### SLSA Plot")
        st.pyplot(fig)

    st.header("Step 6: Segment Level Stability Across Solutions (SLSA)")
    ari_matrix = calculate_slsa(transformed_data)
    plot_slsa(ari_matrix)


# For Running in terminal Use Command: streamlit run path of the code of the file where it existed/T2code.py