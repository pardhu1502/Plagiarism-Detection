from sklearn.cluster import AgglomerativeClustering


def cluster_students(
        similarity_matrix,
        threshold=0.75):

    distance_matrix = 1 - similarity_matrix

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1-threshold,
        n_clusters=None
    )

    labels = clustering.fit_predict(
        distance_matrix
    )

    return labels