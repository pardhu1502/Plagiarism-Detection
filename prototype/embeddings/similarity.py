from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(embeddings):

    return cosine_similarity(
        embeddings
    )


def find_clusters(
        similarity_matrix,
        filenames,
        threshold=0.80):

    visited = set()

    clusters = []

    n = len(filenames)

    for i in range(n):

        if i in visited:
            continue

        cluster = [filenames[i]]

        for j in range(n):

            if i == j:
                continue

            if similarity_matrix[i][j] >= threshold:

                cluster.append(
                    filenames[j]
                )

        if len(cluster) > 1:

            clusters.append(cluster)

            for name in cluster:

                idx = filenames.index(name)

                visited.add(idx)

    return clusters