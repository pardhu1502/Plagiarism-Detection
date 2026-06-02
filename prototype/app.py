import os
import zipfile
import tempfile

import pandas as pd
import streamlit as st

from embeddings.embedding_model import (
    generate_embedding
)

from embeddings.similarity import (
    compute_similarity_matrix,
    find_clusters
)

from ocr.doctr_extractor import (
    extract_text
)


st.set_page_config(
    page_title="Assignment Plagiarism Detector",
    layout="wide"
)

st.title("Assignment Plagiarism Detector")


uploaded_zip = st.file_uploader(
    "Upload ZIP File",
    type=["zip"]
)


def get_pdf_files(folder):

    pdfs = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(".pdf"):

                pdfs.append(
                    os.path.join(root, file)
                )

    return pdfs


if uploaded_zip:

    threshold_percent = st.slider(
        "Similarity Threshold (%)",
        min_value=50,
        max_value=100,
        value=80
    )

    if st.button("Run Analysis"):

        with tempfile.TemporaryDirectory() as temp_dir:

            zip_path = os.path.join(
                temp_dir,
                uploaded_zip.name
            )

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            extract_dir = os.path.join(
                temp_dir,
                "extracted"
            )

            os.makedirs(
                extract_dir,
                exist_ok=True
            )

            with zipfile.ZipFile(
                    zip_path,
                    "r") as zip_ref:

                zip_ref.extractall(
                    extract_dir
                )

            pdf_files = get_pdf_files(
                extract_dir
            )

            if len(pdf_files) == 0:

                st.error(
                    "No PDF files found."
                )

                st.stop()

            st.success(
                f"{len(pdf_files)} PDFs Found"
            )

            filenames = []

            embeddings = []

            progress = st.progress(0)

            status = st.empty()

            for idx, pdf_path in enumerate(pdf_files):

                filename = os.path.basename(
                    pdf_path
                )

                status.write(
                    f"Processing {filename}"
                )

                text = extract_text(
                    pdf_path
                )

                if len(text.strip()) == 0:

                    continue

                embedding = generate_embedding(
                    text
                )

                filenames.append(
                    filename
                )

                embeddings.append(
                    embedding
                )

                progress.progress(
                    (idx + 1)
                    / len(pdf_files)
                )

            if len(embeddings) == 0:

                st.error(
                    "No valid text extracted."
                )

                st.stop()

            similarity_matrix = (
                compute_similarity_matrix(
                    embeddings
                )
            )

            similarity_df = pd.DataFrame(
                similarity_matrix * 100,
                index=filenames,
                columns=filenames
            )

            clusters = find_clusters(
                similarity_matrix,
                filenames,
                threshold_percent / 100
            )

            st.success(
                "Analysis Completed"
            )

            st.header(
                "Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Assignments",
                    len(filenames)
                )

            with col2:

                st.metric(
                    "Clusters",
                    len(clusters)
                )

            with col3:

                max_similarity = (
                    similarity_df.values.max()
                )

                st.metric(
                    "Highest Similarity",
                    f"{max_similarity:.2f}%"
                )

            st.header(
                "Similarity Matrix"
            )

            st.dataframe(
                similarity_df,
                use_container_width=True
            )

            csv = similarity_df.to_csv()

            st.download_button(
                "Download Similarity Matrix",
                csv,
                "similarity_matrix.csv",
                "text/csv"
            )

            st.header(
                "Suspicious Clusters"
            )

            if len(clusters) == 0:

                st.success(
                    "No suspicious clusters found."
                )

            else:

                for idx, cluster in enumerate(
                        clusters):

                    st.subheader(
                        f"Cluster {idx+1}"
                    )

                    cluster_df = pd.DataFrame(
                        {
                            "Student Files":
                            cluster
                        }
                    )

                    st.dataframe(
                        cluster_df,
                        use_container_width=True
                    )