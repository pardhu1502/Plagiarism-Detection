"""
Generate plagiarism report.
"""

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet

from config import REPORT_NAME


class ReportGenerator:

    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate(
        self,
        document_a,
        document_b,
        score,
        heatmaps
    ):

        pdf = SimpleDocTemplate(REPORT_NAME)

        elements = []

        elements.append(
            Paragraph(
                "<b>Plagiarism Detection Report</b>",
                self.styles["Title"]
            )
        )

        elements.append(Spacer(1, 20))

        elements.append(
            Paragraph(
                f"<b>Document A:</b> {document_a}",
                self.styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Document B:</b> {document_b}",
                self.styles["Normal"]
            )
        )

        elements.append(Spacer(1, 10))

        elements.append(
            Paragraph(
                f"<b>Overall Similarity:</b> {score:.2f}%",
                self.styles["Heading2"]
            )
        )

        elements.append(Spacer(1, 20))

        for page_number, image_path in enumerate(heatmaps, start=1):

            elements.append(
                Paragraph(
                    f"Page {page_number}",
                    self.styles["Heading3"]
                )
            )

            elements.append(
                Image(
                    image_path,
                    width=450,
                    height=600
                )
            )

            elements.append(Spacer(1, 20))

        pdf.build(elements)

        print(f"Report saved as {REPORT_NAME}")