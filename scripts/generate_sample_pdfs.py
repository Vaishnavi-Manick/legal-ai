import os
import sys
import glob
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "sample_pdfs")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "AILA2019", "Object_casedocs")


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that adds header and footer with page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        # Header
        self.drawString(54, 750, "SUPREME COURT OF INDIA — RECORD OF PROCEEDINGS")
        self.setStrokeColorRGB(0.7, 0.7, 0.7)
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        # Footer
        self.line(54, 50, 558, 50)
        self.drawString(54, 38, "CONFIDENTIAL — FOR LEGAL RESEARCH ONLY")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_text)
        self.restoreState()


def create_native_pdf(text_filepath, output_pdf_path):
    with open(text_filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=14,
        leading=18,
        textColor='#1a1a1a',
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor='#444444',
        spaceAfter=12
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor='#222222',
        spaceAfter=8
    )

    story = []
    if lines:
        story.append(Paragraph(f"<b>{lines[0]}</b>", title_style))
    if len(lines) > 1:
        story.append(Paragraph(f"<i>{lines[1]}</i>", meta_style))

    for line in lines[2:]:
        # Escape XML entities for ReportLab
        clean_text = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(clean_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)


def create_scanned_pdf(text_filepath, output_pdf_path):
    """Generates a scanned PDF by rendering text into images (simulating a scanned document)."""
    with open(text_filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Create bitmap image with text
    img = Image.new('RGB', (612, 792), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lines = content.split('\n')
    y = 50
    draw.text((50, y), "[SCANNED JUDGMENT DOCUMENT - POOR QUALITY]", fill=(100, 100, 100), font=font)
    y += 30
    for line in lines[:35]:
        if y > 730:
            break
        draw.text((50, y), line[:80], fill=(30, 30, 30), font=font)
        y += 18

    # Save image as PDF
    img.save(output_pdf_path, "PDF", resolution=100.0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "C*.txt")))[:50]
    print(f"Generating 50 sample PDFs in '{OUTPUT_DIR}' from AILA corpus...")

    for idx, txt_file in enumerate(txt_files):
        filename = f"judgment_{idx+1:03d}.pdf"
        out_path = os.path.join(OUTPUT_DIR, filename)

        # Make the last 5 PDFs scanned PDFs to test OCR/scanned detection pipeline
        if idx >= 45:
            create_scanned_pdf(txt_file, out_path)
            print(f"  [{idx+1}/50] Generated SCANNED PDF: {filename}")
        else:
            create_native_pdf(txt_file, out_path)
            print(f"  [{idx+1}/50] Generated NATIVE PDF: {filename}")

    print("Sample PDF generation complete (50 PDFs: 45 Native, 5 Scanned).")


if __name__ == "__main__":
    main()

