import argparse
from pathlib import Path

import PyPDF2


def read_pdf(file_path: Path) -> str:
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"An error occurred: {e}"


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=repo_root / "documents" / "project_proposal.pdf",
        help="Path to PDF (default: documents/project_proposal.pdf under repo root)",
    )
    args = parser.parse_args()
    pdf_path = args.pdf.resolve()
    print(read_pdf(pdf_path))


if __name__ == "__main__":
    main()
