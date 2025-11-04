"""
Batch processing script to download Arxiv papers, convert their PDFs to images,
and use Tesseract OCR to extract the full text.

Reads from an input JSON, processes each paper, and saves to an output JSON.

Required dependencies:
- Python libraries: pip install requests pdf2image pytesseract pillow tqdm
- System dependencies:
    - Tesseract-OCR (https://github.com/tesseract-ocr/tesseract)
    - Poppler (https://poppler.freedesktop.org/)
"""

import json
import requests
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from tqdm import tqdm
import os

# --- CONFIGURATION ---

# The JSON file containing the list of paper objects
# Assumes each object has a "url" key like "https://arxiv.org/abs/..."
INPUT_JSON = 'arxiv_clean.json'

# The new JSON file where results will be saved
OUTPUT_JSON = 'arxiv_papers_with_ocr.json'

# FOR WINDOWS USERS:
# If you did not install Poppler via choco or add it to your system PATH,
# you must provide the path to the poppler \bin directory here.
# e.g., r"C:\Users\YourUser\Downloads\poppler-23.11.0\bin"
POPPLER_PATH = r"D:\Workspace_AI\Poppler\poppler-25.07.0\Library\bin"

# FOR WINDOWS USERS:
# If you downloaded Tesseract manually, provide the full path to tesseract.exe
# e.g., r"D:\Workspace_AI\Tesseract\tesseract.exe"
TESSERACT_CMD = r"D:\Workspace_AI\Tesseract\tesseract.exe"

# --- END CONFIGURATION ---


def process_paper(paper_entry: dict, poppler_path: str = None) -> str:
    """
    Downloads a single Arxiv paper, OCRs it, and returns the full text.
    
    Args:
        paper_entry: A dictionary object for a single paper.
                     Must contain a "url" key.
        poppler_path: Optional path to the Poppler \bin directory.

    Returns:
        The extracted text (str) from the entire PDF.

    Raises:
        Exception: If any step (download, convert, OCR) fails.
    """
    
    # 1. Get PDF URL from abstract URL
    abs_url = paper_entry.get("url")
    if not abs_url or "/abs/" not in abs_url:
        raise ValueError(f"Invalid or missing Arxiv abstract URL: {abs_url}")
        
    # Transform '.../abs/2401.12345' into '.../pdf/2401.12345.pdf'
    pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

    # 2. Download PDF
    headers = {'User-Agent': 'Mozilla/5.0'} # Be a good citizen
    response = requests.get(pdf_url, headers=headers, timeout=20)
    response.raise_for_status()  # Will raise an HTTPError for bad responses
    
    pdf_bytes = response.content

    # 3. Convert PDF to list of images
    # This is the most memory-intensive step
    images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
    
    if not images:
        raise Exception("PDF was empty or Poppler failed to convert.")

    # 4. OCR Each Image and combine text
    full_text_parts = []
    
    # Process page by page to manage memory
    for i, image in enumerate(images):
        # Use Tesseract to get text from the single image (page)
        page_text = pytesseract.image_to_string(image, lang='eng')
        full_text_parts.append(page_text)
        
    return "\n\n--- PAGE BREAK ---\n\n".join(full_text_parts)


def main():
    """
    Main orchestration function.
    Loads papers, loops through them, processes, and saves.
    """
    print(f"Loading papers from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {INPUT_JSON}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {INPUT_JSON}")
        return

    if not isinstance(papers, list):
        print(f"ERROR: Expected input JSON to be a LIST of paper objects.")
        return

    print(f"Found {len(papers)} papers to process.")
    
    processed_papers = []

    # Use tqdm for a nice progress bar
    for paper in tqdm(papers, desc="Processing Arxiv PDFs"):
        try:
            # Process the paper
            ocr_text = process_paper(paper, poppler_path=POPPLER_PATH)
            paper['full_text_ocr'] = ocr_text
            paper['ocr_error'] = None

        except Exception as e:
            # If anything fails, log the error and continue
            print(f"\n[!] Failed to process {paper.get('url')}: {str(e)}")
            paper['full_text_ocr'] = None
            paper['ocr_error'] = str(e)
            
        processed_papers.append(paper)

    # 5. Save all results to a new file
    print(f"\nSaving {len(processed_papers)} processed entries to {OUTPUT_JSON}...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"ERROR: Could not write to output file {OUTPUT_JSON}: {e}")

    print("Batch processing complete.")


if __name__ == "__main__":
    # Check for Tesseract
    if TESSERACT_CMD:
        print(f"Using Tesseract executable at: {TESSERACT_CMD}")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("="*50)
        print("CRITICAL ERROR: Tesseract-OCR not found.")
        print("Please install it from: https://github.com/tesseract-ocr/tesseract")
        print("On macOS: brew install tesseract")
        print("On Linux: sudo apt install tesseract-ocr")
        print("="*50)
        exit(1) # Exit if Tesseract isn't found

    main()

