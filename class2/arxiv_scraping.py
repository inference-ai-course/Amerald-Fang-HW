import asyncio
import nest_asyncio
from playwright.async_api import async_playwright
from PIL import Image
from io import BytesIO
import pytesseract
import arxiv
import json
import time

# Reconfigure Tesseract in case that's needed
pytesseract.pytesseract.tesseract_cmd = r'D:\Workspace_AI\Tesseract\tesseract.exe'

def extract_abstract_from_text(text):
    if not text:
        return ""
    
    text_lower = text.lower()
    if "abstract" in text_lower:
        start = text_lower.find("abstract")
        end_markers = ["introduction", "keywords", "\n\n\n"]
        end = len(text)
        for marker in end_markers:
            idx = text_lower.find(marker, start + 10)
            if idx != -1 and idx < end:
                end = idx
        
        abstract = text[start:end]
        lines = [l.strip() for l in abstract.split('\n') if len(l.strip()) > 15]
        return ' '.join(lines[1:]).strip() if len(lines) > 1 else ' '.join(lines)
    
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 80]
    return ' '.join(paragraphs[:2]) if paragraphs else text[:400]

async def scrape_arxiv_with_playwright_async(query="game engines", max_results=200):
    # get results synchronously first (small blocking step)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(client.results(search))  # fetch list before entering async loop

    papers_data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1200, 'height': 800})

        for i, result in enumerate(results, 1):
            abs_url = result.entry_id
            try:
                await page.goto(abs_url, timeout=10000)
                await page.wait_for_load_state('networkidle')

                playwright_text = await page.evaluate("""() => {
                    const abstract = document.querySelector('blockquote.abstract');
                    return abstract ? abstract.innerText : document.body.innerText;
                }""")

                screenshot_bytes = await page.screenshot(full_page=False)
                image = Image.open(BytesIO(screenshot_bytes))
                ocr_text = pytesseract.image_to_string(image, lang='eng')

                abstract_playwright = extract_abstract_from_text(playwright_text)
                abstract_ocr = extract_abstract_from_text(ocr_text)

                paper_info = {
                    "url": abs_url,
                    "title": result.title,
                    "abstract": result.summary.replace('\n', ' ').strip(),
                    "abstract_playwright": abstract_playwright,
                    "abstract_ocr": abstract_ocr,
                    "authors": [author.name for author in result.authors],
                    "date": result.published.strftime("%Y-%m-%d")
                }
                papers_data.append(paper_info)

                if i % 20 == 0:
                    with open(f'arxiv_checkpoint_{i}.json', 'w', encoding='utf-8') as f:
                        json.dump(papers_data, f, ensure_ascii=False, indent=2)

                await asyncio.sleep(1)  # rate limit

            except Exception as e:
                papers_data.append({
                    "url": abs_url,
                    "title": result.title,
                    "abstract": result.summary.replace('\n', ' ').strip(),
                    "authors": [author.name for author in result.authors],
                    "date": result.published.strftime("%Y-%m-%d"),
                    "error": str(e)
                })

        await browser.close()

    return papers_data


import sys


def _run_scraper_sync(query="game engines", max_results=50):
    """Run the async scraper from a synchronous context.

    Uses asyncio.run when possible. If an event loop is already running
    (e.g. inside a Jupyter notebook), falls back to applying nest_asyncio
    and running the coroutine on the existing loop.
    """
    try:
        return asyncio.run(scrape_arxiv_with_playwright_async(query, max_results=max_results))
    except RuntimeError as e:
        # If we're inside an existing running loop (common in notebooks),
        # apply nest_asyncio and run until complete on the running loop.
        if 'event loop is running' in str(e) or sys.platform.startswith('win') and 'already running' in str(e):
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(scrape_arxiv_with_playwright_async(query, max_results=max_results))
        raise


def main():
    papers = _run_scraper_sync("game engines", max_results=200)

    # Save final results (small run)
    output_file = 'arxiv_clean.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    # Show a sample output object (truncated)
    if papers:
        print(json.dumps(papers[0], indent=2, ensure_ascii=False)[:800] + '...')
    else:
        print("No papers were returned. Check debug artifacts in debug_arxiv/ to see page contents or errors.")


if __name__ == '__main__':
    main()