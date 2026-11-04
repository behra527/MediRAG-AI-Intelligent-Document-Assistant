import fitz
from app.services.pdf_extractor import extract_pdf
from app.services.text_cleaner import clean_pages
from app.services.compare_parsers import compare


def sample_pdf():
    doc = fitz.open()
    for n in range(3):
        page = doc.new_page()
        page.insert_text((45, 30), 'Example Clinical Guideline')
        page.insert_text((45, 95), 'hypo-')
        page.insert_text((45, 108), 'tension warning')
        page.insert_text((45, 780), f'Page {n + 1} of 3')
    data = doc.tobytes()
    doc.close()
    return data


def test_header_and_hyphen():
    result = clean_pages(['Guideline\nhypo-\ntension\nPage 1 of 3',
                          'Guideline\nhypo-\ntension\nPage 2 of 3',
                          'Guideline\nhypo-\ntension\nPage 3 of 3'])
    assert result.removed_lines >= 6
    assert 'hypotension' in result.text


def test_parsers_smoke():
    data = sample_pdf()
    for parser in ('pymupdf', 'pdfplumber'):
        result = extract_pdf(data, parser)
        assert len(result.pages) == 3
        assert 'warning' in result.pages[0].text
    assert 'metrics' in compare(data)
