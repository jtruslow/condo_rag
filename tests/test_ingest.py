import pytest
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from condo_rag.ingest import chunk_text
from condo_rag.qa import retrieve_semantic_search, retrieve_and_generate, generate_llm_response
from condo_rag.ingest import read_pdf, read_txt, load_documents

MODEL_ALL_MINILM_L6_V2_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
import os
from dotenv import load_dotenv
import tempfile
from fpdf import FPDF

@pytest.fixture
def doc1_text():
    doc = """Fourscore and seven years ago our fathers brought forth on this continent
a new nation, conceived in liberty and dedicated to the proposition that all men are
created equal. Now we are engaged in a great civil war, testing whether that nation
or any nation so conceived and so dedicated can long endure. We are met on a great
battlefield of that war. We have come to dedicate a portion of that field as a final
resting-place for those who here gave their lives that that nation might live. It is
altogether fitting and proper that we should do this. But in a larger sense, we cannot
dedicate, we cannot consecrate, we cannot hallow this ground. The brave men, living
and dead who struggled here have consecrated it far above our poor power to add or
detract.
"""
    return doc

@pytest.fixture
def doc2_text():
    doc = """It is a truth universally acknowledged, that a single man in possession
of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first
entering a neighbourhood, this truth is so well fixed in the minds of the
surrounding families that he is considered as the rightful property of some
one or other of their daughters.
"My dear Mr. Bennet," said his lady to him one day, "have you heard that
Netherfield Park is let at last?"
Mr. Bennet replied that he had not.
"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."
Mr. Bennet made no answer.
"Do not you want to know who has taken it?" cried his wife impatiently.
"You want to tell me, and I have no objection to hearing it."
This was invitation enough.
"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a
young man of large fortune from the north of England; that he came down on
Monday in a chaise and four to see the place, and was so much delighted with it,
that he agreed with Mr. Morris immediately; that he is to take possession before
Michaelmas, and some of his servants are to be in the house by the end of next week."
"What is his name?"
"Bingley."
"Is he married or single?"
"Oh! single, my dear, to be sure! A single man of large fortune; four or five
thousand a year. What a fine thing for our girls!"
"""
    return doc

@pytest.fixture
def doc1_no_image_pdf(doc1_text):
    """
    Create a temporary PDF file with a single page containing the text from doc1_text.
    The PDF has no images, ensuring it's >90% text for testing read_pdf logic.
    Yields the file path and deletes the file after the test.
    """
    os.makedirs('tests/data/test_ingest', exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.pdf', dir='tests/data/test_ingest', delete=False) as tmp_file:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, doc1_text)
        pdf.output(tmp_file.name)
        yield tmp_file.name
    os.unlink(tmp_file.name)

class Test_read_pdf:
    def test_read_pdf_doc1_no_images(self, doc1_no_image_pdf, doc1_text):
        """
        Test that read_pdf correctly extracts text from a PDF with no images.
        The test creates a temporary PDF file containing the text from doc1_text,
        then calls read_pdf and asserts that the extracted text matches the original text.
        """
        extracted_text = read_pdf(doc1_no_image_pdf)
        assert extracted_text.strip() == doc1_text.strip()

