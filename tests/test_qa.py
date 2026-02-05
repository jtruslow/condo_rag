import pytest
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from condo_rag.ingest import chunk_text
from condo_rag.qa import retrieve_semantic_search
MODEL_ALL_MINILM_L6_V2_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

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
def doc1_dct(doc1_text):
    dct = {'source': "doc1_text", 'text': doc1_text}
    return dct

@pytest.fixture
def doc2_dct(doc2_text):
    dct = {'source': "doc2_text", 'text': doc2_text}
    return dct

@pytest.fixture
def doc1_embeddings(doc1_dct):
    model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)

    text_lst = []
    metadata_lst = []

    chunks = chunk_text(doc1_dct['text'])
    for i, c in enumerate(chunks):
        text_lst.append(c)
        metadata_lst.append({"source": doc1_dct.get('source'), "chunk": i})

    embedding_lst = model.encode(text_lst, convert_to_numpy=True, normalize_embeddings=True)
    return embedding_lst

@pytest.fixture
def doc2_embeddings(doc2_dct):
    model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)

    text_lst = []
    metadata_lst = []

    chunks = chunk_text(doc2_dct['text'])
    for i, c in enumerate(chunks):
        text_lst.append(c)
        metadata_lst.append({"source": doc2_dct.get('source'), "chunk": i})

    embedding_lst = model.encode(text_lst, convert_to_numpy=True, normalize_embeddings=True)
    return embedding_lst

@pytest.fixture
def indexA(doc1_embeddings, doc2_embeddings):

    # Stack embeddings from both documents into a single matrix
    embs = np.vstack([doc1_embeddings, doc2_embeddings])
    # Ensure float32 dtype for faiss
    embs = embs.astype('float32')
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

@pytest.fixture
def metadatasA_bigchunk(doc1_dct, doc2_dct):
    """
    Large chunk size.  Basically will be one chunk per document.
    """
    CHUNKSIZE = 1000
    OVERLAP = 200
    metadata_lst = []
    chunks1 = chunk_text(doc1_dct['text'], chunk_size = CHUNKSIZE, overlap=OVERLAP)
    for i, c in enumerate(chunks1):
        metadata_lst.append({"source": doc1_dct.get('source'), "chunk": i})
    chunks2 = chunk_text(doc2_dct['text'], chunk_size = CHUNKSIZE, overlap=OVERLAP)
    for i, c in enumerate(chunks2):
        metadata_lst.append({"source": doc2_dct.get('source'), "chunk": i})
    return metadata_lst

@pytest.fixture
def metadatasA_smallchunk(doc1_dct, doc2_dct):
    """
    Small chunk size.  Will be many chunks per document.
    """
    CHUNKSIZE = 20
    OVERLAP = 0
    metadata_lst = []
    chunks1 = chunk_text(doc1_dct['text'], chunk_size = CHUNKSIZE, overlap=OVERLAP)
    for i, c in enumerate(chunks1):
        metadata_lst.append({"source": doc1_dct.get('source'), "chunk": i})
    chunks2 = chunk_text(doc2_dct['text'], chunk_size = CHUNKSIZE, overlap=OVERLAP)
    for i, c in enumerate(chunks2):
        metadata_lst.append({"source": doc2_dct.get('source'), "chunk": i})
    return metadata_lst

class Test_trivial:
    def test_trivial_1(self):
        assert True

class Test_fixtures:
    def test_doc1_embeddings_1(self, doc1_embeddings):
        """
        Expect length should be > 0
        """
        assert len(doc1_embeddings) > 0

    def test_doc2_embeddings_1(self, doc2_embeddings):
        """
        Expect length should be > 0
        """
        assert len(doc2_embeddings) > 0

    def test_indexA_1(self, indexA):
        """
        Expect indexA to be not None
        """
        assert indexA.d > 0

class Test_retrieve_semantic_search:
    def test_1(self, indexA):
        assert indexA is not None

    def test_query_gettysburg_top_ranked_1(self, indexA, metadatasA_bigchunk):
        """
        Using indexA and medatatasA, ask a question which is clearly best
        answered with document1. Expect top result from retrieve_semantic_search()
        to be from source = "doc1_text"; chunk = 0, 
        and 2nd result to be from source. = "doc2_text", chunk = 0
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "when did our fathers bring forth a new nation?"
        results = retrieve_semantic_search(query, model, indexA, metadatasA_bigchunk, k_top=5)
        # Assert top result corresponds to doc1 (high score for doc1_embeddings)
        assert results[0]['metadata']['source'] == "doc1_text"
        assert results[1]['metadata']['source'] == "doc2_text"

    def test_query_gettysburg_top_ranked_2(self, indexA, metadatasA_smallchunk):
        """
        Using indexA and medatatasA, ask a question which is clearly best
        answered with document1. Expect top result from retrieve_semantic_search()
        to be from source = "doc1_text"; chunk = 0, 
        
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "when did our fathers bring forth a new nation?"
        results = retrieve_semantic_search(query, model, indexA, metadatasA_smallchunk, k_top=5)
        # Assert top result corresponds to doc1 (high score for doc1_embeddings)
        assert results[0]['metadata']['source'] == "doc1_text"
