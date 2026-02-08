import pytest
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from condo_rag.ingest import chunk_text
from condo_rag.qa import retrieve_semantic_search, retrieve_and_generate, generate_llm_response
MODEL_ALL_MINILM_L6_V2_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
import os
from dotenv import load_dotenv

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
def indexA_bigchunk(doc1_embeddings, doc2_embeddings):

    # Stack embeddings from both documents into a single matrix
    embs = np.vstack([doc1_embeddings, doc2_embeddings])
    # Ensure float32 dtype for faiss
    embs = embs.astype('float32')
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

@pytest.fixture
def indexA_smallchunk(doc1_dct, doc2_dct):
    """
    Small chunk size.  Will be many chunks per document.
    """
    CHUNKSIZE = 20
    OVERLAP = 0
    texts = []

    model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)

    for d in [doc1_dct, doc2_dct]:
        chunks = chunk_text(d['text'], chunk_size=CHUNKSIZE, overlap=OVERLAP)
        for c in chunks:
            texts.append(c)

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return {'index': index, 'texts': texts}

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
        """
        Just make sure that pytest is running.  That's it
        """
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

    def test_indexA_1(self, indexA_bigchunk):
        """
        Expect indexA_bigchunk to be not None
        """
        assert indexA_bigchunk.d > 0

class Test_retrieve_semantic_search:
    def test_query_gettysburg_top_ranked_1(self, indexA_bigchunk, metadatasA_bigchunk):
        """
        Using indexA_bigchunk and medatatasA, ask a question which is clearly best
        answered with document1. Expect top result from retrieve_semantic_search()
        to be from source = "doc1_text"; chunk = 0, 
        and 2nd result to be from source. = "doc2_text", chunk = 0
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "when did our fathers bring forth a new nation?"
        results = retrieve_semantic_search(query, model, indexA_bigchunk, metadatasA_bigchunk, k_top=5)
        # Assert top result corresponds to doc1 (high score for doc1_embeddings)
        assert results[0]['metadata']['source'] == "doc1_text"
        assert results[1]['metadata']['source'] == "doc2_text"

    def test_query_gettysburg_top_ranked_2(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Using indexA_smallchunk['index'] and medatatasA, ask a question which is clearly best
        answered with document1. Expect top result from retrieve_semantic_search()
        to be from source = "doc1_text"; chunk = 0.
        Expect all top five results to be from doc1_text
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "when did our fathers bring forth a new nation?"
        results = retrieve_semantic_search(query, model, indexA_smallchunk['index'], metadatasA_smallchunk, k_top=5)
        # Assert top result corresponds to doc1 chunk 0 (high score for doc1_embeddings)
        assert results[0]['metadata']['chunk'] == 0

        # Assert top result corresponds to doc1
        assert results[0]['metadata']['source'] == "doc1_text"
        assert results[1]['metadata']['source'] == "doc1_text"
        assert results[2]['metadata']['source'] == "doc1_text"
        assert results[3]['metadata']['source'] == "doc1_text"
        assert results[4]['metadata']['source'] == "doc1_text"

    def test_query_pride_and_prejudice_1(self, indexA_bigchunk, metadatasA_bigchunk):
        """
        Using indexA_bigchunk and metadatasA_bigchunk, ask a question which is clearly best
        answered with the final sentence of document2. Expect top result from retrieve_semantic_search()
        to be from source = "doc2_text"; chunk = 0, 
        and 2nd result to be from source = "doc1_text", chunk = 0
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "What income does this single man collect each year?"
        results = retrieve_semantic_search(query, model, indexA_bigchunk, metadatasA_bigchunk, k_top=5)
        # Assert top result corresponds to doc2 (high score for doc2_embeddings)
        assert results[0]['metadata']['source'] == "doc2_text"
        assert results[1]['metadata']['source'] == "doc1_text"

    def test_query_pride_and_prejudice_2(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Using indexA_smallchunk['index'] and medatatasA, ask a question which is clearly best
        answered with document2. Expect top result from retrieve_semantic_search()
        to be from source = "doc2_text"; chunk = 12 or 13 (the last two chunks of doc2_text, which contain the key sentence about the single
        Expect all top five results to be from doc1_text
        """
        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "What income does this single man collect each year from his furtune?"
        results = retrieve_semantic_search(query, model, indexA_smallchunk['index'], metadatasA_smallchunk, k_top=5)
        # Assert top result corresponds to doc2 chunk 12 (the second-to-last chunk) or
        # chunk 13 (last chunk)
        assert ( 
            (results[0]['metadata']['chunk'] == 12)
            or (results[0]['metadata']['chunk'] == 13)
        )

        # Assert top result corresponds to doc2
        assert results[0]['metadata']['source'] == "doc2_text"
        assert results[1]['metadata']['source'] == "doc2_text"
        assert results[2]['metadata']['source'] == "doc2_text"
        assert results[3]['metadata']['source'] == "doc2_text"
        assert results[4]['metadata']['source'] == "doc2_text"

class Test_generate_llm_response:
    def test_retrieve_then_generate_gettysburg_1(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Separately test the retrieve_semantic_search() function and the generate_llm_response() function
        using the query setup from test_query_gettysburg_top_ranked_1,
        but send it indexA_smallchunk, because that's the fixure that also includes attribute 'texts'.
        Load OpenAI API key via dotenv, query about Gettysburg, and expect the answer to contain
        a phrase like "87 years ago".
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OpenAI API key not found in environment variables"

        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "When did our fathers bring forth a new nation? " + \
                "Quantify your answer in years"

        retrieved_chunks = retrieve_semantic_search(query, model, indexA_smallchunk['index'], metadatasA_smallchunk, k_top=5)                
        llm_response = generate_llm_response(query, indexA_smallchunk['index'], metadatasA_smallchunk, retrieved_chunks, indexA_smallchunk['texts'], api_key)


        # Assert the answer contains the expected phrase (case-insensitive)
        assert ("87" in llm_response.lower()) or ("eighty-seven" in llm_response.lower())

    def test_generate_gettysburg_1(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Separately test generate_llm_response() function
        Use mockup of results expected from the retrieval stageof test_query_gettysburg_top_ranked_1,
        Send indexA_smallchunk to generate_llm_response(), because that's the fixure that also includes attribute 'texts'.
        Load OpenAI API key via dotenv, query about Gettysburg, and expect the answer to contain
        a phrase like "87 years ago".
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OpenAI API key not found in environment variables"

        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "When did our fathers bring forth a new nation? " + \
                "Quantify your answer in years"

        # mock results from retrieval stage
        retrieved_chunks = [
            {'score': 0.999, 'metadata': {'source': 'doc1_text', 'chunk': 0}, 'idx': 0}
        ]
        llm_response = generate_llm_response(query, indexA_smallchunk['index'], metadatasA_smallchunk, retrieved_chunks, indexA_smallchunk['texts'], api_key)

        # Assert the answer contains the expected phrase (case-insensitive)
        assert ("87" in llm_response.lower()) or ("eighty-seven" in llm_response.lower())

    def test_generate_pride_and_prejudice_1(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Separately test generate_llm_response() function
        Use mockup of results expected from the retrieval stage of test_query_pride_and_prejudice_2,
        Send indexA_smallchunk to generate_llm_response(), because that's the fixure that also includes attribute 'texts'.
        Load OpenAI API key via dotenv, query about a fortune, and expect the answer to contain
        a phrase like "4000 or 5000".  Other numbers will be wrong.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OpenAI API key not found in environment variables"

        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "What annual income does this single man collect each year from his fortune?" + \
                "Quantify your answer"

        # mock results from retrieval stage
        # This is the final few sentences of the passage
        retrieved_chunks = [
            {'score': 0.999, 'metadata': {'source': 'doc2_text', 'chunk': 12}, 'idx': 19},
            {'score': 0.999, 'metadata': {'source': 'doc2_text', 'chunk': 13}, 'idx': 20}
        ]
        llm_response = generate_llm_response(query, indexA_smallchunk['index'], metadatasA_smallchunk, retrieved_chunks, indexA_smallchunk['texts'], api_key)

        # Assert the answer contains the expected phrase (case-insensitive)
        assert (
            ("four thousand" in llm_response.lower())
            or ("five thousand" in llm_response.lower())
            or ("4000" in llm_response.lower())
            or ("5000" in llm_response.lower())
            or ("4,000" in llm_response.lower())
            or ("5,000" in llm_response.lower())
        )


class Test_retrieve_and_generate:
    def test_retrieve_and_generate_gettysburg_1(self, indexA_smallchunk, metadatasA_smallchunk):
        """
        Test the function retrieve_and_generate(). 
        using the query setup from test_query_gettysburg_top_ranked_1,
        but send it indexA_smallchunk, because that's the fixure that also includes attribute 'texts'.
        Load OpenAI API key via dotenv, query about Gettysburg, and expect the answer to contain
        a phrase like "87 years ago".
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OpenAI API key not found in environment variables"

        model = SentenceTransformer(MODEL_ALL_MINILM_L6_V2_NAME)
        query = "When did our fathers bring forth a new nation? " + \
                "Quantify your answer in years"

        llm_response = retrieve_and_generate(query, model, indexA_smallchunk['index'], metadatasA_smallchunk, indexA_smallchunk['texts'], api_key, k_top=5)

        # Assert the answer contains the expected phrase (case-insensitive)
        assert ("87" in llm_response.lower()) or ("eighty-seven" in llm_response.lower())