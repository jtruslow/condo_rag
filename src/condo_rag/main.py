import argparse
from ingest import load_documents, build_index, save_index, load_index
from qa import retrieve_and_generate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

def cli():
    """
    CLI entrypoint for Condo RAG QA.

    Usage examples:
      python main.py ingest --paths docs/a.pdf docs/b.txt --out indexdir
      python main.py ingest --pathfile docs/paths.txt --out indexdir
      python main.py ask "What is condo policy?" --index indexdir --api-key sk-...
    """
    p = argparse.ArgumentParser(description='Condo RAG QA')
    sub = p.add_subparsers(dest='cmd')
    ingest_p = sub.add_parser('ingest')
    
    group = ingest_p.add_mutually_exclusive_group(required=True)
    group.add_argument('--paths', nargs='+', help='Files (PDF or text) to ingest')
    group.add_argument('--pathfile', help='Path to file that contains list of PDF and TXT paths to ingest')
    ingest_p.add_argument('--out', default='data/final/indexdir', help='Output directory for index')
    ingest_p.add_argument('--chunksize', default=256, help='Tokens per chunk (default: 256)')
    ingest_p.add_argument('--overlap', default=64, help='Tokens overlap between chunks (default: 64)')

    ask = sub.add_parser('ask')
    ask.add_argument('query', help='Question to ask')  # Query to run against the ingested index
    ask.add_argument('--index', default='data/final/indexdir', help='Index directory')  # Directory that stores the FAISS index
    ask.add_argument('--api-key', default=None, help='OpenAI API key (optional)')  # Optional API key for generation

    args = p.parse_args()

    if args.cmd == 'ingest':
        
        if args.pathfile:
            docs = load_documents(args.pathfile)
        else:
            docs = load_documents(args.paths)
        
        index, metadatas, embeddings, texts = build_index(docs)
        save_index(index, metadatas, texts, args.out)
        print('Index and texts saved to', args.out)

    elif args.cmd == 'ask':

        if args.api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = args.api_key

        index, metadatas, texts = load_index(args.index)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        llm_response = retrieve_and_generate(args.query, model, index, metadatas, texts, openai_api_key=api_key)
        print()
        print(llm_response)
        print()
    else:
        p.print_help()

if __name__ == '__main__':
    cli()
