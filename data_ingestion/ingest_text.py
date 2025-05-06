import pymupdf4llm
import pymupdf
import os
import tabula
import json
import re
import uuid
import hashlib
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "revola-revola"
SOURCE_FILE_PATH = "data/revola_text_data/revolaguide.pdf"
METADATA_JSON_PATH = "data/revola_text_data/rag_guide_metadata.json"

# Embedding and storing in FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = 3072  # for OpenAI text-embedding-3-large

llm = ChatOpenAI(model="gpt-4o-mini")
pc = Pinecone(api_key=PINECONE_API_KEY)


# --- New Function: store_chunks_as_json ---
def store_chunks_as_json(chunks_to_store, output_filepath, append=False):
    """
    Stores or appends chunk data (ID, text, metadata) to a JSON file.

    Args:
        chunks_to_store (list[dict]): List of dictionaries, each representing a chunk
                                     with 'chunk_id', 'text', and 'metadata' keys.
        output_filepath (str): Path to save the output JSON file.
        append (bool): If True, appends to the file if it exists. Otherwise, overwrites.
    """
    existing_data = []
    if append and os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                print(
                    f"Warning: Existing metadata file {output_filepath} is not a list. Overwriting."
                )
                existing_data = []
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Could not read or parse existing metadata file {output_filepath}. Overwriting. Error: {e}"
            )
            existing_data = []

    # Combine existing data with new data
    all_data = existing_data + chunks_to_store

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Write the combined data back to the JSON file
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        action = "Appended" if append and existing_data else "Stored"
        print(
            f"Successfully {action.lower()} {len(chunks_to_store)} chunks metadata. Total chunks in {output_filepath}: {len(all_data)}"
        )
    except IOError as e:
        print(f"Error writing JSON metadata file: {e}")
    except TypeError as e:
        print(f"Error serializing metadata to JSON: {e}")


# --- (calculate_file_hash remains the same) ---
def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(8192):  # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return None


# --- Modified initialize_vector_store ---
def initialize_vector_store(
    source_filepath=SOURCE_FILE_PATH, metadata_output_path=METADATA_JSON_PATH
):
    """
    Processes a PDF, chunks it, stores metadata externally (first pass), and upserts to Pinecone.

    Args:
        source_filepath (str): Path to the source PDF document.
        metadata_output_path (str): Path to save/update the JSON metadata file.

    Returns:
        tuple(PineconeVectorStore, str, str) or None: The vector store object,
                                                     document_id, content_hash, or None if error.
    """
    print(f"Processing document: {source_filepath}")

    # --- Calculate Document Info ---
    document_id = os.path.basename(source_filepath)
    content_hash = calculate_file_hash(source_filepath)
    if not content_hash:
        print("Could not calculate file hash. Aborting PDF processing.")
        return None

    # --- Load and Process PDF ---
    try:
        md_text = pymupdf4llm.to_markdown(
            source_filepath,
            page_chunks=True,
            show_progress=True,
        )
    except Exception as e:
        print(f"Error processing PDF with pymupdf4llm: {e}")
        return None

    documents = []
    for page_data in md_text:
        if len(page_data["text"].strip()) < 40:
            continue
        text = page_data["text"].replace("\n", " ")
        # --- Unified Metadata for PDF Chunks ---
        metadata = {
            # Chunk specific ID will be generated later
            "parent_document_id": document_id,
            "parent_content_hash": content_hash,
            # Embedding model added later
            "source_type": "pdf_chunk",  # <-- Key differentiator
            "original_source_path": page_data["metadata"][
                "file_path"
            ],  # Path to the PDF
            "page_number": page_data["metadata"]["page"],
        }
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    if not documents:
        print("No processable content found in the PDF.")
        # Still return IDs etc. as tables might exist
        return None, document_id, content_hash

    # --- Chunking Text ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    print(f"Split PDF into {len(split_docs)} text chunks.")

    # --- Prepare Chunks for JSON Storage ---
    pdf_chunks_to_store = []
    for doc in split_docs:
        chunk_id = str(uuid.uuid4())
        # Update metadata with final fields
        doc.metadata["chunk_id"] = chunk_id
        doc.metadata["embedding_model"] = embeddings.model
        pdf_chunks_to_store.append(
            {"chunk_id": chunk_id, "text": doc.page_content, "metadata": doc.metadata}
        )

    # --- Store PDF Chunks Metadata Externally (Overwrite Mode for initial pass) ---
    store_chunks_as_json(pdf_chunks_to_store, metadata_output_path, append=False)

    # --- Prepare for Upsert ---
    chunk_texts = [item["text"] for item in pdf_chunks_to_store]
    chunk_metadatas = [item["metadata"] for item in pdf_chunks_to_store]
    chunk_ids = [item["chunk_id"] for item in pdf_chunks_to_store]

    # --- Upsert to Pinecone ---
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return None  # Critical error
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    print(f"Upserting {len(chunk_texts)} PDF text chunks to Pinecone...")
    vector_db = None  # Initialize vector_db
    try:
        vector_db = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace="text"
        )
        vector_db.add_texts(
            texts=chunk_texts, metadatas=chunk_metadatas, ids=chunk_ids, batch_size=100
        )
        print("PDF Text chunk upsert complete.")
    except Exception as e:
        print(f"Error upserting PDF text chunks to Pinecone: {e}")
        # Don't return None here, maybe tables can still be processed
        # But vector_db object might be None or in an uncertain state

    # Return vector_db object along with doc ID and hash for table processing
    return vector_db, document_id, content_hash


# --- (process_tables remains mostly the same, ensure it saves files) ---
def process_tables(page_num, base_dir, filepath):
    # (Make sure pandas is imported in the main block)
    table_output_dir = os.path.join(base_dir, "tables")
    os.makedirs(table_output_dir, exist_ok=True)
    processed_files = []  # Keep track of files created on this page
    try:
        tables = tabula.read_pdf(
            filepath,
            pages=page_num + 1,
            multiple_tables=True,
            lattice=True,
            stream=False,
            java_options=["-Djava.awt.headless=true"],
        )
        clean_tables = []
        for table in tables:
            tbl = table.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if tbl.empty:
                continue
            clean_tables.append(tbl)

        if not clean_tables:
            return []  # Return empty list if no tables found

        for table_idx, table in enumerate(clean_tables):
            table_text = "\n".join(
                [
                    " | ".join(map(lambda x: str(x) if not pd.isna(x) else "", row))
                    for row in table.values
                ]
            )
            # Use a clear naming convention including page number
            table_file_name = f"{table_output_dir}/{os.path.basename(filepath)}_page{page_num+1}_table{table_idx}.txt"
            try:
                with open(table_file_name, "w", encoding="utf-8") as f:
                    f.write(table_text)
                processed_files.append(
                    table_file_name
                )  # Add path of successfully saved file
            except IOError as e:
                print(f"Error writing table file {table_file_name}: {e}")
        return processed_files  # Return list of created file paths
    except Exception as e:
        print(f"Error extracting tables from page {page_num+1} of {filepath}: {str(e)}")
        return []  # Return empty list on error


# --- Modified upsert_summarized_tables ---
def upsert_summarized_tables(
    table_files_dir,
    metadata_output_path,  # Path to the central JSON
    parent_document_id,  # Info from the source PDF
    parent_content_hash,  # Info from the source PDF
    vector_db,  # Existing PineconeVectorStore object
):
    """
    Reads raw table text files, summarizes them (skipping content duplicates),
    stores unified metadata externally (append), and upserts summaries to Pinecone.

    Args:
        table_files_dir (str): Directory containing the raw table .txt files.
        metadata_output_path (str): Path to the central JSON metadata file.
        parent_document_id (str): ID of the source PDF these tables came from.
        parent_content_hash (str): Hash of the source PDF.
        vector_db (PineconeVectorStore): The vector store instance to upsert into.
    """
    summarized_chunks_to_store = []
    # --- Set to track hashes of raw table content processed in this run ---
    processed_table_content_hashes = set()
    # --------------------------------------------------------------------
    print(f"Processing summarized tables from directory: {table_files_dir}")

    if not os.path.isdir(table_files_dir):
        print(f"Table directory not found: {table_files_dir}")
        return
    if not vector_db:
        print("Vector store object is not valid. Skipping summarized table upsert.")
        return
    if not parent_document_id or not parent_content_hash:
        print("Missing parent document ID or hash. Skipping summarized table upsert.")
        return

    # Regex to extract page number (adjust if filename format changes)
    page_num_regex = re.compile(r"_page(\d+)_")

    # Iterate through relevant text files in the directory
    try:
        # Ensure we only process tables related to the current parent document
        table_filenames = [
            f
            for f in os.listdir(table_files_dir)
            if f.endswith(".txt")
            and parent_document_id in f  # Match parent doc ID in filename
        ]
        print(
            f"Found {len(table_filenames)} raw table files potentially related to {parent_document_id}."
        )
    except FileNotFoundError:
        print(
            f"Error listing files in directory: {table_files_dir}. Skipping table processing."
        )
        return

    processed_count = 0
    skipped_duplicates = 0

    for filename in table_filenames:
        file_path = os.path.join(table_files_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_table_text = file.read()

            if not raw_table_text.strip():
                print(f"Skipping empty table file: {filename}")
                continue

            # --- Check for Duplicate Content ---
            # Normalize whitespace slightly before hashing to catch minor variations
            normalized_content = re.sub(r"\s+", " ", raw_table_text).strip()
            table_content_hash = hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest()

            if table_content_hash in processed_table_content_hashes:
                print(
                    f"Skipping file {filename} as its content is a duplicate of a previously processed table in this run."
                )
                skipped_duplicates += 1
                continue  # Skip to the next file

            # If unique, add hash to set and proceed
            processed_table_content_hashes.add(table_content_hash)
            # -----------------------------------

            # --- LLM Summarization ---
            print(f"Summarizing unique table content from: {filename}")
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert sales representative for the company Revola AI. Your job is to summarize tables from Revola's documentation and give the table a title.
                IMPORTANT: Make sure you reference Revola somewhere in the summarization.
                IMPORTANT: Make sure all the information in the table is included in the summary. Do not omit details.
                Table to be summarized:\n---\n{text}\n---\nSummary:""",
            )
            # Use the original raw text for summarization, not the normalized one
            message = HumanMessage(content=prompt.format(text=raw_table_text))
            summarized_text = llm.invoke([message]).content.strip()
            # --- End LLM Summarization ---

            # --- Create Unified Metadata for Table Summary ---
            chunk_id = str(uuid.uuid4())
            page_number = None
            match = page_num_regex.search(filename)
            if match:
                try:
                    page_number = int(match.group(1))
                except ValueError:
                    pass  # Keep None if conversion fails

            metadata = {
                "chunk_id": chunk_id,
                "parent_document_id": parent_document_id,
                "parent_content_hash": parent_content_hash,
                "embedding_model": embeddings.model,
                "source_type": "summarized_table",
                "original_source_path": file_path,
                "page_number": page_number,
                "images": [],
            }

            summarized_chunks_to_store.append(
                {"chunk_id": chunk_id, "text": summarized_text, "metadata": metadata}
            )
            processed_count += 1

        except FileNotFoundError:
            print(f"File not found during processing: {file_path}")
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
        except Exception as e:
            # Catch potential LLM errors or other issues
            print(f"An unexpected error occurred processing file {filename}: {e}")

    print(
        f"Finished processing table files. Processed {processed_count} unique tables, skipped {skipped_duplicates} duplicates."
    )

    # --- Store Summarized Table Metadata Externally (Append Mode) ---
    if summarized_chunks_to_store:
        print(
            f"Storing metadata for {len(summarized_chunks_to_store)} summarized tables."
        )
        # Assuming store_chunks_as_json is defined as before
        store_chunks_as_json(
            summarized_chunks_to_store, metadata_output_path, append=True
        )

        # --- Prepare for Upsert ---
        chunk_texts = [item["text"] for item in summarized_chunks_to_store]
        chunk_metadatas = [item["metadata"] for item in summarized_chunks_to_store]
        chunk_ids = [item["chunk_id"] for item in summarized_chunks_to_store]

        # --- Upsert Summarized Tables to Pinecone ---
        print(
            f"Upserting {len(chunk_texts)} unique summarized table chunks to Pinecone..."
        )
        try:
            vector_db.add_texts(
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
                batch_size=100,
            )
            print("Summarized table chunk upsert complete.")
        except Exception as e:
            print(f"Error upserting summarized table chunks to Pinecone: {e}")
    else:
        print("No unique summarized tables found to store or upsert.")


# --- Main Execution Flow ---
if __name__ == "__main__":
    vector_store_instance = None
    doc_id = None
    doc_hash = None

    # --- 1. Initialize vector store from PDF (creates index, stores PDF chunk metadata, upserts PDF chunks) ---
    print("--- Initializing Vector Store from PDF ---")
    initialization_result = initialize_vector_store(
        SOURCE_FILE_PATH, METADATA_JSON_PATH
    )

    if initialization_result:
        vector_store_instance, doc_id, doc_hash = initialization_result
        print(
            f"Initialization completed. Document ID: {doc_id}, Hash: {doc_hash[:8]}..."
        )
    else:
        print(
            "PDF processing or initial upsert failed. Further steps might be affected."
        )
        # Decide if you want to exit or try table processing anyway
        # For robustness, let's try to get doc_id and hash even if PDF processing failed
        doc_id = os.path.basename(SOURCE_FILE_PATH)
        doc_hash = calculate_file_hash(SOURCE_FILE_PATH)

    # --- 2. Process Tables (Extracts tables to text files) ---
    print("\n--- Processing Tables ---")
    table_files_created = []
    if pd and doc_id and doc_hash:  # Only proceed if pandas loaded and we have doc info
        base_dir = "data/revola_text_data"
        filepath = os.path.join(base_dir, doc_id)  # Use doc_id as filename
        try:
            doc = pymupdf.open(filepath)
            print(f"Extracting tables from {len(doc)} pages of {doc_id}...")
            for page_num in range(len(doc)):
                print(f"Processing page {page_num+1}/{len(doc)}")
                page_table_files = process_tables(page_num, base_dir, filepath)
                table_files_created.extend(page_table_files)
            doc.close()
            print(
                f"Finished extracting tables. Found and saved {len(table_files_created)} raw table files."
            )
        except FileNotFoundError:
            print(f"Error: PDF file not found at {filepath} for table extraction.")
        except Exception as e:
            print(f"An error occurred during table extraction: {e}")
    elif not pd:
        print("Skipping table processing because pandas library is not installed.")
    else:
        print("Skipping table processing due to missing document ID or hash.")

    # --- 3. Upsert Summarized Tables (Reads table text files, summarizes, stores metadata, upserts summaries) ---
    print("\n--- Upserting Summarized Tables ---")
    if table_files_created and vector_store_instance and doc_id and doc_hash:
        tables_directory = os.path.join("data/revola_text_data", "tables")
        upsert_summarized_tables(
            tables_directory,
            METADATA_JSON_PATH,
            doc_id,
            doc_hash,
            vector_store_instance,  # Pass the existing vector store object
        )
    elif not table_files_created:
        print("Skipping summarized table upsert as no raw table files were processed.")
    elif not vector_store_instance:
        print(
            "Skipping summarized table upsert as the vector store instance is not available."
        )
    else:
        print("Skipping summarized table upsert due to missing document ID or hash.")
