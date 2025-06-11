import pymupdf4llm
import pymupdf
import os
import tabula  # tabula-py not normal tabula
import json
import re
import uuid
import hashlib
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # OpenAI is not used directly
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv(override=True)

parser = argparse.ArgumentParser(
    description="Parser for PDF directory ingestion to Pinecone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("company", help="Company to handle (used for index and paths)")
parser.add_argument(
    "source_directory",
    help="Directory containing PDF files to process for the company",
)
args = parser.parse_args()
config = vars(args)
company = config["company"].lower()
SOURCE_PDF_DIRECTORY = config["source_directory"]  # Directory of PDFs

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = f"revola-{company}"
# Output paths will be relative to the company's data directory
COMPANY_DATA_BASE_PATH = f"data/{company}"
METADATA_JSON_PATH = f"{COMPANY_DATA_BASE_PATH}/{company}_rag_text_metadata.json"
TABLE_BASE_DIR = f"{COMPANY_DATA_BASE_PATH}/text_data"  # Base for tables extracted

# Embedding and storing
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = 3072  # for OpenAI text-embedding-3-large

llm = ChatOpenAI(model="gpt-4o-mini")
pc = Pinecone(api_key=PINECONE_API_KEY)


def store_chunks_as_json(chunks_to_store, output_filepath):
    """
    Stores chunk data (ID, text, metadata) to a JSON file, overwriting if it exists.
    This will be called once at the end with all chunks from all documents.

    Args:
        chunks_to_store (list[dict]): List of dictionaries, each representing a chunk.
        output_filepath (str): Path to save the output JSON file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(chunks_to_store, f, indent=4, ensure_ascii=False)
        print(
            f"Successfully stored metadata for {len(chunks_to_store)} chunks in {output_filepath}"
        )
    except IOError as e:
        print(f"Error writing JSON metadata file: {e}")
    except TypeError as e:
        print(f"Error serializing metadata to JSON: {e}")


def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def initialize_pdf_processing(source_filepath, vector_db_instance):
    """
    Processes a single PDF, upserts its chunks to Pinecone, and returns chunk data for JSON storage.

    Args:
        source_filepath (str): Path to the source PDF document.
        vector_db_instance (PineconeVectorStore): Initialized Pinecone vector store.

    Returns:
        tuple(list, str, str) or tuple(None, None, None):
            - List of chunk data (dict) for JSON storage.
            - document_id (basename of the PDF).
            - content_hash of the PDF.
            Returns (None, None, None) if a critical error occurs.
    """
    print(f"\n--- Processing PDF: {source_filepath} ---")

    document_id = os.path.basename(source_filepath)
    content_hash = calculate_file_hash(source_filepath)
    if not content_hash:
        print(f"Could not calculate file hash for {source_filepath}. Skipping.")
        return None, None, None

    try:
        md_text = pymupdf4llm.to_markdown(
            source_filepath,
            page_chunks=True,
            show_progress=False,  # Set to True for individual PDF progress
        )
    except Exception as e:
        print(f"Error processing PDF {source_filepath} with pymupdf4llm: {e}")
        return (
            None,
            document_id,
            content_hash,
        )  # Return IDs for potential table processing

    documents = []
    for page_data in md_text:
        if len(page_data["text"].strip()) < 40:
            continue
        text = page_data["text"].replace("\n", " ")
        metadata = {
            "parent_document_id": document_id,
            "parent_content_hash": content_hash,
            "source_type": "pdf_chunk",
            "original_source_path": page_data["metadata"]["file_path"],
            "page_number": page_data["metadata"]["page"],
        }
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    if not documents:
        print(f"No processable text content found in PDF: {source_filepath}")
        # Still return doc_id and hash as tables might exist independently
        return [], document_id, content_hash

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    print(f"Split PDF {document_id} into {len(split_docs)} text chunks.")

    pdf_chunks_for_json = []
    chunk_texts_for_pinecone = []
    chunk_metadatas_for_pinecone = []
    chunk_ids_for_pinecone = []

    for doc in split_docs:
        chunk_id = str(uuid.uuid4())
        doc.metadata["chunk_id"] = chunk_id
        doc.metadata["embedding_model"] = embeddings.model

        pdf_chunks_for_json.append(
            {"chunk_id": chunk_id, "text": doc.page_content, "metadata": doc.metadata}
        )
        chunk_texts_for_pinecone.append(doc.page_content)
        chunk_metadatas_for_pinecone.append(doc.metadata)
        chunk_ids_for_pinecone.append(chunk_id)

    if chunk_texts_for_pinecone:
        print(
            f"Upserting {len(chunk_texts_for_pinecone)} PDF text chunks for {document_id} to Pinecone..."
        )
        try:
            vector_db_instance.add_texts(
                texts=chunk_texts_for_pinecone,
                metadatas=chunk_metadatas_for_pinecone,
                ids=chunk_ids_for_pinecone,
                batch_size=100,
            )
            print(f"PDF Text chunk upsert complete for {document_id}.")
        except Exception as e:
            print(f"Error upserting PDF text chunks for {document_id} to Pinecone: {e}")
            # Continue, but chunks won't be in Pinecone for this PDF

    return pdf_chunks_for_json, document_id, content_hash


def process_tables_from_pdf(pdf_filepath, company_specific_table_base_dir):
    """
    Extracts tables from all pages of a given PDF and saves them as text files.

    Args:
        pdf_filepath (str): Path to the PDF file.
        company_specific_table_base_dir (str): Base directory to save table files
                                               (e.g., data/company_x/text_data).

    Returns:
        list[str]: List of file paths where tables were saved.
    """
    table_output_dir = os.path.join(company_specific_table_base_dir, "tables")
    os.makedirs(table_output_dir, exist_ok=True)
    all_processed_table_files = []
    pdf_filename_base = os.path.basename(pdf_filepath)

    try:
        doc = pymupdf.open(pdf_filepath)
        print(f"Extracting tables from {len(doc)} pages of {pdf_filename_base}...")
        for page_num in range(len(doc)):
            # print(f"Processing tables on page {page_num + 1}/{len(doc)} of {pdf_filename_base}")
            try:
                tables_on_page = tabula.read_pdf(
                    pdf_filepath,
                    pages=page_num + 1,
                    multiple_tables=True,
                    lattice=True,
                    stream=False,
                    java_options=["-Djava.awt.headless=true"],
                    # silent=True # May help reduce console noise for tabula
                )
                clean_tables = []
                for table in tables_on_page:
                    tbl = table.dropna(how="all", axis=0).dropna(how="all", axis=1)
                    if not tbl.empty:
                        clean_tables.append(tbl)

                if not clean_tables:
                    continue

                for table_idx, table_df in enumerate(clean_tables):
                    table_text = "\n".join(
                        [
                            " | ".join(
                                map(lambda x: str(x) if not pd.isna(x) else "", row)
                            )
                            for row in table_df.values
                        ]
                    )
                    # Unique filename incorporating PDF name, page, and table index
                    table_file_name = f"{table_output_dir}/{pdf_filename_base}_page{page_num+1}_table{table_idx}.txt"
                    try:
                        with open(table_file_name, "w", encoding="utf-8") as f:
                            f.write(table_text)
                        all_processed_table_files.append(table_file_name)
                    except IOError as e:
                        print(f"Error writing table file {table_file_name}: {e}")
            except Exception as e:  # Catch tabula errors for specific page
                if "is zero" in str(e).lower() or "No tables found" in str(
                    e
                ):  # Common tabula messages for no tables
                    pass  # print(f"No tables found by tabula on page {page_num+1} of {pdf_filename_base}.")
                else:
                    print(
                        f"Error extracting tables from page {page_num+1} of {pdf_filename_base}: {str(e)}"
                    )
        doc.close()
        print(
            f"Finished extracting tables for {pdf_filename_base}. Found and saved {len(all_processed_table_files)} raw table files."
        )
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_filepath} for table extraction.")
    except Exception as e:
        print(f"An error occurred during table extraction for {pdf_filepath}: {e}")

    return all_processed_table_files


def upsert_summarized_tables_for_doc(
    table_files_dir,  # Specific to company: e.g. data/company_x/text_data/tables
    parent_document_id,
    parent_content_hash,
    vector_db_instance,
    company_name,  # for the LLM prompt
):
    """
    Reads raw table text files FOR A SPECIFIC DOCUMENT, summarizes them,
    and upserts summaries to Pinecone. Returns chunk data for JSON.

    Args:
        table_files_dir (str): Directory containing raw table .txt files.
        parent_document_id (str): ID of the source PDF these tables came from.
        parent_content_hash (str): Hash of the source PDF.
        vector_db_instance (PineconeVectorStore): The vector store instance.
        company_name (str): Name of the company for prompt.

    Returns:
        list[dict]: List of summarized table chunk data for JSON storage.
    """
    summarized_chunks_for_json = []
    processed_table_content_hashes = set()
    print(
        f"Processing summarized tables from {table_files_dir} for document {parent_document_id}"
    )

    if not os.path.isdir(table_files_dir):
        print(
            f"Table directory not found: {table_files_dir}. Skipping for {parent_document_id}."
        )
        return []
    if not vector_db_instance:
        print(
            f"Vector store object not valid. Skipping summarized table upsert for {parent_document_id}."
        )
        return []

    page_num_regex = re.compile(r"_page(\d+)_")

    # Filter for table files belonging to the current parent_document_id
    try:
        relevant_table_filenames = [
            f
            for f in os.listdir(table_files_dir)
            if f.endswith(".txt")
            and f.startswith(parent_document_id)  # Ensures files are for this PDF
        ]
        print(
            f"Found {len(relevant_table_filenames)} raw table files related to {parent_document_id}."
        )
    except FileNotFoundError:
        print(f"Error listing files in {table_files_dir} for {parent_document_id}.")
        return []

    chunk_texts_for_pinecone = []
    chunk_metadatas_for_pinecone = []
    chunk_ids_for_pinecone = []
    processed_count = 0
    skipped_duplicates = 0

    for filename in relevant_table_filenames:
        file_path = os.path.join(table_files_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_table_text = file.read()

            if not raw_table_text.strip():
                # print(f"Skipping empty table file: {filename}") # Can be noisy
                continue

            normalized_content = re.sub(r"\s+", " ", raw_table_text).strip()
            table_content_hash = hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest()

            if table_content_hash in processed_table_content_hashes:
                # print(f"Skipping file {filename} as its content is a duplicate.") # Can be noisy
                skipped_duplicates += 1
                continue
            processed_table_content_hashes.add(table_content_hash)

            # print(f"Summarizing unique table content from: {filename}") # Can be noisy
            prompt = PromptTemplate(
                input_variables=["text", "company"],
                template="""You are an expert sales representative for the company {company}. Your job is to summarize tables from {company}'s documentation and give the table a title.
                IMPORTANT: Make sure you reference {company} somewhere in the summarization.
                IMPORTANT: Make sure all the information in the table is included in the summary. Do not omit details.
                Table to be summarized:\n---\n{text}\n---\nSummary:""",
            )
            message = HumanMessage(
                content=prompt.format(text=raw_table_text, company=company_name)
            )
            summarized_text = llm.invoke([message]).content.strip()

            chunk_id = str(uuid.uuid4())
            page_number = None
            match = page_num_regex.search(filename)
            if match:
                try:
                    page_number = int(match.group(1))
                except ValueError:
                    pass

            metadata = {
                "chunk_id": chunk_id,
                "parent_document_id": parent_document_id,
                "parent_content_hash": parent_content_hash,
                "embedding_model": embeddings.model,
                "source_type": "summarized_table",
                "original_source_path": file_path,  # Path to the .txt table file
                "page_number": page_number,
                "images": [],  # Placeholder if you add image relations later
            }

            summarized_chunks_for_json.append(
                {"chunk_id": chunk_id, "text": summarized_text, "metadata": metadata}
            )
            chunk_texts_for_pinecone.append(summarized_text)
            chunk_metadatas_for_pinecone.append(metadata)
            chunk_ids_for_pinecone.append(chunk_id)
            processed_count += 1

        except FileNotFoundError:
            print(f"File not found during table summary processing: {file_path}")
        except IOError as e:
            print(f"Error reading table file {file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing table file {filename}: {e}")

    print(
        f"For {parent_document_id}: Processed {processed_count} unique tables, skipped {skipped_duplicates} duplicates."
    )

    if chunk_texts_for_pinecone:
        print(
            f"Upserting {len(chunk_texts_for_pinecone)} unique summarized table chunks for {parent_document_id} to Pinecone..."
        )
        try:
            vector_db_instance.add_texts(
                texts=chunk_texts_for_pinecone,
                metadatas=chunk_metadatas_for_pinecone,
                ids=chunk_ids_for_pinecone,
                batch_size=100,
            )
            print(f"Summarized table chunk upsert complete for {parent_document_id}.")
        except Exception as e:
            print(
                f"Error upserting summarized table chunks for {parent_document_id} to Pinecone: {e}"
            )
    else:
        print(f"No unique summarized tables found to upsert for {parent_document_id}.")

    return summarized_chunks_for_json


# --- Main Execution Flow ---
if __name__ == "__main__":
    if not os.path.isdir(SOURCE_PDF_DIRECTORY):
        print(f"Error: Source directory '{SOURCE_PDF_DIRECTORY}' not found.")
        exit(1)

    pdf_files_to_process = [
        f for f in os.listdir(SOURCE_PDF_DIRECTORY) if f.lower().endswith(".pdf")
    ]

    if not pdf_files_to_process:
        print(f"No PDF files found in '{SOURCE_PDF_DIRECTORY}'.")
        exit(0)

    print(
        f"Found {len(pdf_files_to_process)} PDF(s) to process in '{SOURCE_PDF_DIRECTORY}'."
    )

    # --- 1. Initialize Pinecone (once for all PDFs) ---
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            exit(1)  # Critical error
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    try:
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace="text"
        )
        print("Successfully connected to Pinecone vector store.")
    except Exception as e:
        print(f"Error initializing PineconeVectorStore: {e}")
        exit(1)

    all_chunks_for_json_storage = []  # Accumulate all chunks here

    # --- 2. Process each PDF file ---
    for pdf_filename in pdf_files_to_process:
        current_pdf_filepath = os.path.join(SOURCE_PDF_DIRECTORY, pdf_filename)
        print(f"\n==================================================================")
        print(f"Starting processing for: {current_pdf_filepath}")
        print(f"==================================================================")

        # --- 2a. Process PDF text chunks ---
        pdf_chunks, doc_id, doc_hash = initialize_pdf_processing(
            current_pdf_filepath, vector_store
        )
        if pdf_chunks is not None:  # pdf_chunks can be an empty list if no text content
            all_chunks_for_json_storage.extend(pdf_chunks)

        if not doc_id or not doc_hash:
            print(
                f"Skipping table processing for {pdf_filename} due to earlier errors."
            )
            continue  # Move to the next PDF

        # --- 2b. Extract tables from the current PDF to text files ---
        # TABLE_BASE_DIR is like "data/company_name/text_data"
        # process_tables_from_pdf will create "data/company_name/text_data/tables"
        if pd:  # Check if pandas is available
            print(f"\n--- Extracting Tables from {doc_id} ---")
            # company_specific_table_base_dir is where the "tables" subfolder will be created.
            # e.g. data/revola/text_data
            company_specific_table_base_dir = TABLE_BASE_DIR

            # This function saves files like: data/revola/text_data/tables/THE_PDF_NAME.pdf_page1_table0.txt
            raw_table_files_paths = process_tables_from_pdf(
                current_pdf_filepath, company_specific_table_base_dir
            )

            if raw_table_files_paths:
                # --- 2c. Summarize and upsert tables for the current PDF ---
                print(f"\n--- Upserting Summarized Tables for {doc_id} ---")
                # The directory containing the actual .txt files for tables
                # e.g. data/revola/text_data/tables
                actual_tables_txt_dir = os.path.join(
                    company_specific_table_base_dir, "tables"
                )

                table_summary_chunks = upsert_summarized_tables_for_doc(
                    actual_tables_txt_dir,  # Pass the directory with the .txt files
                    doc_id,
                    doc_hash,
                    vector_store,
                    company,  # Pass company name for LLM prompt
                )
                all_chunks_for_json_storage.extend(table_summary_chunks)
            else:
                print(
                    f"No raw table files were extracted for {doc_id}, skipping summarization."
                )
        else:
            print(
                f"Skipping table processing for {doc_id} because pandas library is not available."
            )

        print(f"Finished processing for: {current_pdf_filepath}")

    # --- 3. Store all accumulated metadata to JSON (once at the end) ---
    if all_chunks_for_json_storage:
        print(
            f"\n--- Storing all ({len(all_chunks_for_json_storage)}) collected metadata to {METADATA_JSON_PATH} ---"
        )
        store_chunks_as_json(all_chunks_for_json_storage, METADATA_JSON_PATH)
    else:
        print("\nNo chunks were processed to store in JSON metadata file.")

    print("\n--- All PDF processing complete. ---")
