import pandas as pd
import os
import base64
from pathlib import Path
from PIL import Image
import mimetypes
import json
import uuid
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Document not strictly needed if using add_texts, but keep if useful elsewhere
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
CSV_FILE_PATH = "data/cirrascale/cirrascale_qa.csv"
IMAGE_FOLDER = Path("data/cirrascale/answer_images")
OUTPUT_FILE = "data/cirrascale/cirrascale_image_captions_output.csv"
IMAGE_METADATA_JSON_PATH = "data/cirrascale/cirrascale_image_rag_metadata.json"
IMAGE_COLUMN = "image_link"
ANSWER_COLUMN = "answer"
QUESTION_COLUMN = "question"
GENERATED_CAPTION_COLUMN = "generated_caption"
MAX_IMAGES_TO_PROCESS = None
OPENAI_MODEL = "gpt-4o"

# Pinecone and Embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "revola-cirrascale"
PINECONE_NAMESPACE = "images"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = 3072
pc = Pinecone(api_key=PINECONE_API_KEY)


def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found at {filepath} for hashing.")
        return None
    except IOError as e:
        print(f"Warning: Error reading file {filepath} for hashing: {e}")
        return None


def store_chunks_as_json(chunks_to_store, output_filepath, append=False):
    """Stores or appends chunk data to a JSON file."""
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
                f"Warning: Could not read/parse existing metadata file {output_filepath}. Overwriting. Error: {e}"
            )
            existing_data = []

    all_data = existing_data + chunks_to_store
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        action = "Appended" if append and existing_data else "Stored"
        print(
            f"Successfully {action.lower()} {len(chunks_to_store)} chunks metadata. Total chunks in {output_filepath}: {len(all_data)}"
        )
    except (IOError, TypeError) as e:
        print(f"Error writing or serializing JSON metadata to {output_filepath}: {e}")


# --- (encode_image_to_data_url and caption_images remain the same) ---
# Function to encode image to base64 Data URL
def encode_image_to_data_url(image_path):
    """Encodes an image file to a base64 data URL."""
    try:
        # Validate image path
        if not image_path or not isinstance(image_path, (str, Path)):
            print(f"Warning: Invalid image path provided: {image_path}. Skipping.")
            return None
        # Convert to Path object if it's a string
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"Warning: Image file not found at {image_path_obj}. Skipping.")
            return None
        if not image_path_obj.is_file():
            print(f"Warning: Path is not a file {image_path_obj}. Skipping.")
            return None

        # Guess the MIME type
        mime_type, _ = mimetypes.guess_type(image_path_obj)
        if not mime_type or not mime_type.startswith("image/"):
            try:
                with Image.open(image_path_obj) as img:
                    mime_type = Image.MIME.get(img.format)
                    if not mime_type:
                        print(
                            f"Warning: Could not determine MIME type for {image_path_obj}. Trying png."
                        )
                        mime_type = "image/png"  # Default guess
            except Exception as e:
                print(
                    f"Warning: Could not open image {image_path_obj} to determine type: {e}. Skipping."
                )
                return None

        # Read and encode the image
        with open(image_path_obj, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def caption_images(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        exit()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit()

    # Ensure the image folder exists
    if not IMAGE_FOLDER.is_dir():
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
        exit()

    # Initialize the ChatOpenAI model
    try:
        chat = ChatOpenAI(model=OPENAI_MODEL)
    except Exception as e:
        print(f"Error initializing OpenAI model: {e}")
        exit()

    results = []
    processed_count = 0

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        if (
            MAX_IMAGES_TO_PROCESS is not None
            and processed_count >= MAX_IMAGES_TO_PROCESS
        ):
            print(
                f"Reached maximum limit of {MAX_IMAGES_TO_PROCESS} images to process."
            )
            break

        # Get data from row, handle potential missing values
        image_filename = row.get(IMAGE_COLUMN)
        answer_text = row.get(ANSWER_COLUMN, "")
        question_text = row.get(QUESTION_COLUMN, "")

        if pd.isna(image_filename) or not image_filename:
            print(f"Row {index + 1}: Skipping - No image filename.")
            results.append(
                {**row.to_dict(), GENERATED_CAPTION_COLUMN: "N/A - No image filename"}
            )
            continue

        if pd.isna(answer_text):
            answer_text = ""
        if pd.isna(question_text):
            question_text = ""

        # Construct the full image path
        image_path = IMAGE_FOLDER / str(image_filename).strip()

        print(f"\nProcessing Row {index + 1}: Image '{image_filename}'")

        # Encode the image
        image_data_url = encode_image_to_data_url(image_path)

        if not image_data_url:
            print(f"Row {index + 1}: Skipping - Could not encode image.")
            results.append(
                {
                    **row.to_dict(),
                    GENERATED_CAPTION_COLUMN: "N/A - Image encoding error or file not found",
                }
            )
            continue

        # Construct the prompt
        prompt_text = f"""
        Please analyze the following image and the provided text context by doing the following:
        1. You are to generate a 2-3 sentences summary for the image that relates to the context.
        2. You are to also generate a visual summary of what is being shown on the image. Describe the image in as much specificity as possible, enough so that someone can recreate the image without seeing it. For example: "There are two characters doing ___.".
        Give ONLY the caption and the visual summary separated by JUST a ';'. NO OTHER TEXT.

        Context from CSV:
        Question: {question_text if question_text else 'N/A'}
        Answer: {answer_text if answer_text else 'N/A'}
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )

        # Call the OpenAI API
        try:
            print("Sending request to OpenAI...")
            response = chat.invoke([message])
            generated_caption_raw = response.content.strip()
            # Basic validation of the response format
            if ";" not in generated_caption_raw:
                print(
                    f"Warning: OpenAI response for row {index + 1} did not contain ';'. Using full response as caption."
                )
                generated_caption = generated_caption_raw + "; N/A"  # Add placeholder
            else:
                generated_caption = generated_caption_raw

            print(f"Generated Output: {generated_caption}")
            results.append(
                {**row.to_dict(), GENERATED_CAPTION_COLUMN: generated_caption}
            )
            processed_count += 1
        except Exception as e:
            print(f"Error calling OpenAI API for row {index + 1}: {e}")
            results.append({**row.to_dict(), GENERATED_CAPTION_COLUMN: f"Error: {e}"})

    # Save results
    if results:
        output_df = pd.DataFrame(results)
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"\nError saving results to {OUTPUT_FILE}: {e}")
    else:
        print("\nNo images were processed or no results generated.")


# --- Modified initialize_image_vector_store ---
def initialize_image_vector_store(captioned_csv_path):
    """
    Loads captioned image data, stores detailed metadata externally (JSON),
    and upserts captions to Pinecone using generated IDs and a specific namespace.

    Args:
        captioned_csv_path (str): Path to the CSV file containing generated captions
                                  (output from caption_images).

    Returns:
        PineconeVectorStore or None: The vector store object connected to the index,
                                     or None if a critical error occurs.
    """
    try:
        # Load the CSV file with captions
        df = pd.read_csv(captioned_csv_path)
        print(f"Loaded {len(df)} rows with captions from {captioned_csv_path}")
    except FileNotFoundError:
        print(f"Error: Captioned CSV file not found at {captioned_csv_path}")
        return None
    except Exception as e:
        print(f"Error reading captioned CSV file: {e}")
        return None

    columns_to_drop = [
        "S No.",
        "IMAGE/ SCREENSHOT",
        "ANSWER QUALITY",
        "IMAGE RECOMMENDATION",
        "FEEDBACK",
        "RAG Description",
    ]
    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns], errors="ignore"
    )
    df.columns = df.columns.str.lower().str.replace(
        " ", "_"
    )  # Standardize column names

    image_chunks_to_store = []
    processed_count = 0
    skipped_count = 0

    print("Processing rows for metadata generation and upsert preparation...")
    for idx, row in df.iterrows():
        # --- Get core data ---
        image_filename = row.get(IMAGE_COLUMN)  # Use the standardized column name
        full_caption_text = row.get(GENERATED_CAPTION_COLUMN, "")

        # --- Validate required data ---
        if pd.isna(image_filename) or not image_filename:
            # print(f"Row {idx + 1}: Skipping - Missing image filename.")
            skipped_count += 1
            continue
        if (
            pd.isna(full_caption_text)
            or not full_caption_text
            or ";" not in full_caption_text
        ):
            # print(f"Row {idx + 1}: Skipping - Missing or invalid generated caption: '{full_caption_text}'")
            skipped_count += 1
            continue

        # --- Construct paths and calculate hash ---
        image_path = IMAGE_FOLDER / str(image_filename).strip()
        image_hash = calculate_file_hash(image_path)
        if not image_hash:
            print(
                f"Row {idx + 1}: Skipping - Could not hash image file at {image_path}"
            )
            skipped_count += 1
            continue  # Skip if image file missing or unreadable

        # --- Split caption and prepare text for embedding ---
        try:
            caption_part, visual_summary_part = [
                part.strip() for part in full_caption_text.split(";", 1)
            ]
        except ValueError:
            print(
                f"Row {idx + 1}: Skipping - Could not split caption correctly: '{full_caption_text}'"
            )
            skipped_count += 1
            continue

        # Text to be embedded (can be caption, summary, or both combined)
        # Let's use the full text for embedding as it was generated together
        text_for_embedding = full_caption_text

        # --- Generate Chunk ID ---
        chunk_id = str(uuid.uuid4())

        # --- Create Unified Metadata ---
        # Store original CSV data excluding the full caption itself
        original_csv_data = row.drop(
            labels=[GENERATED_CAPTION_COLUMN], errors="ignore"
        ).to_dict()
        # Convert potential non-serializable types (like Timestamps if any)
        for k, v in original_csv_data.items():
            if pd.isna(v):
                original_csv_data[k] = None  # Convert NaN/NaT to None
            # Add other type conversions if necessary (e.g., Timestamp to string)

        metadata = {
            "chunk_id": chunk_id,
            "image_filename_id": str(image_filename).strip(),
            "image_content_hash": image_hash,
            "embedding_model": embeddings.model,
            "source_type": "captioned_image",  # Differentiator
            "original_image_path": str(image_path.resolve()),  # Store absolute path
            "caption_text": caption_part,
            "visual_summary_text": visual_summary_part,
        }

        # --- Add to list for JSON storage ---
        image_chunks_to_store.append(
            {
                "chunk_id": chunk_id,
                "text": text_for_embedding,  # The text that will be embedded
                "metadata": metadata,
            }
        )
        processed_count += 1

    print(
        f"Prepared {processed_count} valid image captions for storage and upsert. Skipped {skipped_count} rows."
    )

    # --- Store Image Chunk Metadata Externally (Overwrite for initial run) ---
    if image_chunks_to_store:
        print(f"Storing image metadata to {IMAGE_METADATA_JSON_PATH}...")
        store_chunks_as_json(
            image_chunks_to_store, IMAGE_METADATA_JSON_PATH, append=False
        )  # Use append=True for updates
    else:
        print("No valid image chunks to store. Exiting.")
        return None

    # --- Ensure Pinecone Index Exists ---
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
            return None
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    # --- Prepare for Upsert ---
    chunk_texts = [item["text"] for item in image_chunks_to_store]
    chunk_metadatas = [item["metadata"] for item in image_chunks_to_store]
    chunk_ids = [item["chunk_id"] for item in image_chunks_to_store]

    # --- Upsert to Pinecone with Namespace and IDs ---
    print(
        f"Upserting {len(chunk_texts)} image captions to Pinecone namespace '{PINECONE_NAMESPACE}'..."
    )
    vector_db = None
    try:
        # Connect to the index
        vector_db = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE,  # Specify the namespace
        )
        # Upsert data using add_texts with specified IDs
        vector_db.add_texts(
            texts=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids,  # Pass the generated IDs
            batch_size=100,  # Adjust as needed
            namespace=PINECONE_NAMESPACE,  # Specify namespace again for add_texts
        )
        print("Image caption upsert complete.")
    except Exception as e:
        print(f"Error upserting image captions to Pinecone: {e}")
        # Depending on the error, vector_db might be partially initialized or None
        return None  # Indicate failure

    return vector_db  # Return the vector store object


# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Generate captions if the output file doesn't exist or needs updating
    # You might want to add logic here to check if captioning is needed
    if not os.path.exists(OUTPUT_FILE):
        print(
            f"Caption output file {OUTPUT_FILE} not found. Running captioning process..."
        )
        caption_images(CSV_FILE_PATH)
    else:
        print(f"Using existing caption output file: {OUTPUT_FILE}")
        # Optionally add logic to force re-captioning if needed

    # Step 2: Initialize the vector store with captions and metadata
    print("\n--- Initializing Image Vector Store ---")
    # Pass the path to the *captioned* CSV file
    vector_store = initialize_image_vector_store(OUTPUT_FILE)

    if vector_store:
        print("\nImage vector store initialization and upsert successful.")
    else:
        print("\nImage vector store initialization failed.")

    print("\n--- Script Finished ---")
