import pandas as pd
import os
import base64
from pathlib import Path
from PIL import Image
import mimetypes
import json
import uuid
import hashlib
import argparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(
    description="parser for image ingenstion to pinecone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("company", help="Company to handle")
parser.add_argument("-c", "--csv", action="store_true", help="Caption from csv file")
args = parser.parse_args()
config = vars(args)
company = config["company"].lower()
caption_from_csv = args.csv

# --- Constants ---
CSV_FILE_PATH = f"data/{company}/{company}_qa.csv"
IMAGE_FOLDER = Path(f"data/{company}/answer_images")
OUTPUT_FILE = f"data/{company}/{company}_captions_output.csv"
IMAGE_METADATA_JSON_PATH = f"data/{company}/{company}_image_rag_metadata.json"
IMAGE_COLUMN = "image_link"  # Used for CSV reading
ANSWER_COLUMN = "answer"
DESCRIPTION_COLUMN = "description"
QUESTION_COLUMN = "question"
GENERATED_CAPTION_COLUMN = "generated_caption"
IMAGE_FILENAME_COLUMN = "image_filename"
MAX_IMAGES_TO_PROCESS = None
OPENAI_MODEL = "gpt-4.1"

# Pinecone and Embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = f"revola-{company}"
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


# --- NEW HELPER FUNCTION ---
def find_image_file(base_folder: Path, filename_from_csv: str) -> Path | None:
    """
    Tries to find an image file in the base_folder.
    It checks:
    1. The filename as provided in the CSV.
    2. The filename from CSV with common image extensions appended (if it doesn't already have a common one).
    3. The base name (filename from CSV without its original extension) with common image extensions appended.
    """
    if not filename_from_csv:  # Handle empty or None filenames
        return None

    stripped_filename = str(filename_from_csv).strip()
    common_extensions = [".png", ".jpg", ".jpeg"]

    # Attempt 1: Check the filename as-is from CSV
    path_as_is = base_folder / stripped_filename
    if path_as_is.exists() and path_as_is.is_file():
        return path_as_is

    name_part, csv_ext_part = os.path.splitext(stripped_filename)
    csv_ext_lower = csv_ext_part.lower()

    # Attempt 2: If filename from CSV has no common extension (or no extension at all),
    # try appending common extensions to the full stripped_filename.
    if csv_ext_lower not in common_extensions:
        for ext_to_add in common_extensions:
            potential_path = base_folder / (stripped_filename + ext_to_add)
            if potential_path.exists() and potential_path.is_file():
                return potential_path

    # Attempt 3: Try appending common extensions to the name_part (filename without any original extension).
    # This covers cases where CSV has "file.txt" but image is "file.png",
    # or CSV has "file.jpeg" (not found) but "file.png" exists.
    for ext_to_add in common_extensions:
        potential_path = base_folder / (name_part + ext_to_add)
        # Avoid re-checking the same path_as_is if it was already name_part + its_own_common_ext
        if potential_path == path_as_is and csv_ext_lower in common_extensions:
            continue
        if potential_path.exists() and potential_path.is_file():
            return potential_path

    return None


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

    try:
        chat = ChatOpenAI(model=OPENAI_MODEL)
    except Exception as e:
        print(f"Error initializing OpenAI model: {e}")
        exit()

    results = []
    processed_count = 0

    for index, row in df.iterrows():
        if (
            MAX_IMAGES_TO_PROCESS is not None
            and processed_count >= MAX_IMAGES_TO_PROCESS
        ):
            print(
                f"Reached maximum limit of {MAX_IMAGES_TO_PROCESS} images to process."
            )
            break

        image_filename_csv = row.get(IMAGE_COLUMN)  # Get filename from CSV
        answer_text = row.get(DESCRIPTION_COLUMN, "")
        question_text = row.get(QUESTION_COLUMN, "")

        if (
            pd.isna(image_filename_csv)
            or not str(image_filename_csv).strip()  # Ensure it's a string before strip
            or str(image_filename_csv).strip() == "Not Required"
        ):
            print(f"Row {index + 1}: Skipping - No image filename provided in CSV.")
            results.append(
                {**row.to_dict(), GENERATED_CAPTION_COLUMN: "N/A - No image filename"}
            )
            continue

        # Ensure image_filename_csv is treated as a string for find_image_file
        image_filename_csv_str = str(image_filename_csv).strip()

        if pd.isna(answer_text):
            answer_text = ""
        if pd.isna(question_text):
            question_text = ""

        # --- MODIFIED: Use find_image_file helper ---
        image_path = find_image_file(IMAGE_FOLDER, image_filename_csv_str)

        if not image_path:
            print(
                f"Row {index + 1}: Skipping - Image file not found for '{image_filename_csv_str}' (tried common extensions) in {IMAGE_FOLDER}."
            )
            results.append(
                {
                    **row.to_dict(),
                    GENERATED_CAPTION_COLUMN: "N/A - Image file not found",
                }
            )
            continue

        actual_image_filename = image_path.name  # Get the name with the found extension
        print(
            f"\nProcessing Row {index + 1}: Image '{actual_image_filename}' (found from CSV entry '{image_filename_csv_str}')"
        )

        image_data_url = encode_image_to_data_url(image_path)

        if not image_data_url:
            print(
                f"Row {index + 1}: Skipping - Could not encode image '{actual_image_filename}'."
            )
            results.append(
                {
                    **row.to_dict(),
                    GENERATED_CAPTION_COLUMN: "N/A - Image encoding error",
                }
            )
            continue

        prompt_text = f"""
        Please analyze the following image and the provided text context by doing the following:
        1. You are to generate a 4-5 sentence summary for the image that relates to the context.
        2. You are to also generate a visual summary of what is being shown on the image. Describe the image in as much specificity as possible, enough so that someone can recreate the image without seeing it. For example: "There are two characters doing ___.".
        Give ONLY the slide image summary and the slide visual summary separated by JUST a ';'. NO OTHER TEXT before or after. Example: Image summary text; Visual summary text.

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

        try:
            print(f"Sending request to OpenAI for '{actual_image_filename}'...")
            response = chat.invoke([message])
            generated_caption_raw = response.content.strip()
            if ";" not in generated_caption_raw:
                print(
                    f"Warning: OpenAI response for row {index + 1} ('{actual_image_filename}') did not contain ';'. Using full response as caption."
                )
                generated_caption = generated_caption_raw + "; N/A"
            else:
                generated_caption = generated_caption_raw
            print(f"Generated Output: {generated_caption}")
            results.append(
                {**row.to_dict(), GENERATED_CAPTION_COLUMN: generated_caption}
            )
            processed_count += 1
        except Exception as e:
            print(
                f"Error calling OpenAI API for row {index + 1} ('{actual_image_filename}'): {e}"
            )
            results.append({**row.to_dict(), GENERATED_CAPTION_COLUMN: f"Error: {e}"})

    if results:
        output_df = pd.DataFrame(results)
        try:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"\nError saving results to {OUTPUT_FILE}: {e}")
    else:
        print("\nNo images were processed or no results generated.")


def caption_images_in_directory(
    input_directory_str: str,
    output_csv_filename_str: str = "image_captions_output.csv",
):
    """
    Reads all PNG or JPG/JPEG files from a specified directory, generates captions
    for them using an AI model, and stores the filenames and captions in a CSV file
    within the same input directory.
    (This function already handles extensions correctly via glob, so no changes needed here for the user's specific problem)
    """
    input_directory = Path(input_directory_str).resolve()
    output_csv_path = input_directory / output_csv_filename_str

    if not input_directory.is_dir():
        print(f"Error: Image directory '{input_directory}' not found.")
        return

    try:
        chat = ChatOpenAI(model=OPENAI_MODEL)
        print(f"Successfully initialized OpenAI model: {OPENAI_MODEL}")
    except Exception as e:
        print(f"Error initializing OpenAI model: {e}")
        return

    results = []
    processed_count = 0
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_directory.glob(ext)))

    if not image_files:
        print(
            f"No compatible image files (PNG, JPG, JPEG) found in '{input_directory}'."
        )
        return

    print(f"Found {len(image_files)} images in '{input_directory}'.")
    print(f"Output CSV will be saved to: {output_csv_path}")

    for i, image_path_obj in enumerate(
        image_files
    ):  # Renamed image_path to image_path_obj to avoid conflict
        if (
            MAX_IMAGES_TO_PROCESS is not None
            and processed_count >= MAX_IMAGES_TO_PROCESS
        ):
            print(
                f"Reached maximum limit of {MAX_IMAGES_TO_PROCESS} images to process."
            )
            break

        image_filename = image_path_obj.name
        print(
            f"\nProcessing Image {processed_count + 1}/{len(image_files)}: '{image_filename}'"
        )

        image_data_url = encode_image_to_data_url(image_path_obj)

        if not image_data_url:
            print(f"Skipping '{image_filename}' - Could not encode image.")
            results.append(
                {
                    IMAGE_FILENAME_COLUMN: image_filename,
                    GENERATED_CAPTION_COLUMN: "N/A - Image encoding error or file not found",
                }
            )
            continue

        prompt_text = f"""
        Please analyze the following images of presentation slides and do the following:
        1. Generate a 4-5 sentence summary for the slide image. This summary should be descriptive and engaging and should summarize what the slide is trying to show.
        2. Generate a visual summary of what is actually being shown on the slide image. Describe the slide with enough specificity that someone could visualize or even attempt to recreate the scene without seeing the original image. For example: "A close-up shot of two red apples on a rustic wooden table, with soft morning light filtering from a window on the left."
        Give ONLY the slide image summary and the slide visual summary separated by JUST a ';'. NO OTHER TEXT before or after. Example: Image summary text; Visual summary text.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
        try:
            print(f"Sending request to OpenAI for '{image_filename}'...")
            response = chat.invoke([message])
            generated_caption_raw = response.content.strip()
            if ";" not in generated_caption_raw:
                print(
                    f"Warning: OpenAI response for '{image_filename}' did not contain ';'. Using full response as caption and marking visual summary as N/A."
                )
                generated_caption = (
                    generated_caption_raw + "; N/A - Visual summary format error"
                )
            else:
                generated_caption = generated_caption_raw
            print(f"Generated Output: {generated_caption}")
            results.append(
                {
                    IMAGE_FILENAME_COLUMN: image_filename,
                    GENERATED_CAPTION_COLUMN: generated_caption,
                }
            )
            processed_count += 1
        except Exception as e:
            print(f"Error calling OpenAI API for '{image_filename}': {e}")
            results.append(
                {
                    IMAGE_FILENAME_COLUMN: image_filename,
                    GENERATED_CAPTION_COLUMN: f"Error: {e}",
                }
            )
    if results:
        output_df = pd.DataFrame(results)
        try:
            output_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            print(
                f"\nProcessing complete. {processed_count} images processed. Results saved to {output_csv_path}"
            )
        except Exception as e:
            print(f"\nError saving results to {output_csv_path}: {e}")
    else:
        print("\nNo images were processed or no results were generated to save.")


def initialize_image_vector_store(captioned_csv_path):
    try:
        df = pd.read_csv(captioned_csv_path)
        print(f"Loaded {len(df)} rows with captions from {captioned_csv_path}")
    except FileNotFoundError:
        print(f"Error: Captioned CSV file not found at {captioned_csv_path}")
        return None
    except Exception as e:
        print(f"Error reading captioned CSV file: {e}")
        return None

    columns_to_drop = [  # Example columns, adjust if necessary
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
    # Ensure column names are standardized before accessing them
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")

    image_chunks_to_store = []
    processed_count = 0
    skipped_count = 0

    print("Processing rows for metadata generation and upsert preparation...")
    for idx, row in df.iterrows():
        # Use standardized column name for image_link from CSV
        image_filename_csv = row.get(IMAGE_COLUMN)  # IMAGE_COLUMN is "image_link"
        full_caption_text = row.get(
            GENERATED_CAPTION_COLUMN, ""
        )  # Already standardized format

        if pd.isna(image_filename_csv) or not str(image_filename_csv).strip():
            skipped_count += 1
            continue

        # Ensure image_filename_csv is treated as a string
        image_filename_csv_str = str(image_filename_csv).strip()

        if (
            pd.isna(full_caption_text)
            or not full_caption_text
            or ";" not in full_caption_text
        ):
            skipped_count += 1
            continue

        # --- MODIFIED: Use find_image_file helper ---
        # IMAGE_FOLDER needs to be the same as used in caption_images if they are from the same source
        found_image_path = find_image_file(IMAGE_FOLDER, image_filename_csv_str)

        if not found_image_path:
            print(
                f"Row {idx + 1}: Skipping - Image file not found for '{image_filename_csv_str}' (tried common extensions) in {IMAGE_FOLDER}. Cannot calculate hash."
            )
            skipped_count += 1
            continue

        actual_image_filename_with_ext = (
            found_image_path.name
        )  # Actual filename with extension
        image_hash = calculate_file_hash(found_image_path)
        if not image_hash:
            print(
                f"Row {idx + 1}: Skipping - Could not hash image file at {found_image_path}"
            )
            skipped_count += 1
            continue

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

        # Use standardized column names for answer and question
        original_answer = row.get(
            DESCRIPTION_COLUMN.lower().replace(" ", "_"), ""
        )  # e.g., "answer"
        original_question = row.get(
            QUESTION_COLUMN.lower().replace(" ", "_"), ""
        )  # e.g., "question"

        text_for_embedding = f"{caption_part} {visual_summary_part} Question: {original_question} Answer: {original_answer}".strip()
        chunk_id = str(uuid.uuid4())

        metadata = {
            "chunk_id": chunk_id,
            "image_filename_id": actual_image_filename_with_ext,  # Use actual filename with extension
            "image_content_hash": image_hash,
            "embedding_model": embeddings.model,
            "source_type": "captioned_image",
            "caption_text": caption_part,
            "visual_summary_text": visual_summary_part,
            "original_question": (
                original_question if pd.notna(original_question) else None
            ),
            "original_answer": original_answer if pd.notna(original_answer) else None,
            # Add other original CSV data if needed, ensure they are serializable
        }
        # Clean up NaN from metadata before storing
        metadata = {k: (v if pd.notna(v) else None) for k, v in metadata.items()}

        image_chunks_to_store.append(
            {
                "chunk_id": chunk_id,
                "text": text_for_embedding,
                "metadata": metadata,
            }
        )
        processed_count += 1

    print(
        f"Prepared {processed_count} valid image captions for storage and upsert. Skipped {skipped_count} rows."
    )

    if image_chunks_to_store:
        print(f"Storing image metadata to {IMAGE_METADATA_JSON_PATH}...")
        store_chunks_as_json(
            image_chunks_to_store, IMAGE_METADATA_JSON_PATH, append=False
        )
    else:
        print("No valid image chunks to store. Exiting.")
        return None

    # Check if Pinecone index name is valid
    if not PINECONE_INDEX_NAME:
        print("Error: PINECONE_INDEX_NAME is not set or empty.")
        return None

    # Ensure Pinecone index exists
    index_exists = False
    try:
        if PINECONE_INDEX_NAME in pc.list_indexes().names:
            index_exists = True
    except Exception as e:
        print(f"Error checking for Pinecone index {PINECONE_INDEX_NAME}: {e}")

    if not index_exists:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", region="us-east-1"
                ),  # Ensure region is correct
            )
            print(f"Pinecone index {PINECONE_INDEX_NAME} created successfully.")
        except Exception as e:  # Catch specific Pinecone errors if possible
            if (
                "already exists" in str(e).lower()
            ):  # Handle race condition or stale cache
                print(
                    f"Pinecone index {PINECONE_INDEX_NAME} may already exist (error during creation check). Proceeding."
                )
            else:
                print(f"Error creating Pinecone index {PINECONE_INDEX_NAME}: {e}")
                return None
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    chunk_texts = [item["text"] for item in image_chunks_to_store]
    chunk_metadatas = [item["metadata"] for item in image_chunks_to_store]
    chunk_ids = [item["chunk_id"] for item in image_chunks_to_store]

    vector_db = None
    try:
        vector_db = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE,
        )
        print(
            f"Upserting {len(chunk_texts)} image captions to Pinecone namespace '{PINECONE_NAMESPACE}'..."
        )
        vector_db.add_texts(
            texts=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
            batch_size=100,  # Pinecone recommends batch sizes up to 100 for optimal performance
            namespace=PINECONE_NAMESPACE,
        )
        print("Image caption upsert complete.")
    except Exception as e:
        print(f"Error upserting image captions to Pinecone: {e}")
        return None

    return vector_db


# --- Main Execution ---
if __name__ == "__main__":
    if caption_from_csv:
        if not Path(CSV_FILE_PATH).exists():
            print(f"Error: Input CSV file for captioning not found: {CSV_FILE_PATH}")
            print(
                "Please ensure the CSV file exists or run without -c flag for directory processing."
            )
        else:
            if not os.path.exists(OUTPUT_FILE):
                print(
                    f"Caption output file {OUTPUT_FILE} not found. Running captioning process..."
                )
                caption_images(CSV_FILE_PATH)
            else:
                print(f"Using existing caption output file: {OUTPUT_FILE}")

            if os.path.exists(OUTPUT_FILE):
                print("\n--- Initializing Image Vector Store ---")
                vector_store = initialize_image_vector_store(OUTPUT_FILE)
                if vector_store:
                    print("\nImage vector store initialization and upsert successful.")
                else:
                    print("\nImage vector store initialization failed.")
            else:
                print(
                    f"Skipping vector store initialization as caption output file {OUTPUT_FILE} was not found or created."
                )

    else:
        image_directory_to_process = (
            f"data/{company}/slides"  # Example, adjust as needed
        )
        print(f"Captioning images from directory: {image_directory_to_process}")
        caption_images_in_directory(
            image_directory_to_process, f"{company}_directory_captions.csv"
        )

    print("\n--- Script Finished ---")
