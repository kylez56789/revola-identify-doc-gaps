import json
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec, PodSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "revola-revola"
JSON_FILE_PATH = "data/revola/revola_image_rag_metadata.json"
IDS_TO_DELETE_BATCH_SIZE = 100
CSV_FILE_PATH = "data/revola/revola_images_to_delete.csv"


# --- Helper Function to Load IDs from JSON ---
def load_ids_from_json(file_path, images_to_delete_file_path):
    """
    Loads chunk_ids from a JSON file.
    Assumes the JSON file contains a list of objects,
    each with a "chunk_id" key.
    """

    df = pd.read_csv(images_to_delete_file_path)
    image_names_to_delete = []
    ids = []
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        for idx, row in df.iterrows():
            if row.get("IMAGE RECOMMENDATION") == "Remove":
                image_names_to_delete.append(row.get("IMAGE LINK"))
        for item in data:
            if (
                isinstance(item, dict)
                and "metadata" in item
                and item["metadata"].get("image_filename_id") in image_names_to_delete
            ):
                ids.append(item["chunk_id"])
            else:
                print(
                    f"Skipping item with image name: {item["metadata"].get("image_filename_id")}"
                )
        return ids
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return None


# --- Main Deletion Logic ---
def delete_records_from_pinecone():
    """
    Connects to Pinecone, loads IDs from the JSON file,
    and deletes the corresponding records from the specified index.
    """
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY must be set.")
        print("Please set them as environment variables or directly in the script.")
        return

    try:
        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )  # environment is automatically picked up if using new client
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return

    print(f"Connecting to index: '{INDEX_NAME}'...")
    try:
        index = pc.Index(INDEX_NAME)
        print(f"Successfully connected to index '{INDEX_NAME}'.")
        # You can print index stats to confirm connection (optional)
        # print(f"Index stats: {index.describe_index_stats()}")
    except Exception as e:
        print(f"Error connecting to Pinecone index '{INDEX_NAME}': {e}")
        return

    print(f"Loading IDs to delete from '{JSON_FILE_PATH}'...")
    ids_to_delete = load_ids_from_json(JSON_FILE_PATH, CSV_FILE_PATH)

    if not ids_to_delete:
        print("No IDs found or an error occurred while loading IDs. Exiting.")
        return

    print(f"Found {len(ids_to_delete)} IDs to delete.")

    # Delete in batches
    deleted_count = 0
    failed_ids = []

    for i in range(0, len(ids_to_delete), IDS_TO_DELETE_BATCH_SIZE):
        batch_ids = ids_to_delete[i : i + IDS_TO_DELETE_BATCH_SIZE]
        print(f"Attempting to delete batch of {len(batch_ids)} IDs...")
        try:
            delete_response = index.delete(ids=batch_ids, namespace="images")
            print(
                f"Successfully submitted delete request for {len(batch_ids)} IDs. Response: {delete_response}"
            )
            deleted_count += len(
                batch_ids
            )  # Assuming all in batch are processed if no error
        except Exception as e:
            print(f"Error deleting batch of IDs: {e}")
            print(f"Failed to delete IDs: {batch_ids}")
            failed_ids.extend(batch_ids)

    print("\n--- Deletion Summary ---")
    print(f"Total IDs targeted for deletion: {len(ids_to_delete)}")
    # Note: The actual number of vectors deleted in Pinecone might be less if some IDs didn't exist.
    # The `delete` operation doesn't fail if an ID is not found.
    print(f"Number of IDs submitted for deletion: {deleted_count - len(failed_ids)}")
    if failed_ids:
        print(f"Number of IDs that failed to submit for deletion: {len(failed_ids)}")
        print(f"Failed IDs: {failed_ids}")
    else:
        print("All targeted IDs were submitted for deletion successfully.")


if __name__ == "__main__":
    delete_records_from_pinecone()
