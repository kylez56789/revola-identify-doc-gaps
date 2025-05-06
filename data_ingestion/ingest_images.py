import pandas as pd
import os
import base64
from pathlib import Path
from PIL import Image
import mimetypes

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Options for captioning images
CSV_FILE_PATH = "data/revola_answer_images/revola_main_qa.csv"
IMAGE_FOLDER = "data/revola_answer_images"
OUTPUT_FILE = "data/revola_captioned_data/revola_main_image_captions_output.csv"
IMAGE_COLUMN = "image_link"
ANSWER_COLUMN = "answer"
QUESTION_COLUMN = "question"
MAX_IMAGES_TO_PROCESS = None
OPENAI_MODEL = "gpt-4o"

# For Pinecone vector db
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "revola-revola"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = 3072  # for OpenAI text-embedding-3-large
pc = Pinecone(api_key=PINECONE_API_KEY)


# Function to encode image to base64 Data URL
def encode_image_to_data_url(image_path):
    """Encodes an image file to a base64 data URL."""
    try:
        # Validate image path
        if not image_path or not isinstance(image_path, (str, Path)):
            print(f"Warning: Invalid image path provided: {image_path}. Skipping.")
            return None
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return None

        # Guess the MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            # Fallback or attempt to determine type if not guessed
            try:
                with Image.open(image_path) as img:
                    mime_type = Image.MIME.get(img.format)
                    if not mime_type:
                        print(
                            f"Warning: Could not determine MIME type for {image_path}. Trying png."
                        )
                        mime_type = "image/png"  # Default guess
            except Exception as e:
                print(
                    f"Warning: Could not open image {image_path} to determine type: {e}. Skipping."
                )
                return None

        # Read and encode the image
        with open(image_path, "rb") as image_file:
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
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
        print("Please create it and place the images inside.")
        exit()

    # Initialize the ChatOpenAI model
    try:
        chat = ChatOpenAI(model=OPENAI_MODEL)  # Adjust max_tokens as needed
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
        question_text = row.get(QUESTION_COLUMN, "")  # Optional

        if pd.isna(image_filename) or not image_filename:
            print(
                f"Row {index + 1}: Skipping - No image filename found in column '{IMAGE_COLUMN}'."
            )
            results.append(
                {**row.to_dict(), "generated_caption": "N/A - No image filename"}
            )
            continue

        if pd.isna(answer_text):
            answer_text = "No answer provided in CSV."  # Provide default if needed

        # Construct the full image path
        image_path = os.path.join(IMAGE_FOLDER, str(image_filename).strip())

        print(f"\nProcessing Row {index + 1}: Image '{image_filename}'")

        # Encode the image
        image_data_url = encode_image_to_data_url(image_path)

        if not image_data_url:
            print(
                f"Row {index + 1}: Skipping - Could not encode image '{image_filename}'."
            )
            results.append(
                {
                    **row.to_dict(),
                    "generated_caption": "N/A - Image encoding error or file not found",
                }
            )
            continue

        # --- Construct the prompt for the OpenAI model ---
        prompt_text = f"""
        Please analyze the following image and the provided text context by doing the following:
        1. You are to generate a 2-3 sentences summary for the image that relates to the context.
        2. You are to also generate a visual summary of what is being shown on the image. Describe the image in as much specificity as possible, enough so that someone can recreate the image without seeing it. For example: "There are two characters doing ___.".
        Give only the caption and the visual summary separated by just a ';'.

        Context from CSV:
        Question: {question_text if pd.notna(question_text) else 'N/A'}
        Answer: {answer_text}

        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ]
        )

        # --- Call the OpenAI API ---
        try:
            print("Sending request to OpenAI...")
            response = chat.invoke([message])
            generated_caption = response.content
            print(f"Generated Caption: {generated_caption}")
            results.append({**row.to_dict(), "generated_caption": generated_caption})
            processed_count += 1

        except Exception as e:
            print(f"Error calling OpenAI API for row {index + 1}: {e}")
            results.append({**row.to_dict(), "generated_caption": f"Error: {e}"})

    # --- Save results ---
    if results:
        output_df = pd.DataFrame(results)
        try:
            output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"\nError saving results to {OUTPUT_FILE}: {e}")
    else:
        print("\nNo images were processed or no results generated.")


def initialize_image_vector_store(csv_file_path):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    columns_to_drop = ["S No.", "IMAGE RECOMMENDATION", "ANSWER QUALITY"]
    for column in columns_to_drop:
        if column in df.columns:
            df = df.drop(columns=column)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    documents = []
    for idx, row in df.iterrows():
        # Extract the text from the "generated_caption" column
        text = row.get("generated_caption", "").strip()
        if not row.get("image_link") or not text or len(text.split(";")) < 2:
            print(f"Skipping row with not image: {text}")
            continue

        # Use the remaining columns as metadata
        metadata = row.drop(labels=["generated_caption"]).to_dict()
        metadata["caption"], metadata["visual_summary"] = text.split(";")
        # print(metadata)
        # Create a Document object
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    print(len(documents))

    # Chunking text for efficient retrieval
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # split_docs = text_splitter.split_documents(documents)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Get Pinecone index object
    # index = pc.Index(PINECONE_INDEX_NAME)

    vector_db = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=PINECONE_INDEX_NAME, namespace="images"
    )
    return vector_db


# caption_images(CSV_FILE_PATH)
initialize_image_vector_store(OUTPUT_FILE)
