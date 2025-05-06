from services.scrape import start_evaluation, crawl_website
from services.extract import process_text_content, extract_keywords
from services.evaluate import evaluate_information_adequacy
import json
import os


def scrape_website(url, company_name, max_pages=100):
    # Take in URL and initialize
    base_domain, visited_urls, pages_to_visit, extracted_text, image_urls = (
        start_evaluation(url)
    )

    print(f"Base Domain: {base_domain}")
    print(f"Visited Urls: {visited_urls}")
    print(f"Extracted Text: {len(extracted_text)} pages")
    print(f"Image URLs: {len(image_urls)} found so far")

    data = []
    for url in visited_urls:
        data.append({"url": url, "text": extracted_text.get(url, "")})

    # Step 4: Write data to JSON file
    with open(f"{company_name}_text_data.json", "w") as f:
        json.dump(data, f, indent=2)
    with open(f"{company_name}_image_urls.json", "w") as f:
        json.dump(image_urls, f, indent=2)

    print(f"Data saved")


def extract_from_data(filename, company_name):
    """Load extracted text from a JSON file and process it."""
    with open(filename, "r") as f:
        data = json.load(f)

    # Extract the text content
    extracted_text = {item["url"]: item["text"] for item in data}

    # Process the text content
    sentences, entities, key_phrases, spacy_entities, spacy_keywords = (
        process_text_content(extracted_text)
    )

    # Print debugging information
    # print(f"NLTK entities: {entities[:20]}")
    # print(f"spaCy entities: {spacy_entities[:20]}")
    # print(f"Key phrases: {key_phrases[:20]}")

    # Save the entities and other data to JSON files
    keywords_filename = f"{company_name}_keywords.json"
    sentences_filename = f"{company_name}_sentences.json"
    nltk_entities_filename = f"{company_name}_nltk_entities.json"
    spacy_entities_filename = f"{company_name}_spacy_entities.json"
    spacy_keywords_filename = f"{company_name}_spacy_keywords.json"

    with open(keywords_filename, "w") as f:
        json.dump(key_phrases, f, indent=2)

    with open(sentences_filename, "w") as f:
        json.dump(sentences, f, indent=2)

    with open(nltk_entities_filename, "w") as f:
        json.dump(entities, f, indent=2)

    with open(spacy_entities_filename, "w") as f:
        json.dump(spacy_entities, f, indent=2)
    with open(spacy_keywords_filename, "w") as f:
        json.dump(spacy_keywords, f, indent=2)

    print(
        f"Data saved to {keywords_filename}, {sentences_filename}, {nltk_entities_filename}, and {spacy_entities_filename}"
    )


def evaluate_company_documentation(url, max_pages=100):
    """Main function to evaluate company documentation."""

    # # Step 5: Evaluate information adequacy
    # evaluation = evaluate_information_adequacy(keywords, extracted_text)

    # # Prepare final report
    # report = {
    #     "url": url,
    #     "pages_crawled": len(visited_urls),
    #     "total_images": len(image_urls),
    #     "top_keywords": keywords[:20],
    #     "information_adequacy": evaluation["overall_rating"],
    #     "adequacy_percentage": evaluation["overall_adequacy_percentage"],
    #     "keyword_evaluations": evaluation["keyword_evaluations"],
    # }

    # return report


# Example usage
if __name__ == "__main__":
    # Get URL from user
    company_url = "https://www.cirrascale.com/"
    company_name = company_url.split("//")[-1].strip(".com/").replace("www.", "")
    json_filename = f"{company_name}_text_data.json"

    if not os.path.exists(json_filename):
        print(f"Data file {json_filename} not found. Scraping website...")
        scrape_website(company_url, company_name)

    extract_from_data(json_filename, company_name)

    # # Run the evaluation
    # report = evaluate_company_documentation(company_url)

    # # Save report to file
    # with open("documentation_evaluation.json", "w") as f:
    #     json.dump(report, f, indent=2)

    # print(f"Evaluation completed. Report saved to documentation_evaluation.json")
