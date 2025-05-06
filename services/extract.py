import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import yake
import spacy
import string
from dotenv import load_dotenv

load_dotenv()


def extract_named_entities_spacy(text):
    """Improved NER using spaCy (alternative version)"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [
        (ent.text, ent.label_)
        for ent in doc.ents
        if ent.label_
        in [
            "ORG",
            "CARDINAL",
            "DATE",
            "GPE",
            "PERSON",
            "MONEY",
            "PRODUCT",
            "TIME",
            "PERCENT",
            "WORK_OF_ART",
            "QUANTITY",
            "NORP",
            "LOC",
            "EVENT",
            "ORDINAL",
            "FAC",
            "LAW",
            "LANGUAGE",
        ]
    ]


def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


def process_text_content(extracted_text):
    """Process the extracted text to identify key information."""
    nltk.download("maxent_ne_chunker_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    # Combine all extracted text into a single string
    all_text = " ".join(extracted_text.values())

    # Tokenize text into sentences
    sentences = sent_tokenize(all_text)

    # Extract named entities (people, organizations, locations)
    entities = extract_named_entities(all_text)

    # Extract key phrases and topics
    keywords, tfidf_keywords, rake_keywords, yake_keywords = (
        [],
        [],
        [],
        [],
    )
    for chunk in split_text_into_chunks(all_text):
        combined_keywords, tfidf_keywords, rake_keywords, yake_keywords = (
            extract_keywords(chunk)
        )
        keywords.extend(combined_keywords)
        tfidf_keywords.extend(tfidf_keywords)
        rake_keywords.extend(rake_keywords)
        yake_keywords.extend(yake_keywords)
    spacy_keywords = extract_keywords_spacy(all_text)

    print(f"Num tfidf keywords: {len(tfidf_keywords)}")
    print(f"Num yake keywords: {len(yake_keywords)}")
    print(f"Num rake keywords: {len(rake_keywords)}")
    print(f"Num total keywords: {len(keywords)}")
    spacy_entities = extract_named_entities_spacy(all_text)

    return sentences, entities, keywords, spacy_entities, spacy_keywords


def extract_keywords_spacy(text, max_keywords=50, min_word_length=3, min_term_count=2):
    """
    Extract important keywords and phrases from text.

    Args:
        text (str): Input text to analyze
        max_keywords (int): Number of top keywords to return
        min_word_length (int): Minimum length for individual words
        min_term_count (int): Minimum occurrence count for terms

    Returns:
        list: Sorted list of keywords with counts
    """
    # Process text with spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    keywords = []
    custom_stop_words = {
        "product",
        "service",
        "solution",
        "system",
    }  # Add domain-specific stop words
    for doc_chunk in split_text_into_chunks(text):
        doc = nlp(doc_chunk)
        # Extract entities and noun chunks
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "TECH"]:  # Filter entity types
                keywords.append(ent.text.lower())

        for chunk in doc.noun_chunks:
            # Remove leading determiners/possessives and trailing stop words
            while len(chunk) > 0 and chunk[0].dep_ in ["det", "poss"]:
                chunk = chunk[1:]
            if len(chunk) > 0:
                phrase = chunk.text.lower().strip()
                if len(phrase) > min_word_length and phrase not in custom_stop_words:
                    keywords.append(phrase)

        # Extract meaningful individual words
        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN", "ADJ"]
                and not token.is_stop
                and not token.is_punct
                and len(token.text) >= min_word_length
                and token.text.lower() not in custom_stop_words
            ):
                keywords.append(token.lemma_.lower())

        # Count and filter keywords
        keyword_counts = Counter(keywords)
        filtered_counts = {
            k: v
            for k, v in keyword_counts.items()
            if v >= min_term_count
            and all(c not in string.punctuation for c in k)
            and not any(word in k.split() for word in custom_stop_words)
        }
    keywords.extend(
        set(
            sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[
                :max_keywords
            ]
        )
    )
    # Return sorted by frequency
    return keywords


def process_image_information(image_urls, soup):
    """Extract information from images using alt text and surrounding context."""
    image_info = []

    for img_url in image_urls:
        # Find the image tag
        img_tags = soup.find_all("img", src=lambda src: img_url in src)

        for img in img_tags:
            # Get alt text
            alt_text = img.get("alt", "")

            # Get surrounding text
            context = get_image_context(img)

            image_info.append(
                {"url": img_url, "alt_text": alt_text, "context": context}
            )

    return image_info


def combine_keyword_results(rake_keywords, yake_keywords, tfidf_keywords, top_n=30):
    """Combine and rank keywords from RAKE, YAKE, and TF-IDF."""
    # Create frequency counters
    rake_counter = Counter(rake_keywords)
    yake_counter = Counter(yake_keywords)
    tfidf_counter = Counter(tfidf_keywords)

    # Merge scores using weighted average
    combined = {}
    for kw in set(rake_keywords + yake_keywords + tfidf_keywords):
        combined[kw] = (
            rake_counter.get(kw, 0) * 0.5  # Weight RAKE higher
            + yake_counter.get(kw, 0) * 0.3
            + tfidf_counter.get(kw, 0) * 0.2
        )

    # Sort by combined score
    sorted_keywords = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # Remove duplicates and return top results
    seen = set()
    unique_keywords = [
        kw
        for kw, score in sorted_keywords
        if not (kw.lower() in seen or seen.add(kw.lower()))
    ]

    return unique_keywords[:top_n]


def extract_keywords(text_data):
    """Extract keywords using RAKE, YAKE, and TF-IDF, and combine results."""

    # Method 1: RAKE
    rake_keywords = extract_keywords_rake(text_data)

    # Method 2: YAKE
    yake_keywords = extract_keywords_yake(text_data)

    # Method 3: TF-IDF
    tfidf_keywords = extract_keywords_tfidf(text_data)

    # Combine results
    combined_keywords = combine_keyword_results(
        rake_keywords, yake_keywords, tfidf_keywords
    )

    return combined_keywords, tfidf_keywords, rake_keywords, yake_keywords


def extract_keywords_rake(text):
    """Extract keywords using RAKE with custom settings."""
    # Use a custom stopword list (optional)
    custom_stopwords = stopwords.words("english")
    rake_class = Rake(
        stopwords=custom_stopwords, max_length=5
    )  # Allow phrases up to 5 words
    rake_class.extract_keywords_from_text(text)
    extracted_keywords = rake_class.get_ranked_phrases()

    return extracted_keywords


def extract_keywords_yake(text):
    """Extract keywords using YAKE with optimized parameters."""
    language = "en"
    max_ngram_size = 5  # Allow phrases up to 5 words
    deduplication_threshold = 0.9  # Reduce redundancy
    deduplication_algo = "seqm"
    window_size = 3  # Consider a larger context window
    num_keywords = 50  # Extract more keywords

    extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=window_size,
        top=num_keywords,
    )

    keywords = extractor.extract_keywords(text)
    # Convert from (keyword, score) tuples to just keywords
    extracted_keywords = [kw for kw, _ in keywords]

    return extracted_keywords


def extract_keywords_tfidf(text, top_n=20):
    """Extract keywords using TF-IDF."""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Use TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.001, ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Rank keywords by TF-IDF score
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = list(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

    # Return top N keywords
    return [kw for kw, score in sorted_keywords[:top_n]]


def extract_named_entities(text):
    """Extract named entities using NLTK's chunk parser."""

    # Tokenize and tag text
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Parse named entities
    tree = ne_chunk(tagged)

    # Extract entities
    entities = []
    current_entity = []
    current_label = None

    for node in tree:
        if isinstance(node, nltk.Tree):
            if current_label != node.label():
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity = []
                current_label = node.label()
            current_entity.extend([word for word, tag in node.leaves()])
        else:
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity = []
                current_label = None

    # Filter and categorize entities
    entity_types = {"PERSON", "ORGANIZATION", "GPE", "LOCATION", "DATE", "TIME"}

    filtered_entities = [
        (ent, label) for ent, label in entities if label in entity_types
    ]

    return filtered_entities


def caption_image(url):
    try:
        # Download the image
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        # Use GPT-4 Vision (example)
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail:"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
