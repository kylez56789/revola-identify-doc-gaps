from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def evaluate_information_adequacy(keywords, extracted_text):
    """Evaluate if there's enough information about key topics."""
    all_text = " ".join(extracted_text.values())
    sentences = sent_tokenize(all_text)

    evaluation_results = {}

    for keyword in keywords:
        # Generate questions about this keyword
        questions = generate_questions_for_keyword(keyword)

        # Find relevant content
        relevant_content = find_relevant_content(keyword, sentences)

        # Check if questions can be answered
        answer_scores = evaluate_answers(questions, relevant_content)

        # Determine if information is adequate
        is_adequate = any(score > 0.5 for score in answer_scores)

        evaluation_results[keyword] = {
            "has_adequate_info": is_adequate,
            "relevant_content_count": len(relevant_content),
            "answer_scores": answer_scores,
        }

    # Calculate overall adequacy score
    total_keywords = len(keywords)
    keywords_with_adequate_info = sum(
        1 for k, v in evaluation_results.items() if v["has_adequate_info"]
    )
    adequacy_percentage = (
        (keywords_with_adequate_info / total_keywords) * 100
        if total_keywords > 0
        else 0
    )

    return {
        "keyword_evaluations": evaluation_results,
        "overall_adequacy_percentage": adequacy_percentage,
        "overall_rating": get_adequacy_rating(adequacy_percentage),
    }


def get_adequacy_rating(percentage):
    """Convert adequacy percentage to a rating."""
    if percentage >= 90:
        return "Excellent"
    elif percentage >= 75:
        return "Good"
    elif percentage >= 50:
        return "Adequate"
    elif percentage >= 30:
        return "Needs Improvement"
    else:
        return "Poor"
