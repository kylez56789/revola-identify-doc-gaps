import os
import json
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import chain
import pandas as pd

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Embedding and storing in FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
dim = 3072  # for OpenAI text-embedding-3-large

pc = Pinecone(api_key=PINECONE_API_KEY)

llm = ChatOpenAI(model="gpt-4o-mini")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:

        results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)

        # Handle the case where no results are returned
        if not results:
            return []

        docs, scores = zip(*results)
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs


def init_retriever():
    vector_db = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    metadata_field_info = [
        AttributeInfo(
            name="file_path",
            description="path to original document",
            type="string",
        ),
        AttributeInfo(
            name="page_number",
            description="page number of chunk from original pdf",
            type="integer",
        ),
        # AttributeInfo(
        #     name="toc",
        #     description="table of contents",
        #     type="string or list[string]",
        # ),
        AttributeInfo(
            name="images",
            description="paths to parsed images",
            type="string or list[string]",
        ),
        # AttributeInfo(
        #     name="tables",
        #     description="paths to parsed tables",
        #     type="string or list[string]",
        # ),
    ]
    document_content_description = "Company documentation for Revola AI"
    llm = OpenAI(temperature=0)
    retriever = CustomSelfQueryRetriever.from_llm(
        llm,
        vector_db,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )
    return retriever


def generate_questions(company_name, icp_filename):
    with open(icp_filename, "r") as f:
        company_data = json.load(f)
    # Concatenate all "pain_points" and "decision_makers" from "icp_table_1"
    business_summary = company_data["business_summary"]
    icp_data = [
        {
            "icp_name": icp_entry["icp_name"],
            "industry": icp_entry.get("industry", ""),
            "personas": icp_entry.get("decision_makers", [])
            + icp_entry.get("users", []),
            "pain_points": icp_entry.get("pain_points", []),
            "decision_makers": icp_entry.get("decision_makers", []),
            "users": icp_entry.get("users", []),
            "revenue_potential": icp_entry.get("revenue_potential", ""),
        }
        for icp_entry in company_data["icp_data"]["icp_summary_table"]["icp_table_1"]
    ]
    products = ", ".join(
        company_data["icp_data"]["metadata"]["product_list"].split(";")
    )
    use_cases = ", ".join(company_data["icp_data"]["metadata"]["use_cases"].split(";"))

    prompt = PromptTemplate(
        input_variables=[
            "business_summary",
            "icp_name",
            "pain_points",
            "products",
            "use_cases",
            "company_name",
            "persona",
        ],
        template="""
            You are a {persona} and work for a {icp_name} company and the main pain points for your company are {pain_points}.
            You have found a company called {company_name}. The context for this company is {business_summary}.
            The company offers these products: {products}.
            The use cases for these products are: {use_cases}.
            You would like to know if this company and their products will be the right solution for your company by asking important questions.
            You will generate questions by following these steps:
            1. Determine which products are the best fit as a solution for your role.
            2. Determine the use cases for those products.
            3. Generate deep, multihop questions to ask regarding this product and the uses cases across many categories.  
            Example categories are: Product Functionality, Pricing & Cost Structure, Onboarding & Implementation, Integration & Compatibility, Support & Customer Success, Trial & Proof of Concept, Security & Compliance, Scalability & Performance, Location & Availability, Customization & Flexibility, Analytics & Reporting, Updates & Roadmap, User Experience & Accessibility, Company Stability & Reputation, Contract Terms & Exit Strategy.
            4. Return only the questions separated by a ";" and nothing else.
            """,
    )
    questions = []
    for icp in icp_data:
        for persona in icp["personas"]:
            message = HumanMessage(
                content=prompt.format(
                    business_summary=business_summary,
                    icp_name=icp["icp_name"],
                    pain_points=icp["pain_points"],
                    products=products,
                    use_cases=use_cases,
                    company_name=company_name,
                    persona=persona,
                )
            )
            response = llm.invoke([message]).content
            questions.extend(response.split(";"))
    return questions


def generate_answers(questions_filename, qa_filename):
    # retriever = init_retriever()
    index = pc.Index(PINECONE_INDEX_NAME)

    with open(questions_filename, "r") as f:
        questions = json.load(f)

    results = []
    for idx, query in enumerate(questions[:100]):
        print(f"{idx+1}. {query}")
        context, max_score = retrieve_context_and_scores(query)

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Answer the query using just the context provided. Be as specific as you can and use as much of the context as you can in your answer. If there is not enough context to determine an answer, respond with "Not enough context."
            Context: {context}
            Query: {query}
            """,
        )
        message = HumanMessage(content=prompt.format(query=query, context=context))
        answer = llm.invoke([message]).content.strip()
        results.append(
            {
                "question": query,
                "answer": answer,
                "context": context,
                "score": max_score,
            }
        )

    # Create a DataFrame from the questions_and_answers list
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    df.to_excel(qa_filename, index=False)
    print(f"Answers saved to {qa_filename}")

    return results


def retrieve_context_and_scores(query):
    chunk_ids = set()
    documents = []

    prompt = PromptTemplate(
        input_variables=["query"],
        template="""Rewrite the query by splitting it up into multiple queries if there are multiple parts to the question. Return the rewritten questions separated by just a ';'."
        Query: {query}
        """,
    )
    message = HumanMessage(content=prompt.format(query=query))
    answer = llm.invoke([message]).content.strip()
    split_questions = answer.split(";")
    split_questions.append(query)
    # print(split_questions)
    index = pc.Index(PINECONE_INDEX_NAME)

    for question in split_questions:
        embedding = embeddings.embed_query(question)

        # Search the index for the three most similar vectors
        retrieved_docs = index.query(
            namespace="",
            vector=embedding,
            top_k=5,
            include_values=False,
            include_metadata=True,
        )

        # print(retrieved_docs)
        for x in retrieved_docs["matches"]:
            if x["id"] not in chunk_ids:
                documents.append({"id": x["id"], "text": x["metadata"]["text"]})
                chunk_ids.add(x["id"])

    # Use the SentenceTransformer reranker model to rerank the documents
    reranker_inputs = [(query, doc["text"]) for doc in documents]
    reranker_scores = reranker.predict(reranker_inputs)
    # print(reranker_scores)

    # Combine documents with their scores and sort by score in descending order
    reranked_docs = sorted(
        zip(documents, reranker_scores), key=lambda x: x[1], reverse=True
    )
    top_docs = reranked_docs[:3]
    context = " ".join([doc["text"] for doc, _ in top_docs])
    scores = [score for _, score in top_docs]
    max_score = max(scores, default=0)
    return context, max_score


if __name__ == "__main__":
    company_name = "revola"
    dir = f"questions/{company_name}"
    icp_filename = os.path.join(dir, f"{company_name}_icp_data.json")
    questions_filename = os.path.join(dir, f"{company_name}_main_questions.json")
    qa_filename = os.path.join(dir, f"{company_name}_main_questions_and_answers.xlsx")
    # questions = generate_questions(company_name, icp_filename)
    # # print(questions)
    # # Save the questions to a JSON file
    # with open(questions_filename, "w") as f:
    #     json.dump(questions, f, indent=2)
    # print(f"Questions saved to {questions_filename}")
    generate_answers(questions_filename, qa_filename)

    # # retriever = init_retriever()
    # query = "What is the pricing structure for Reva, and are there any additional costs associated with scaling its use for a growing customer base?"
    # # result = retriever.invoke(query)

    # split_context = retrieve_context_and_scores(query)
    # print(split_context)
