import requests
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="o1-mini")


def fetch_with_timeout(company_url, backend_api, timeout=180):
    """Mimic the JavaScript fetchWithTimeout API call."""
    # Construct the API URL
    api_url = f"{backend_api}/icp?company_url={requests.utils.quote(company_url)}"

    # Set the headers
    headers = {"Accept": "application/json"}

    # Make the API call with a timeout
    try:
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the JSON response

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


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


if __name__ == "__main__":
    dir = "questions"
    backend_api = os.environ.get("VITE_BACKEND_API")
    company_url = "https://www.revola.ai/"
    company_name = "revola"
    icp_filename = os.path.join(dir, f"{company_name}_icp_data.json")
    questions_filename = os.path.join(dir, f"{company_name}_questions.json")

    if not os.path.exists(icp_filename):
        company_data = fetch_with_timeout(company_url, backend_api)
        if company_data:
            print("API Response:", company_data)
        else:
            print("Failed to fetch data from the API.")
        with open(icp_filename, "w") as f:
            json.dump(company_data, f, indent=2)
    # questions = generate_questions(company_name, icp_filename)
    # # print(questions)
    # # Save the questions to a JSON file
    # with open(questions_filename, "w") as f:
    #     json.dump(questions, f, indent=2)
    # print(f"Questions saved to {questions_filename}")
