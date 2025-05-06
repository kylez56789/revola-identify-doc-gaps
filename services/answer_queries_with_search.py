import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

company_name = "cirrascale"
question_filename = os.path.join("questions/", f"{company_name}_questions.json")
response_filename = os.path.join(
    "generated_context/", f"{company_name}_generated_context.json"
)

with open(question_filename, "r") as f:
    questions = json.load(f)

responses = []

for question in questions[3:4]:
    # print(question)
    response = client.chat.completions.create(
        model="gpt-4o-mini-search-preview-2025-03-11",
        web_search_options={},
        messages=[
            {
                "role": "user",
                "content": f"Answer the following question: {question}.",
            }
        ],
    )
    responses.append(
        {
            "question": question,
            "response": response.choices[0].message.content,
            # "annotations": response.choices[0].message.annotations,
        }
    )
print(responses[-1]["response"])

with open(response_filename, "w") as f:
    json.dump(responses, f, indent=2)

print(f"Responses saved to {response_filename}")
