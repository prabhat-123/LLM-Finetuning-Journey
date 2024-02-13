import os
import openai
from dotenv import load_dotenv


class ReceipeGenerator:

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = "gpt-3.5-turbo"
        self.role = "user"

    def generate_receipe(self, query, receipe_context):
        prompt = """Based on the provided user query on food receipies and the context 
retreived, generate a receipe that best fits the users query"""
        user_query = """User Query: \n{}""".format(query)
        retrieved_context = """\nRetrieved Context: \n{}\n""".format(receipe_context)
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                "role": self.role,
                "content": user_query + retrieved_context + prompt,
            }
            ],
            max_tokens=512,
            temperature=0.01,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response = completion["choices"][0]["message"]["content"]
        return response
