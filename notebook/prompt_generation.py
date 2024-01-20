import OpenAI
import json
import os
import sys
import warnings
import langchain
import weaviate

class automatic_prompt_generator(object):

    def __init__(self, openai_api_key: str, vectordb_api_key: str, vectordb_url: str, vectordb_model: str):
        self.openai_api_key = openai_api_key
        self.vectordb_api_key = vectordb_api_key
        self.vectordb_url = vectordb_url
        self.vectordb_model = vectordb_model
        self.openai = OpenAI(api_key=self.openai_api_key)
        self.weaviate = weaviate.Client(url=self.vectordb_url, api_key=self.vectordb_api_key)
        self.langchain = langchain.LangChain(self.weaviate, self.vectordb_model)

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=ImportWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    def generate_prompt(self, context: str) -> str:
        """Return the prompt to be completed.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        prompt = self.langchain.generate_prompt(context)
        return prompt
    
    def generate_completion(self, prompt: str, user_message: str, context: str) -> str:
        """Return the classification of the hallucination.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        API_RESPONSE = self.openai.Completion.create(
            prompt=prompt.replace("{Context}", context).replace("{Question}", user_message),
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        system_msg = str(API_RESPONSE.choices[0].text)
        print(system_msg)
        return system_msg
    
    def evaluate(self, prompt: str, user_message: str, context: str) -> str:
        """Return the classification of the hallucination.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        API_RESPONSE = self.openai.Completion.create(
            prompt=prompt.replace("{Context}", context).replace("{Question}", user_message),
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        system_msg = str(API_RESPONSE.choices[0].text)
        print(system_msg)
        return system_msg
    
    def generate_completion_vectordb(self, prompt: str, user_message: str, context: str) -> str:
        """Return the classification of the hallucination.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        API_RESPONSE = self.openai.Completion.create(
            prompt=prompt.replace("{Context}", context).replace("{Question}", user_message),
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        system_msg = str(API_RESPONSE.choices[0].text)
        print(system_msg)
        return system_msg
    
    def evaluate_vectordb(self, prompt: str, user_message: str, context: str) -> str:
        """Return the classification of the hallucination.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        API_RESPONSE = self.openai.Completion.create(
            prompt=prompt.replace("{Context}", context).replace("{Question}", user_message),
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        system_msg = str(API_RESPONSE.choices[0].text)
        print(system_msg)
        return system_msg
    
    def generate_completion_langchain(self, prompt: str, user_message: str, context: str) -> str:
        """Return the classification of the hallucination.
        @parameter prompt: the prompt to be completed.
        @parameter user_message: the user message to be classified.
        @parameter context: the context of the user message.
        @returns classification: the classification of the hallucination.
        """
        API_RESPONSE = self.openai.Completion.create(
            prompt=prompt.replace("{Context}", context).replace("{Question}", user_message),
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        system_msg = str(API_RESPONSE.choices[0].text)
        print(system_msg)
        return system_msg
    