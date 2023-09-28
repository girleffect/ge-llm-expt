from flask import Flask, request, jsonify
testapp = Flask(__name__)
import json
import csv
import matplotlib.pyplot as plt
import os
from langchain.llms import OpenAI
import ast
import openai
import pandas as pd
import tiktoken
from scipy import spatial
os.environ["OPENAI_API_KEY"] = "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"

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
# GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-4"

# download pre-chunked text and pre-computed embeddings
embeddings_path = "bigsiscontent_embeddings.csv"

embedding_df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type

embedding_df['embedding'] = embedding_df['embedding'].apply(ast.literal_eval)

# llm initialization
llm = OpenAI(openai_api_key="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")

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, embedding_df)
    introduction = 'Use the attached content sets on topics related to sexual and reproductive health to answer the subsequent question. If the answer cannot be found in the content sets, write "I could not find an answer." DO NOT RESPOND WITH ANYTHING OUTSIDE OF THE CONTENT SETS.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_set = f'\n\nContent set:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_set + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_set
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = embedding_df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about topics related to sexual and reproductive health. Use a casual tone similar to the content sets that young people can relate to. Use relevant emojis. An example response might sound like the following examples: 1. '🙌 Glad ur asking about safe sex. There's so much info out there it can be confusing sometimes. I'm here to help give u the accurate information 🤓so u can make ur own choice on what might be right for u if the time comes 👌' or 2. 'The 📖 penis is a male sex organ. It releases sperm during ejaculation. (It's also used for peeing but don't worry, it doesn't happen at the same time! 😂) It's attached to the scrotum, (also called balls). This houses the testicles which are full of sperm & hormones'. Make sure you do only respond with 'I could not find an answer' if the content is not available in the content sets provided to you."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

@testapp.route('/myfunction', methods=['POST'])
def my_function():
    data = request.get_json(force=True)
    input_text = data['input']
    output = ask(input_text)  # replace with your function
    return jsonify({'output': output})

if __name__ == '__main__':
    testapp.run(debug=True)