import os
import random
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import caikit_tgis_langchain
import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'ChatBot')

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MODEL_ID = os.getenv('MODEL_ID')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
MIN_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 100))
TEMPLATE = os.getenv('TEMPLATE',"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift AI, aka RHOAI.
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Question: {question}
Context: {context} [/INST]
""")

DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')

STREAMING = False
TIMEOUT = 300

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = qa_chain({"query": input_text})
        sources = remove_source_duplicates(resp['source_documents'])
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

############################
# LLM chain implementation #
############################

# Document store: PGvector store
model_path = "./model/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_path)
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings)

# LLM
llm = caikit_tgis_langchain.CaikitLLM(
    inference_server_url=INFERENCE_SERVER_URL,
    model_id=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    min_new_tokens=MIN_NEW_TOKENS,
    certificate_chain="", #certificate.pem
    streaming=STREAMING,
    timeout=TIMEOUT,
    run_manager=QueueCallback(q)
)

QA_CHAIN_PROMPT = PromptTemplate.from_template(TEMPLATE)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "distance_threshold": 0.5}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
    )

# Gradio implementation
def ask_llm(message, history):
    if STREAMING:
        for next_token, content in stream(message):
            yield(content)
    else:
        response = qa_chain({"query": message})
        print("RESPONSE ", response)
        yield f"{response['result']}\nSources:{response['source_documents']}"

with gr.Blocks(title="HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None,'assets/robot-head.svg'),
        render=False,
        )
    gr.ChatInterface(
        fn=ask_llm,
        chatbot=chatbot,
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
        )

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico'
        )
