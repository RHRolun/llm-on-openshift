import os
import random
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

load_dotenv()

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'Talk with your documentation')

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_URL = os.getenv('S3_URL')
S3_BUCKET = os.getenv('S3_BUCKET')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

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

def ingest_data_s3(data_folder = "./documents"):
    s3 = boto3.resource(
        's3', endpoint_url=S3_URL,
        aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY
    )
    bucket = s3.Bucket(S3_BUCKET)

    download_count = 0
    for s3_object in bucket.objects.all():
        print(download_count)
        key = s3_object.key
        local_file_path = os.path.join(data_folder, os.path.basename(key))
        bucket.download_file(key, local_file_path)
        download_count += 1
    print("Finished ingestion")


# Document store: Faiss vector store
print("Loading database...")
pdf_folder_path = 'documents'
os.mkdir(pdf_folder_path)
ingest_data_s3(pdf_folder_path)
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 40)
all_splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(all_splits, embeddings)

# LLM
llm = HuggingFaceTextGenInference(
    inference_server_url=INFERENCE_SERVER_URL,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    typical_p=TYPICAL_P,
    temperature=TEMPERATURE,
    repetition_penalty=REPETITION_PENALTY,
    streaming=True,
    verbose=False,
    callbacks=[QueueCallback(q)]
)

# Prompt
# template="""<s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift Data Science, aka RHODS.
# You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
# Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>

# Question: {question}
# Context: {context} [/INST]
# """
template="""<s>[INST] <<SYS>>
Du är en hjälpsam, respektfull och ärlig assistant som svarar på frågor om TDOks, styrande och stödjande dokument för traffikverket.
Du kommer bli given en fråga som du måste svara, samt kontext med information. Du måste svara frågan baserat på kontexten så mycket som möjligt.
Svara alltid så hjälpsammt och säkert som möjligt. Ditt svar ska inte innehålla något skadligt, oetiskt, rasistiskt, sexistiskt, giftigt, farligt eller olagligt innehåll. Se till att dina svar är socialt opartiska och positiva till sin natur.

Om en fråga inte är meningsfull, eller inte är faktamässigt sammanhängande, förklara varför istället för att svara på något som inte är korrekt. Om du inte vet svaret på en fråga, vänligen dela inte falsk information.
<</SYS>>

Frågan: {question}
Kontexten: {context} [/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "distance_threshold": 0.5}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
    )

# Gradio implementation
def ask_llm(message, history):
    for next_token, content in stream(message):
        yield(content)

with gr.Blocks(title="HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None,'assets/robot-head.svg'),
        render=False
        )
    gr.ChatInterface(
        ask_llm,
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
