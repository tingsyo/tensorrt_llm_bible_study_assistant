# -*- coding: utf-8 -*-
import uvicorn
import openai
import pandas as pd
import logging
import argparse
from operator import itemgetter
from typing import List, Annotated

import faiss
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import load_index_from_storage, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.nvidia_tensorrt import LocalTensorRTLLM
from llama_index.legacy.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Request, Response
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2024~, Akira Dialog Technology"
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2024-02-20'

'''
This app serve as a Bible Study Assistant through FastAPI. We used 
TensorRT-LLM as a local replacement of OpenAI LLM. Also, a locally 
installed BAAI/BGE-M3 is used as the embedding model to create the 
chapter-based vector-store of the Bible.
'''
#-----------------------------------------------------------------
# Define global parameters
#-----------------------------------------------------------------
BIB_VECTOR_STORE_PATH = "./storage"
BIB_TEXT_PATH = "./data/the_holy_bible_zhtw_chapters.csv"

#-----------------------------------------------------------------
# Define BSA
#-----------------------------------------------------------------
def invoke_RAG_chain(question):
    ''' Invoke the RAG chain. '''
    # Load vectorstore
    vector_store = FaissVectorStore.from_persist_dir(BIB_VECTOR_STORE_PATH)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=BIB_VECTOR_STORE_PATH
    )
    index = load_index_from_storage(storage_context=storage_context)
    # Define prompts
    prompt = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Please also write the answer in English.\n"
        "Query: {query_str}\n"
        "ANSWER: "
    )
    text_qa_template = PromptTemplate(prompt)
    # Define answer
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    # Query index
    response = index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize",
                text_qa_template=text_qa_template,
                llm=llm06,
            ).query(question)
    logging.debug('Answer:\t'+response.response)
    logging.info('Sources:\t'+str([node.node.metadata['source_zhtw'] for node in response.source_nodes]))
    # Done
    return(response)

def select_bible_chapters(res_rag):
    ''' Select chapters of the Bible according to RAG results. '''
    bib_chapters = pd.read_csv(BIB_TEXT_PATH, encoding='utf-8')
    selected_chapters = []
    # Get cited sources
    for node in res_rag.source_nodes:
        src = node.node.metadata['source']
        selected_chapters.append(bib_chapters.loc[bib_chapters['source']==src,['source_zhtw','content']])
        #logging.info(doc.metadata['source_zhtw'])
    selected_chapters = pd.concat(selected_chapters).reset_index(drop=True)
    return(selected_chapters)

def elaborate_bible_chapter(statement, selected_chapter):
    ''' Use LLM to generate an interpretation of the Bible chapters to support the statement. '''
    index = VectorStoreIndex.from_documents([Document(text=selected_chapter['content'])])
    # Define prompt
    text_qa_template_str = (
        "You're a helpful AI assistant. Given the \n\n{query_str}\n\n \
        and the \n\n{context_str}\n\n, elaborate on the context \
        to support the {query_str} in English. The length is \
        about 500 words. Here is the context:{context_str}.\n\n \
        Translate the answer to English."
    )
    text_qa_template = PromptTemplate(text_qa_template_str, context_str=selected_chapter['content'])
    # Elaborate each selected chapter
    response = index.as_query_engine(
                    response_mode="tree_summarize",
                    text_qa_template=text_qa_template,
                    llm=llm06,
                ).query(statement)
    #
    return(response.response)


def elaborate_bible_chapters(statement, selected_chapters):
    ''' Use LLM to generate an interpretation of the Bible chapters to support the statement. '''
    elaborations = []
    for i in range(selected_chapters.shape[0]):
        chapter = selected_chapters.iloc[i,:].to_dict()
        tmp = elaborate_bible_chapter(statement, chapter)
        elaborations.append(tmp)
        logging.debug(chapter['source_zhtw']+':\t'+tmp)
    return(elaborations)


def generate_prayer(context, statement):
    ''' Use LLM to generate a Christian prayer that echoes the statement. '''
    # Define prompt
    text_qa_template_str = (
        "You're a helpful AI assistant. Given a \n\n{query_str}\n\n and \
        the context, write a formal Christian prayer in \
        English. The length is about 200 words. Here is the context:\
        \n\n{context_str}\n\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str, context_str=context)
    # Invoke the prayer chain
    index = VectorStoreIndex.from_documents([Document(text=context)])
    response = index.as_query_engine(
                response_mode="tree_summarize",
                text_qa_template=text_qa_template,
                llm=llm06,
            ).query(statement)
    return(str(response))

def run_BSA(question):
    ''' The complete procedure of the Bible Study Assistant. '''
    # Run through the process
    res_rag = invoke_RAG_chain(question)
    selected_chapters = select_bible_chapters(res_rag)
    elaborations = elaborate_bible_chapters(res_rag.response, selected_chapters)
    summary = summarize_elaborations(elaborations)
    prayer = generate_prayer(selected_chapters, summary)
    # Aggregate the results
    results = {
        'question': question,
        'answer': res_rag.response,
        'related_chapters':[node.node.metadata['source_zhtw'] for node in res_rag.source_nodes],
        'cited_chapter': selected_chapters.iloc[0,:].to_dict()['content'],
        'elaborations': elaborations,
        'prayer':prayer,
    }
    logging.debug(results)
    # Done
    return(results)

def beautify_results(results, format='md'):
    ''' Format the results for display '''
    output=("## 您的提問：\n>> " + results['question']+"\n\n")
    output+=("## 聖經查詢的結果：\n")
    output+=("### 摘要：\n>> "+results['answer'].replace('\n','\n>> ')+"\n\n")
    output+=("### 聖經中相關的章節：\n")
    for chapter in results['related_chapters']:
        output+=("- "+chapter.replace('\n','\n>> ')+"\n")
    output+=("\n### 讓我們一起來閱讀《"+results['related_chapters'][0]+"》：\n")
    output+=(">> "+results['cited_chapter'].replace('\n','\n> ')+"\n\n")
    output+=(">> "+results['elaborations'][0].replace('\n','\n> ')+"\n\n")
    output+=("### 其它章節裡的說法：\n")
    for text in results['elaborations'][1:]:
        output+=(">> "+text.replace('\n','\n> ')+"\n\n")
    #output+=("### 小結：\n")
    #output+=(">> "+results['summary'].replace('\n','\n> ')+"\n\n")
    output+=("## 最後，讓我們一起來禱告：\n")
    output+=(">> "+results['prayer'].replace('\n','\n> ')+"\n\n")
    return(output)

#-----------------------------------------------------------------
# Define FastAPI app
#-----------------------------------------------------------------
app = FastAPI()
#logger = logging.getLogger(__name__)

# locate templates
templates = Jinja2Templates(directory="templates")

chat_log = []

@app.get("/", response_class=HTMLResponse)
async def bsa_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "chat_log": chat_log})


@app.post("/", response_class=HTMLResponse)
async def bsa_run(request: Request, user_input: Annotated[str, Form()]):

    #chat_log.append({'role': 'user', 'content': user_input})
    logging.info('[QUESTION]'+user_input)

    #response = run_BSA(user_input)
    res_rag = invoke_RAG_chain(user_input)
    #chat_log.append(res_rag.response)
    #chat_log.append('\n\n'+str([node.node.metadata['source_zhtw'] for node in res_rag.source_nodes]))
    selected_chapters = select_bible_chapters(res_rag)
    #chat_log.append(selected_chapters.iloc[0,:].to_dict()['content'])
    elaborations = elaborate_bible_chapters(res_rag.response, selected_chapters)
    #chat_log.append(str(elaborations))
    prayer = generate_prayer('\n\n'.join(list(selected_chapters['content'])), user_input)
    #chat_log.append(prayer)
    results = {
        'question': user_input,
        'answer': res_rag.response,
        'related_chapters':[node.node.metadata['source_zhtw'] for node in res_rag.source_nodes],
        'cited_chapter': selected_chapters.iloc[0,:].to_dict()['content'],
        'elaborations': elaborations,
        #'summary':summary,
        'prayer':prayer,
    }
    response = beautify_results(results)
    chat_log.append(response)

    return templates.TemplateResponse("home.html", {"request": request, "chat_log": chat_log})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create an argument parser
    parser = argparse.ArgumentParser(description='TensorRT-LLM Parameters')

    # Add arguments
    parser.add_argument('--trt_engine_path', type=str, required=False,
                        help="Path to the TensorRT engine.", default="./model/")
    parser.add_argument('--trt_engine_name', type=str, required=False,
                        help="Name of the TensorRT engine.", default="llama_float16_tp1_rank0.engine")
    parser.add_argument('--tokenizer_dir_path', type=str, required=False,
                        help="Directory path for the tokenizer.", default="./model/")
    parser.add_argument('--verbose', type=bool, required=False,
                        help="Enable verbose logging.", default=False)
    # Parse the arguments
    args = parser.parse_args()
    #
    logging.basicConfig(level=logging.DEBUG)
    # Use the provided arguments
    trt_engine_path = args.trt_engine_path
    trt_engine_name = args.trt_engine_name
    tokenizer_dir_path = args.tokenizer_dir_path
    verbose = args.verbose

    # create trt_llm engine object
    llm06 = LocalTensorRTLLM(
        model_path=trt_engine_path,
        engine_name=trt_engine_name,
        tokenizer_dir=tokenizer_dir_path,
        temperature=0.6,
        max_new_tokens=1024,
        context_window=3900,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False
    )

    Settings.llm = llm06
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    #run server
    uvicorn.run(app)