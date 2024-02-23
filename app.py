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
        "Please also write the answer in Traditional Chinese.\n"
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
                llm=llm01,
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
    index = VectorStoreIndex.from_documents(Document(text=selected_chapter['content']))
    # Define prompt
    text_qa_template_str = (
        "You're a helpful AI assistant. Given the \n\n{query_str}\n\n \
        and the \n\n{context_str}\n\n, elaborate on the context \
        to support the {query_str} in Traditional Chinese. The length is \
        about 500 words. Here is the context:{context_str}."
    )
    text_qa_template = PromptTemplate(text_qa_template_str)
    #print(text_qa_template)
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
        the context, write a formal Christian prayer in Traditional \
        Chinese. The length is about 200 words. Here is the context:\
        \n\n{context_str}\n\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str, context_str=context)
    # Invoke the prayer chain
    response = index.as_query_engine(
                response_mode="tree_summarize",
                text_qa_template=text_qa_template,
                llm=llm12,
            ).query("#zh_tw"+statement)
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
    logging.info(user_input)

    #response = run_BSA(user_input)
    #response = {'question': '為什麼神在舊約中如此憤怒，在新約中卻充滿慈愛？', 'answer': '根據聖經中的描述，神在舊約中顯示了 祂的公義和憤怒，因為人們離棄了祂的命令，拜偶像，違背約束。這導致了神的憤怒和審判。然而，在新約中，神透過基督耶穌的降臨，彰顯了祂的慈愛和恩典，為要拯救世人，使他們得到永生。基督的教導強調愛、寬恕、憐憫和和平，讓人們可以通過信仰和悔改得到救贖。因此，神在新約中展現了更多的慈愛和寬恕，讓人們有機會與祂建立親密的關係，並得到永生的盼望。', 'related_chapters': ['耶 利米哀歌：第1章', '申命記：第29章', '約伯記：第23章', '羅馬書：第15章'], 'cited_chapter': '他夜間痛哭、淚流滿腮．在一切 所親愛的中間、沒有一個安慰他的．他的朋友、都以詭詐待他、成為他的仇敵。\n猶大因遭遇苦難、又因多服勞苦、就遷到外邦．他住在列國中、尋不著安息．追逼他的都在狹窄之地將他追上。\n錫安的路徑、因無人來守聖節就悲傷．他的城門淒涼、他的祭司歎息．他的處女受艱難、自己也愁苦。\n他的敵人為首．他的仇敵亨通．因耶和華為他許多的罪過、使他受苦．他的孩童被敵人擄去。\n錫安城的威榮〔城原文作女子下同〕、全都失去．他的首領、像找不著草場的鹿、在追趕的人前、無力行走。\n耶路撒冷在困苦窘迫之時、就追想古時一切的樂境。他百姓落在敵人手中、無人救濟．敵人看見、就因他的荒涼嗤笑。\n耶路撒冷大大犯罪．所以成為不潔之物．素來尊敬他的、見他赤露就都藐視他．他自己也歎息退後。\n他的污穢是在衣襟上．他不思想自己的結局．所以非常的敗落、無人安慰他。他說、耶和華阿、求你看我的苦難、因為仇敵誇大。\n敵人伸手、奪取他的美物．他眼見外邦人進入他的聖所．論這外邦人你曾吩咐不可入你的會中。\n他的民都歎息、尋求食物．他們用美物換糧食、要救性命。他們說、耶和華阿、求你觀看、因為我甚是卑賤。\n你們一切過路的人哪、這事你們不介意麼．你們要觀看、有像這臨到我的痛苦沒有、就是耶和華在他發烈怒的日子使我所受的苦。\n他從高天使火進入我的骨頭、剋制了我．他鋪下網羅、絆我的腳、使我轉回．他使我終日淒涼發昏。\n我罪過的軛、是他手所綁的、猶如軛繩縛在我頸項上．他使我的力量衰敗。主將我交在我所不能敵擋的人手中。\n主輕棄我中間的一切勇士、招聚多人〔原文作大會〕攻擊我、要壓碎我的少年人．主將猶大居民踹下、像在酒醡中一樣。\n我因這些事哭泣．我眼淚汪汪．因為那當安慰我、救我性命的、離我甚遠．我的兒女孤苦、因為仇敵得了勝。\n錫安舉手、無人安慰．耶和華論雅各已經出令、使四圍的人作他仇敵．耶路撒冷在他們中間、像不潔之物。\n耶和華是公義的．他這樣待我、是因我違背他的命令。眾民哪、請聽我的話、看我的痛苦、我的處女和少年人、都被擄去。\n我招呼我所親愛的、他們卻愚弄我．我的祭司和長老、正尋求食物救性命的時候、就在城中絕氣。\n耶和華阿、求你觀看、因為我在急難中．我心腸擾亂．我心在我裡面翻轉．因我大大悖逆．在外刀劍使人喪子、在家猶如死亡。\n聽見我歎息的有人．安慰我的卻無人．我的仇敵、都聽見我所遭的患難．因你作這事他們都喜樂。你必使你報告的日子來到、他們就像我一樣。\n願他們的惡行、都呈在你面前．你怎樣因我的一切罪過待我、求你照樣待他們．因我歎息甚多、心中發昏。', 'elaborations': ['在舊約中，聖經中的描述揭示了神對以色列人的懲罰和審判， 因為他們違背神的誡命，拜偶像，背叛神的約束。這導致了神的憤怒和悲慘的結局。在《耶利米哀歌》中，記載了耶路撒冷被圍困時的絕望景象，城中的人民飢餓、遭敵人侵略，祭司和領袖無助，孩童被擄走，一切都充滿苦難和哀傷。\n\n然而，新約中帶來了轉變和希望。神透過差遣祂的獨生子─耶穌基督，向世人彰顯了祂的慈愛和恩典。基督的降臨、生平、教導、受死和復活，展現了神對人類的無比愛和 憐憫。基督的教導強調愛、寬恕、憐憫和和平，呼籲人們悔改，相信福音，得以蒙恩、得救。\n\n透過基督的救贖工作，神向人類顯示了更廣闊的慈愛和寬恕。神不再僅僅以憤怒和審判示人，而是透過基督的犧牲，向世人彰顯了無盡的慈愛和恩典。這是一個全新的約，一個建立在基督的救贖之上，讓每個相信的人都可以透過信仰與神建立親密的關係，並且獲得永生的盼望。\n\n在新約中，神的慈愛和寬恕得到了更深的彰顯，祂願意寬恕並拯救一切悔改的人。這是神對人類的大恩大慈，讓我們有機會與祂親近，經歷祂豐盛的恩典和慈愛。這段新的關係基於信心和悔改，使我們可以在基督裡找到真正的盼望和平安，並且得以享受永生的應許。\n\n因此，從舊約到新約，神的形象在聖經中得到了完整的展現。祂是公義的，也是滿有慈愛和恩典的神。透過基督的救贖，神向我們顯示了祂的大愛和寬恕，讓我們可以與祂建立親密的關係，並且擁有永恆的盼望。', '根據《申命記》中的描述，神向以色列人顯示了祂的公義和憤怒，因為人們離棄了祂的命令，拜偶像，違背約束。這導致了神的憤怒和審判，並警告人們必須忠心順從祂的律法。然而，在新約中，神透過差遣基督耶穌來彰顯了祂的慈愛和恩典，為要拯救世人，使他們得到永生。\n\n基督的降臨代表了神對人類的慈愛和寬恕，使人們可以透過基督的教導和犧牲來認識神的愛。新約中強調了愛、寬恕、憐憫和和平的價值觀，教導人們如何與神建立親密的關係，並通過信仰和悔改得到救贖。基督的教導和生平展現了神對人的深愛和寬恕，為人類帶來了永生的盼望。\n\n因此，從舊約到新約，神的形象在人類心目中的呈現有所轉變。舊約中彰顯了神的公義和憤怒，對人的背叛和罪惡進行審判；而新約中則更多地著重於神的慈愛和寬恕，透過基督的教導和犧牲來拯救世人，讓人們得以與神建立親密的關係，並享受永生的盼望。這轉變展現了神對人類的愛和關懷，顯示了祂願意接納和拯救所有悔改的人，讓他們在神的慈愛中找到安慰和盼望。', '根據《約伯記》中的描述，約伯在面對困境和痛苦時，表達了對神的渴望和尋求。他希望能親自站在神的面前，陳明自己的冤屈，並與神直接對話。約伯強調他對神的信任和遵循，即使在受苦的時候也不放棄尋求神的旨意。這段經文中展現了人類對神的信賴和尋求公義的渴望。\n\n在舊約中，神確實展現了對人類的公義和憤怒，因為人們常常違背祂的旨意，拜偶像，犯罪違法。這導致了神的審判和懲罰，顯示了神對罪惡的憤怒。然而，隨著新約的來臨，神透過基督耶穌的降臨展現了更多的慈愛和恩典。\n\n基督耶穌的降臨代表了神對人類的愛和拯救。神透過基督的教導彰顯了愛、寬恕、憐憫和和平的價值。基督的教導強調了悔改和信仰的重要性，這是人們得以與神建立親密關係並得到永生的途徑。基督的教導教導人們如何去愛彼此，寬恕彼此，並接受神的寬恕和恩典。\n\n因此，從舊約到新約，我們看到了神對人類的愛和憐憫的演變。儘管在舊約中神顯示了祂的公義和憤怒，但在新約中，神透過基督的降臨展現了更多的慈愛和寬恕。這使得人們有機會通過信仰和悔改來與神建立親密的關係，並得到永生的盼望。這是聖經中神的愛和救贖計劃的一部分，為人類帶來了希望和救恩的信息。', '根據聖經中的描述，確實可以看到在舊約中，神向以色列人顯示了祂的公義和憤怒，因為人們時常離棄祂的命令，拜偶像，違背約束，並犯下種種的罪惡。這些行為觸怒了神的公義，導致祂的憤怒和審判臨到他們身上。舊約中記載了許多關於神的懲罰和審判的故事，以提醒人們要尊敬和順從神的旨意，遵守祂的誡命。\n\n然而，在新約中，神透過基督耶穌的降臨，向世人彰顯了祂的慈愛和恩典。基督的降臨代表著神對人類的愛和拯救計劃的實現。基督的教導強調了愛、寬恕、憐憫和和平，並呼籲人們悔改和信靠神，以得到救贖和永生的盼望。基督的生平和教導展現了神對人類的慈愛和寬恕，使人們可以通過信仰和悔改建立親密的關係，並享受永恆的生命。\n\n因此，從舊約到新約，神的形象從公義和憤怒轉變為慈愛和寬恕，這展現了神對人類的奇妙恩典和愛。透過基督的救贖工作，人們有機會與神建立更親密的關係，並且獲得得救的機會。新約中的信息強調了神的慈愛和寬恕，呼籲人們回到神的懷抱，並且藉著基督的救贖享受永生的盼望。這是一個從律法到恩典的轉變，展現了神對人類的愛和關懷，讓人們可以在神裡面找到安慰和盼望。'], 'summary': '舊約中描述了神對以色列人的懲罰和審判，因為他們違背神的誡命。然而，新約帶來了轉變和希望，透過基督的降臨展現了神的慈愛和恩典。基督的教導強調愛、寬恕、憐憫和和平，呼籲人們悔改，相信福音，得以蒙恩、得救。透過基督的救贖工作，神向人類顯示了更廣闊的慈愛和寬恕，讓每個相信的人都可以透過信仰與神建立親密的關係，並且獲得永生的盼望。從舊約到新約，神的形象從公義和憤怒轉變為慈愛和寬恕，展現了神對人類的奇妙恩典和愛。這是一個從律法到恩典的轉變，讓人們可以在神裡面找到安慰和盼望。', 'prayer': '慈愛的天父，感謝祢在不同時代向人類顯示祢奇妙的恩典和愛。在舊約中，我們看到了祢對以色列人的公義懲罰，但也見證了祢對他們的審判與憐憫。藉著基督的降臨和救贖工作，祢彰顯了祢更廣大的愛和寬恕，讓每個相信的人都能與祢建立親密關係，得以蒙恩、得救。基督的教導告訴我們應該彼此相愛、寬恕、憐憫，與和平相處，呼籲我們悔改，相信福音，接受祢的救恩和慈愛。\n\n天父啊，我們感謝祢的恩典和寬恕，以及讓我們透過基督找到盼望和安慰的慈愛。願我們在祢的愛中繼續成長，彼此寬恕，彼此憐憫，為彼此祈禱。幫助我們以祢為榜樣，彰顯祢的愛和寬恕，讓我們在這個世界上成為光和鹽，建立在基督裡的愛的團契。求祢富憐憫的手能伸展到每個人的生命中，讓更多人能明白祢的恩典，得著救恩，結出與基督相符的果子。奉主耶穌基督的名禱告，阿們。'}
    res_rag = invoke_RAG_chain(user_input)
    #chat_log.append(res_rag.response)
    #chat_log.append('\n\n'+str([node.node.metadata['source_zhtw'] for node in res_rag.source_nodes]))
    selected_chapters = select_bible_chapters(res_rag)
    #chat_log.append(selected_chapters.iloc[0,:].to_dict()['content'])
    elaborations = elaborate_bible_chapters(res_rag.response, selected_chapters)
    #chat_log.append(str(elaborations))
    prayer = generate_prayer(selected_chapters, user_input)
    #chat_log.append(prayer)
    results = {
        'question': user_input,
        'answer': res_rag.response,
        'related_chapters':[node.node.metadata['source_zhtw'] for node in res_rag.source_nodes],
        'cited_chapter': selected_chapters.iloc[0,:].to_dict()['content'],
        'elaborations': elaborations,
        'summary':summary,
        'prayer':prayer,
    }
    response = beautify_results(results)
    chat_log.append(esponse)

    return templates.TemplateResponse("home.html", {"request": request, "chat_log": chat_log})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create an argument parser
    parser = argparse.ArgumentParser(description='TensorRT-LLM Parameters')

    # Add arguments
    parser.add_argument('--trt_engine_path', type=str, required=True,
                        help="Path to the TensorRT engine.", default="")
    parser.add_argument('--trt_engine_name', type=str, required=True,
                        help="Name of the TensorRT engine.", default="")
    parser.add_argument('--tokenizer_dir_path', type=str, required=True,
                        help="Directory path for the tokenizer.", default="")
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
    llm01 = LocalTensorRTLLM(
        model_path=trt_engine_path,
        engine_name=trt_engine_name,
        tokenizer_dir=tokenizer_dir_path,
        temperature=0.1,
        max_new_tokens=1024,
        context_window=3900,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False
    )
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
    llm12 = LocalTensorRTLLM(
        model_path=trt_engine_path,
        engine_name=trt_engine_name,
        tokenizer_dir=tokenizer_dir_path,
        temperature=1.2,
        max_new_tokens=1024,
        context_window=3900,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False
    )
    Settings.llm = llm01
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    #run server
    uvicorn.run(app)