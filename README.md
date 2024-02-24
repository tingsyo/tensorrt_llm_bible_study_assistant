# 查經小幫手：TensortRT-LLM 在 Windows 上的應用
## BSA: An LLM-basde assistant on Windows for Bible study 

本專案使用 [NVIDIA TensorRT-LLM](https://developer.nvidia.com/tensorrt#inference) 作為本地端的大語言模型引擎，來打造聖經研讀的小幫手。實作上，我們將 TensorRT-LLM 結合 Llama-2 ([Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)) 部屬在 Windows 11 環境下，使用 [LlamaIndex](https://www.llamaindex.ai/) 作為應用開發的架構，經由 [FastAPI](https://fastapi.tiangolo.com/) 讓使用者可以透過瀏覽器使用。

本應用會依據使用者對於經文釋義、或章節內容的提問，透過 Retrieval Augmented Generation 在聖經文本中尋找相關的章節，提供經文的導讀、釋義、並針對提問給予摘要總結。最後再以LLM生成禱告文做為總結。這個應用可以做為個人或小組查經的輔助工具，做為慕道友或者是基督教友快速理解聖經意涵的途徑。

由於語言模型的限制，目前提供的是英文釋義。


> [!IMPORTANT]  
> This branch is developed and tested with TRT-LLM release v0.7.1.

## Getting Started

1. 安裝 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) for Windows，可以參考[這裡](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/windows/README.md)的安裝說明。

2. 確認可以使用 [Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) 上的 Llama 2

3. 本專案使用的是自行製作的 TRT Engine，如果開發環境與以下相符，可以直接在[ Google Drive ](https://drive.google.com/drive/folders/16wE7Fz9U-cp6LuWenKvETAJJaa9iPVCT?usp=sharing)下載:
    - LLaMa 2 13B AWQ 4bit quantized model (v1.4)
    - TensorRT 9.2.0
    - GeForce RTX 4090
    - TensorRT-LLM 0.7.1


### Setup this App

1. Clone this repository: 
```
git clone https://github.com/tingsyo/tensorrt_llm_bible_study_assistant.git
```
2. Place the TensorRT engine for LLaMa 2 13B model in the model/ directory
- For GeForce RTX 4090 users: Download the pre-built TRT engine [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2) and place it in the model/ directory.
- For other NVIDIA GPU users: Build the TRT engine by following the instructions provided [here](#building-trt-engine).
3. Acquire the llama tokenizer (tokenizer.json, tokenizer.model and tokenizer_config.json) [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main).
4. Download AWQ weights for building the TensorRT engine model.pt [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2). (For RTX 4090, use the pregenerated engine provided earlier.)
5. Install the necessary libraries: 
```
pip install -r requirements.txt
```
6. Launch the application using the following command:
```
python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --data_dir <Data folder>

```

