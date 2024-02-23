# 查經小幫手：TensortRT-LLM 在 Windows 上的應用
## BSA: An LLM-basde assistant on Windows for Bible study 

本專案使用 [NVIDIA TensorRT-LLM](https://developer.nvidia.com/tensorrt#inference) 作為本地端的大語言模型引擎，來打造聖經研讀的小幫手。實作上，我們將 TensorRT-LLM 結合 Llama-2 ([Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)) 部屬在 Windows 11 環境下，使用 [LlamaIndex](https://www.llamaindex.ai/) 作為應用開發的架構，經由 [FastAPI](https://fastapi.tiangolo.com/) 讓使用者可以透過瀏覽器使用。

本應用會依據使用者對於經文釋義、或章節內容的提問，透過 Retrieval Augmented Generation 在聖經文本中尋找相關的章節，提供經文的導讀、釋義、並針對提問給予摘要總結。最後再以LLM生成禱告文做為總結。這個應用可以做為個人或小組查經的輔助工具，做為慕道友或者是基督教友快速理解聖經意涵的途徑。

## Installation

本應用需要先安裝 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/)，

### Pre-requisites
1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) for Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1/windows#quick-start).


### This App
1. Clone this repo to your 

