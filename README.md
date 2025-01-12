# DSPy docs RAG 

## Overview
This repository serves as a baseline for [Implementing RAG with DSPy: A Technical Guide](https://medium.com/@arancibia.juan22/implementing-rag-with-dspy-a-technical-guide-a6ae15f6a455) available on Medium. The main component is the Jupyter Notebook file, dspy_docs_rag.ipynb, which illustrates the process of loading, splitting, and embedding Markdown documents using the LangChain library within a Retrieval-Augmented Generation (RAG) framework.

## Main Notebook Contents
The [notebook file](./dspy_docs_rag.ipynb) includes the following key sections:
- Loading Markdown Files: Demonstrates how to load Markdown documents from a specified directory using LangChain's DirectoryLoader.
- Document Splitting: Shows the implementation of a text splitter that divides the loaded documents into smaller, manageable chunks for easier processing.
- Generating Document Embeddings: Explains how to create embeddings for the split documents using OpenAI's embedding models.
- Storing Embeddings: Illustrates how to store the generated embeddings in a persistent database using Chroma.
- DSPy RAG: Discusses various approaches to implement Retrieval-Augmented Generation using DSPy.
- Evaluation: Provides methods to evaluate the performance and effectiveness of the RAG implementation.
- Optimization: Covers MIPROv2 strategy for optimizing the RAG process to enhance efficiency and accuracy.

This notebook provides practical examples and code snippets that guide users through each step of the document processing workflow, making it a valuable resource for those looking to implement RAG techniques with DSPy.
