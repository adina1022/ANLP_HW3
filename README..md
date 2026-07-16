# Simple RAG System

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) system in Python using Ollama.

The program:

- loads a text knowledge base,
- creates embeddings for each chunk,
- stores the embeddings in an in-memory vector database,
- retrieves the most relevant chunks using cosine similarity,
- generates a grounded answer using the retrieved context.

## Repository Structure

hw3_q1_rag/
├── cat-facts.txt
├── cat-db-fiction-confusing.txt
├── demo.py
├── README.md


---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd hw3_q1_rag
```

### 2. Install the required Python package

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

The official Ollama documentation is available at:

https://ollama.com

For this assignment, I used an older compatible Ollama version because my machine is an Intel running macOS Ventura (13), while the latest official Ollama release requires macOS Sonoma (14) or newer.

The version used for this project was downloaded from:

https://ollama.de.uptodown.com/mac/dw/1111174286

### 4. Download the required models

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

---

## Running the Program

Run the program from the project directory:

```bash
python3 demo.py
```

The program will first prompt you to select the knowledge base:

```text
Choose a knowledge base:
1 - cat-facts.txt
2 - cat-db-fiction-confusing.txt

Enter 1 or 2:
```

After selecting a knowledge base, the program will:

1. Load the selected knowledge base.
2. Build the in-memory vector database.
3. Ask the user for a question.
4. Retrieve the three most relevant chunks.
5. Display the retrieved chunks together with their similarity scores.
6. Generate and stream a grounded answer using the retrieved context.

To reproduce the evaluation:

- Select **1** to use `cat-facts.txt`.
- Select **2** to use `cat-db-fiction-confusing.txt`.

## Evaluation

| File used                      | Question no. | `top_n` | Generated answer | Correct? | If incorrect, how could it be fixed?
| `cat-facts.txt`                |            1 |       3 | The Jacobson's organ, also known as the vomeronasal organ, is located on the upper surface of the mouth and allows cats to detect pheromones. | Yes   |
| `cat-facts.txt`                |            2 |       3 | According to the context, a nine-year-old cat has been awake for approximately three years of its life. | Yes | -                                                                                                                                                       |
| `cat-db-fiction-confusing.txt` |            3 |       3 | Northside Luna wears a blue collar. She needs rabbit-based food because her record lists a chicken allergy. | No| Improve the prompt so the model keeps records with the same name separate and uses the location (Northside vs. Riverside) to identify the correct Luna. |
| `cat-db-fiction-confusing.txt` |            4 |       3 | Juniper cannot attend the adoption event because a veterinary note dated 2026-05-01 requires seven days of rest. | Yes | -                                      
