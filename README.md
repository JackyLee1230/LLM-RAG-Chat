<!-- write about this LLM RAG Chatbot -->

# LLM RAG Chatbot

This sample project make uses of LM Studio (Or Other Local LLM Server, including Ollama), RAG with ChromaDB and to create a conversational AI model that can generate responses to user queries.

The pretrained PDF is about MC Cheung Tin Fu, a famous Hong Kong musician under Warner Music Hong Kong Ltd.
The model is fine-tuned on the dataset of the actor's information. The model is able to answer questions about the actor's personal information, career, and other related information.

## Setup

1. Install ChromaDB and LM Studio (or other LLM server) on your local machine.
2. Run any LLM Model on LM Studio, preferably a Chat model.
3. Clone this repository.
4. Run the following command to install the required packages: </br>
   Running in a Conda environment is recommended.

```bash
pip install -r requirements.txt
```

5. (Optional)
   Change the name of PDF you want to train the RAG model on in the `vectorize.py` file.
   And change the system prompt in the `chatbot.py` file.

```bash
python vectorize.py
```

6. Run the following command to start the chatbot:

```bash
python chatbot.py
```

7. Ask questions about the actor's personal information, career, and other related information that can be found within the trained PDF document.

