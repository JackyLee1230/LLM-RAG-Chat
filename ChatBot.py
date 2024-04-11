# Conversational agent with RAG
#
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    # local ollama or lm studio or llamacpp server
    # can adapt to using langchain connector
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_ps",
                       embedding_function=embedding_function)

    chat_logs = [
        {"role": "system", "content": "REMEMBER YOU ARE MC Cheung Tin Fu, an artist under Warner Music Hong Kong. You are now talking to your fans that like your song. Please interact and chat."},
        {"role": "user", "content": "What was your major success?"},
    ]

    while True:
        completion = client.chat.completions.create(
            model="local-model",
            messages=chat_logs,
            temperature=0.3,  # higher temperature for more creative, lower for more accurate
            stream=True,  # stream the response or wait til whole respond
        )

        user_msg = {"role": "assistant", "content": ""}

        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                user_msg["content"] += chunk.choices[0].delta.content

        chat_logs.append(user_msg)

        print("\n")
        next_input = input("You > ")
        # use chroma to find top-2 most similar documents
        search_results = vector_db.similarity_search(next_input, k=2)
        rag_context = ""
        for result in search_results:
            rag_context += result.page_content + "\n\n"
        chat_logs.append({"role": "user", "content": rag_context + next_input})


if __name__ == "__main__":
    main()
