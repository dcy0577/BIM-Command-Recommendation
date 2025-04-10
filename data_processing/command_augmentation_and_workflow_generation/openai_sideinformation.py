import getpass
from openai import OpenAI
import os

import time
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import tiktoken
import requests
import torch
from dotenv import load_dotenv


from tenacity import retry, stop_after_attempt, wait_fixed

input_folder_path = r"/data/VW_documentation_rag/data_markdown"
persist_directory = r'/data/VW_documentation_rag'

def count_tokens(text, encoding_name='cl100k_base'):
    """Helper function to count tokens in a text."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def ingest():
    # Load the documents
    documents_text = []
    for file in os.listdir(input_folder_path):
        if file.endswith('.md'):
            md_path = os.path.join(input_folder_path, file)
            loader_txt = TextLoader(md_path, encoding="utf-8")
            txt = loader_txt.load()
            documents_text.extend(txt)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=768)

    max_tokens_per_batch = 800000  # Adjust this to stay within your limits
    current_batch = []
    current_tokens = 0

    for doc in documents_text:
        doc_tokens = count_tokens(doc.page_content)

        # Check if adding this document would exceed the token limit
        if current_tokens + doc_tokens > max_tokens_per_batch:
            # Process and persist the current batch
            db = Chroma.from_documents(
                documents=current_batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            db.persist()
            print("Batch persisted. Pausing for 1 minute to avoid rate limits.")
            time.sleep(60)  # Pause to avoid rate limits

            # Reset the batch
            current_batch = []
            current_tokens = 0

        # Add the current document to the batch
        current_batch.append(doc)
        current_tokens += doc_tokens

    # Persist any remaining documents in the final batch
    if current_batch:
        db = Chroma.from_documents(
            documents=current_batch,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        db.persist()

    print("All documents processed and persisted.")
    return documents_text

def load_persist_db(persist_directory, embedding_function):
    with os.scandir(persist_directory) as it:
        if any(it):
            print("Loading existing vector database.")
            return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        else:
            raise Exception("No vector db found. Please run ingest() to create one.")

def load_chain():
    # Initialize the model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=768)

    # Load the existing database
    db = load_persist_db(persist_directory, embeddings)

    return db

if __name__ == "__main__":
    load_dotenv()

    print("API Key for openai embedding and chatgpt")
    api_key = os.environ.get("OPENAI_API_KEY")

    # Use the API key in the client
    client = OpenAI(api_key=api_key)

    # Load the DataFrame
    df_messages = pd.read_parquet('/data/merged_logs_with_workflow.parquet')

    message_content = df_messages['message_content'].tolist()

    # Uncomment the following line to ingest documents and create the vector database
    documents_text = ingest()

    db = load_chain()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    summary = []
    classification = []
    target = []

    for user_input in tqdm(message_content, desc='Processing and Retrieving'):

        if user_input is None or user_input.lower() in ['exit', 'quit']:
            continue

        docs = retriever.invoke(user_input)

        if not docs:
            print("No documents matched the query.")
            continue

        context = [doc.page_content for doc in docs]
        model = 'gpt-4o-mini'  


        try:
            context_text = "\n".join(context)

            # Generate summary using chat model

            summary_prompt_content = f"""
            You are an expert AI assistant specializing in summarizing and augmenting information for BIM authoring tool logs. You possess in-depth knowledge of the BIM authoring tool 'Vectorworks' on PC and are proficient in interpreting and analyzing logs generated during its operations.
            For each step, you will receive a unique log content corresponding to a specific operation. Using your advanced analytical capabilities, you will process and interpret the log and associated information to provide a detailed summary and augmented insights.
            In prior steps, you successfully extracted relevant information from the log content and retrieved the three most relevant markdown files from the official 'Vectorworks' documentation, which likely contain detailed explanations and descriptions of the logs.
            The provided log content and relevant documentations are:

            Log content:
            <$user_input$>

            Relevant Documentations:
            <$context_text$>


            Log_Description:
            1. Provide a concise and comprehensive description of the log in two to three sentences, focusing solely on its most critical aspects.
            2. Ensure a consistent format for all log descriptions. Do not reference the source of the documentation or mention if relevant information was unavailable.
            3. If relevant information cannot be derived directly from the provided documentation, deduce a plausible description based on your knowledge of the same BIM authoring tool. Ensure the description adheres to the same format without referencing comparisons or assumptions.  
            4. The description must remain general and abstract, providing a high-level explanation of the log's purpose or functionality without delving into specific usage scenarios or details.
            """
            
            summary_user_content = f"""
                Log content: {user_input}
                =========
                Relevant Documentations: {context_text}
                =========
                Answer: """
            

            summary_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": summary_prompt_content},
                    {"role": "user", "content": summary_user_content}
                ],
                temperature=0.2
            )
            summary_text = summary_response.choices[0].message.content.strip()
            summary.append(summary_text)

            # Generate classification
            classification_prompt_content = f"""
            You are an expert AI assistant specializing in classifying BIM authoring tool logs. With in-depth knowledge of the BIM authoring tool 'Vectorworks' on PC, you excel at interpreting and analyzing logs generated during its operations.

            For each step, you will receive unique log content corresponding to a specific operation. Using your advanced analytical capabilities, you will process and interpret the log and its associated information to determine and provide a single-word class summarizing its primary function.
            
            In prior steps, you successfully extracted relevant information from the log content and retrieved the three most relevant markdown files from the official 'Vectorworks' documentation, which likely contain detailed explanations and descriptions of the logs. You have also provided a summary of the log based on the official documentation and existing classes.

            The provided log content, relevant documentations and summary are:

            Log Content: <$user_input$>
            Relevant Documentations: <$context_text$>
            Log Summary: <$summary_text$>
            Existing Classes: <$classification%>



            Classification Rules:
            1. The class must be a single, concise word that best summarizes the primary function or purpose of the log (e.g., 'Create', 'Read', 'Update'). Avoid using modifiers, additional context, or extra information.
            2. The classification should be general and abstract, offering a high-level representation of the log's purpose without specific references to detailed scenarios or unique contexts. For instance, if multiple terms represent similar actions (e.g., moving or dragging), classify them under a high-level category such as 'Move.'
            3. Determine the class based on the log description, log name, and relevant documentation, ensuring the chosen class reflects the log's core functionality.
            4. If the log's purpose cannot be explicitly identified or no clear class can be assigned, output only "other" for consistency.
            5. Ensure uniformity by adhering to standard BIM authoring tool terminology, maintaining clarity, and avoiding ambiguity.
            6. When assigning a class, consider existing classes first. If an appropriate class already exists, use it. If none applies, create a new class that fits the log's purpose while following the above rules.


             """
            classification_user_content = f"""                
                Command name: {user_input}
                =========
                Documentation: {context_text}
                =========
                Summary: {summary_text}
                =========
                Existing Classes: {classification}
                Answer: """

            classification_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": classification_prompt_content},
                    {"role": "user", "content": classification_user_content}
                ],
                max_tokens=50,
                temperature=0.2
            )
            classification_text = classification_response.choices[0].message.content.strip()
            classification.append(classification_text)

            # Generate target description
            target_prompt_content = f"""
            You are an expert AI assistant specializing in detecting the target of the logs from BIM authoring tools. With in-depth knowledge of the BIM authoring tool 'Vectorworks' on PC, you excel at interpreting and analyzing logs generated during its operations.

            For each step, you will receive unique log content corresponding to a specific operation. Using your advanced analytical capabilities, you will process and interpret the log and its associated information to determine and provide a single-word target.

            In prior steps, you successfully extracted relevant information from the log content and retrieved the three most relevant markdown files from the official 'Vectorworks' documentation, which likely contain detailed explanations and descriptions of the logs. You have also provided a summary of the log based on the official documentation, the classes of the log information and existing targets.

            The provided log content, relevant documentation, and summary are:

            Log Content: <$user_input$>

            Relevant Documentations: <$context_text$>

            Log Summary: <$summary_text$>

            Classification: <$classification_text$>

            Target Detection Rules:
            As an application expert and a helpful assistant, you can determine the most relevant items in the BIM authoring tool for the current log. You should obey the following rules:
            1. The target must be a single, concise word summarizing the primary focus or function of the log (e.g., 'Object', 'Group'). Avoid adding modifiers, extra context, or unnecessary details.
            2. The classification should remain general and abstract, offering a high-level representation of the log's purpose without delving into specific scenarios or unique contexts.
            3. If the target represents a general or broad object (e.g., tool, roof, component), classify it uniformly as 'Object.' Apply this rule consistently across similar cases.
            4. Use the provided log content, relevant documentation, log summary, and classification to deduce the target accurately and concisely.
            5. If the target cannot be explicitly determined from the given information, output only "other" for consistency.
            6. Maintain a uniform format for all target classifications, ensuring alignment with standard BIM authoring tool terminology.
            7. Avoid mentioning processes, sources, or the absence of information in your response. Focus solely on providing the most appropriate target word.
            8. When assigning a target, consider existing targets first. If an appropriate target already exists, use it. If none applies, create a new target that fits the log's purpose while following the above rules.


            Your task is to interpret the log and output the correct target word while strictly adhering to these rules.
            """

            target_user_content = f"""                
                
                Command name: {user_input}
                =========
                Documentation: {context_text}
                =========
                Summary: {summary_text}
                =========
                Classification: {classification_text}
                =========
                Existing Targets: {target}
                Answer: """

            target_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": target_prompt_content},
                    {"role": "user", "content": target_user_content}
                ],
                max_tokens=50,
                temperature=0.2
            )
            target_text = target_response.choices[0].message.content.strip()
            target.append(target_text)

        except Exception as e:
            print(f"An error occurred for input '{user_input}': {e}")
            summary.append(None)
            classification.append(None)
            target.append(None)

    df_messages['summary'] = summary
    df_messages['classification'] = classification
    df_messages['target'] = target

    # Save the DataFrame to a CSV file
    output_path = os.path.join(persist_directory, 'data/command_information_augmentations.csv')
    df_messages.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")