import os
import json
import pandas as pd
from pathlib import Path
from typing import Annotated, List
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from settings import settings, config_list_extraction, config_list_user_proxy  # Import all config_lists

import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Data Loading and Vector Database Setup ---

# Define predefined CSV column names and descriptions
LEADS_CSV_COLUMNS = [
    {"name": "Company-Name", "description": "Name of the company"},
    {"name": "Company-Size", "description": "Size of the company (e.g., number of employees, revenue bracket)"},
    {"name": "Website", "description": "Company website URL"},
    {"name": "Revenue", "description": "Company's annual revenue"},
    {"name": "Location", "description": "Company's primary location (city, state, country)"},
    {"name": "Full-Name", "description": "Full name of the contact person"},
    {"name": "First-Name", "description": "First name of the contact person"},
    {"name": "Last-Name", "description": "Last name of the contact person"},
    {"name": "Title", "description": "Job title of the contact person"},
    {"name": "Email", "description": "Email address of the contact person"},
    {"name": "LinkedIn-Profile-Link", "description": "LinkedIn profile URL of the contact person"}
]


# Load your leads data from CSV using pandas
try:
    leads_df = pd.read_csv(settings.leads_csv_path)
    print(f"Successfully loaded leads data from: {settings.leads_csv_path}")
except FileNotFoundError:
    print(f"Error: Leads CSV file not found at: {settings.leads_csv_path}")
    leads_df = pd.DataFrame()

# --- Get absolute path to leads.csv ---
leads_csv_absolute_path = os.path.abspath(settings.leads_csv_path)
print(f"Absolute path to leads.csv: {leads_csv_absolute_path}") # Optional: Print for verification


# Initialize Sentence Transformer model for embeddings
embedding_model_name = "all-mpnet-base-v2"  # Choose a suitable model
embed_model = SentenceTransformer(embedding_model_name)

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection_name = "lead_data_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

def initialize_vector_db_with_leads_data(leads_df: pd.DataFrame):
    """
    Initializes the ChromaDB vector database with data from the leads DataFrame.
    """
    if leads_df.empty:
        print("Leads DataFrame is empty, skipping vector DB initialization.")
        return

    documents = []
    ids = []
    embeddings = []
    metadatas = []

    for index, row in leads_df.iterrows():
        document_text = " ".join(str(v) for v in row.values) # Flatten row to text
        documents.append(document_text)
        ids.append(f"lead_{index}") # Unique ID for each lead
        metadata = row.to_dict() # Store row data as metadata
        metadatas.append(metadata)
        embeddings.append(embed_model.encode(document_text).tolist()) # Generate embedding

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Vector database initialized with {len(leads_df)} leads.")


# Initialize the vector database when the script starts
initialize_vector_db_with_leads_data(leads_df)


# --- 2. Vector Database Search Function ---

def vector_search_leads_db(query: str, num_results: int = 5) -> str:
    """
    Searches the ChromaDB vector database based on a user query using vector similarity.
    Returns the top num_results most relevant lead information as JSON.
    """
    if collection.count() == 0:
        return "Error: Vector database is empty. Please check if leads data was loaded and indexed correctly."

    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
        include=["documents", "metadatas"] # Include document text and metadata in results
    )

    if results and results["documents"] and results["documents"][0]:
        # Format results for agent consumption (returning metadata for structured info)
        formatted_results = []
        for i in range(len(results["documents"][0])): # Iterate through results for each query (only 1 query here)
            formatted_results.append({
                "document": results["documents"][0][i], # Full text document (optional, can remove if only metadata is needed)
                "metadata": results["metadatas"][0][i] # Metadata (CSV row data)
            })
        return json.dumps(formatted_results, indent=2)
    else:
        return "No relevant leads found in the vector database for your query."



# --- 3. Configure Agents ---

# Create a directory for code execution
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)


# Format column definitions for system message
column_definitions_text = ""
for col_def in LEADS_CSV_COLUMNS:
    column_definitions_text += f"- '{col_def['name']}' : {col_def['description']}\n"


# Create Extraction Agent (AssistantAgent - now with stronger instructions)
extraction_agent = AssistantAgent(
    name="lead_extraction_agent",
    llm_config={"config_list": config_list_extraction,"seed": 42,
    "temperature": 0,},
    system_message="""You are a Lead Extraction Expert using Retrieval-Augmented Generation (RAG).
    Your goal is to answer user queries about leads by using information retrieved from a vector database, and ONLY from the provided 'leads.csv' data.
    You MUST ONLY use the information provided in the retrieved context from the 'vector_search_leads_db' tool.
    Do not access external websites, browse the internet, or use any external databases or tools. Do not rely on pre-existing knowledge - ONLY use the 'leads.csv' data.
    If the retrieved context does not contain the answer to the user's query, you MUST respond with: 'I could not find relevant information in the leads data for your query.' Do not make up answers.

    Here are the columns available in the 'leads.csv' data and their descriptions:
    {column_definitions}

    You MUST use these column names EXACTLY as provided when referring to columns in the leads data.
    For example, to refer to the company's name, use 'Company-Name', to refer to revenue, use 'Revenue', and so on.

    When responding to user queries, you should primarily focus on using information from these columns.
    If a user query refers to company size, you should look for information in the 'Company-Size' column.
    If they ask for contact emails, you should use the 'Email' column, and so forth.

    Understand the meaning of each column based on its description to provide accurate answers.

    If the user's query requires data analysis, calculations, or more complex processing of the lead data, you should guide the User Proxy agent to write and execute Python code.
    Clearly instruct the User Proxy on what code to execute and what you need to achieve with the code.
    Ensure the code reads data ONLY from the provided 'leads.csv' file using the absolute path: {leads_csv_absolute_path}.
    After code execution, analyze the results to formulate your final answer.

    Your responses MUST be based ONLY on the data from 'leads.csv'. Do not provide any made-up information or answers from outside the provided data. Be truthful and only provide information you can verify from the leads data.

    Use the 'vector_search_leads_db' tool to retrieve relevant leads data to answer questions.
    """.format(column_definitions=column_definitions_text, leads_csv_absolute_path=leads_csv_absolute_path),
    function_map={
        "vector_search_leads_db": vector_search_leads_db,
    }
)


# Register the vector search tool for the extraction agent
@extraction_agent.register_for_llm(description="Searches the vector database for leads information relevant to the user query. Use this tool to retrieve context for answering questions about leads.  Input is the user query and the number of results to retrieve.")
def get_leads_context(query: Annotated[str, "User query to search leads data"], num_results: Annotated[int, "Number of search results to retrieve"] = 5) -> str:
    """Tool to search and retrieve information from the vector database."""
    return vector_search_leads_db(query=query, num_results=num_results)


# Create Filtering Agent (No model change needed for this example)
filtering_agent = AssistantAgent(
    name="lead_filtering_agent",
    llm_config={"config_list": config_list_extraction},
    system_message="""You are a Lead Filtering Expert. Your role is to refine and filter the lead information
    provided by the Lead Extraction Agent.
    You will receive potentially relevant lead information in JSON format.
    Your task is to examine this information and filter it based on relevance and quality, based on the original user query.
    Focus on removing any irrelevant or low-quality leads and ensure the remaining leads are highly relevant to the original user query.
    Return the filtered lead information in JSON format. If no leads are relevant after filtering, return: 'No relevant leads after filtering.'
    Base your filtering decisions ONLY on the information provided by the Lead Extraction Agent, which is derived from 'leads.csv'.

    Here are the columns available in the 'leads.csv' data and their descriptions:
    {column_definitions}
    """.format(column_definitions=column_definitions_text), # Optional: Add column definitions here as well
)


# Create Summarization Agent (No model change needed for this example)
summarization_agent = AssistantAgent(
    name="lead_summarization_agent",
    llm_config={"config_list": config_list_extraction},
    system_message="""You are a Lead Summarization Expert. Your task is to summarize the filtered lead information
    provided by the Lead Filtering Agent into a concise and user-friendly summary for the user.
    You will receive filtered lead information in JSON format.
    Your goal is to create a brief summary highlighting the key findings and relevant leads
    in a way that is easy for a user to understand quickly.
    Focus on summarizing the most important details for each lead and presenting them clearly.
    If there are no relevant leads to summarize, indicate that no leads were found.  Base your summary ONLY on the filtered lead information.

    Here are the columns available in the 'leads.csv' data and their descriptions:
    {column_definitions}
    """.format(column_definitions=column_definitions_text), # Optional: Add column definitions here too
)


# Create User Proxy Agent (now with deepseek-r1-distill-qwen-32b model and code execution)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config=False,
    # is_termination_msg is removed - we will handle termination programmatically
)


# --- 4. Conversation Flow ---

if __name__ == "__main__":
    while True: # Main loop for continuous interaction
        user_query = input("Enter your lead finding query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break # Exit the loop if user types 'exit'

        user_proxy.initiate_chat(
            extraction_agent,
            message=f"""Hello Lead Extraction Agent, please use the 'get_leads_context' tool to retrieve relevant leads data from the vector database to answer the following query:

            User Query: {user_query}
            The absolute path to the 'leads.csv' file is: {leads_csv_absolute_path}. Use this path if you need to read the CSV file in any code execution.
            Here are the predefined columns in the leads data you can use, along with their descriptions:
            {column_definitions_text}

            Please provide a initial extraction of potential leads based on the retrieved context.
            """,
            summary_method="last_msg" # Keep only the last message in the conversation history for cleaner context
        )

        user_proxy.send(
            recipient=filtering_agent,
            message="""Please filter the lead information I am forwarding from the Lead Extraction Agent.
            Focus on relevance and quality to the user's original query.
            Return the filtered lead information in JSON. If no leads are relevant after filtering, respond with:
            'No relevant leads after filtering.'
            """,
            request_reply=True # Expect a reply from the filtering agent
        )


        user_proxy.send(
            recipient=summarization_agent,
            message="""Please summarize the filtered lead information I am forwarding from the Filtering Agent into a concise summary for the user.
            If the Filtering Agent indicated no relevant leads were found after filtering, please also indicate that in your summary.
            """,
            request_reply=False # No reply expected from summarization agent in this example, it's the final step
        )


        print("\n--- Lead Finding Process Completed. Summary ---")
        # Print the final summary from the summarization agent
        print(user_proxy.chat_messages[summarization_agent][-1]['content']) # Access last message content for summary.

        # Clear the agents' conversation histories to start fresh for the next query
        user_proxy.reset()
        extraction_agent.reset()
        filtering_agent.reset()
        summarization_agent.reset()
        print("-" * 50) # Separator for readability in continuous interaction

    print("Exiting Lead Finder Agent.")