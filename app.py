
    
import os
import re
import uuid
import json
import pandas as pd
# import snowflake.connector
from databricks import sql
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from flask import Flask, request, jsonify
import threading
from flask_cors import CORS
from io import BytesIO
import base64
from PIL import Image
import plotly.io as pio
import time
from databricks import sql

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import ConversationChain

# Define your Databricks connection parameters
DATABRICKS_SERVER_HOSTNAME = "<databricks server hostname>"
DATABRICKS_HTTP_PATH = "/sql/13.2/"
DATABRICKS_ACCESS_TOKEN = "databricks token insert here"
try:
    os.environ["TAVILY_API_KEY"] = "<tavily api key>" 
    tavily_search = TavilySearchResults(max_results=5)
except Exception as e:
    print(f"❌ Failed to Initialize Tavily Search: {e}")
    tavily_search = None
# Initialize global memory dictionary to track sessions
session_memories = {}

def get_or_create_memory(session_id):
    """
    Get or create a memory instance for a specific session
    
    Parameters:
    session_id (str): Unique identifier for the user session
    
    Returns:
    ConversationBufferMemory: Memory instance for the session
    """
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=True)
    return session_memories[session_id]

def generate_response_with_memory(llm, user_question, result_df, session_id):
    """
    Generate a response with memory context
    
    Parameters:
    llm: The language model
    user_question (str): The current user question
    result_df (DataFrame): Query results
    session_id (str): Unique identifier for the user session
    
    Returns:
    str: The generated response
    """
    # Get or create memory for this session
    memory = get_or_create_memory(session_id)
    
    # Get conversation history
    conversation_history = memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []
    
    # Format conversation history for context
    history_text = ""
    if conversation_history:
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                history_text += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                history_text += f"AI: {message.content}\n"
    
    # Create explanation prompt with conversation history
    explanation_prompt = f"""
    You need to generate a concise, relevant explanation based on the query results provided, while considering the conversation history. Follow these strict guidelines:

    1. **Conversation Context:**
    - Use the conversation history to understand context and references to previous information.
    - If the user refers to previous orders, entities, or information mentioned earlier, use that context to provide an appropriate response.

    2. **General Summary:**  
    - Provide a **one-sentence summary** of the query results.  
    - If the column `average_delay` exists in the query result, interpret it in **days**, not weeks.  

    3. **Yes/No Response:**  
    - **Only include 'yes' or 'no'** at the beginning of your response **if and only if** the user's question (`{user_question}`) contains the word **'can'**.  
    - If the question does not contain 'can', **do not include 'yes' or 'no'** in your response.  

    4. **Cost Impact Calculation (Strictly When Asked):**  
    - If and only if the user question explicitly asks for **cost impact**, follow this structured format:  
        - **List available Modes of Transport (MOT).**  
        - Assume the **initial MOT is ROAD** and calculate additional costs for other MOTs (AIR, RAIL).  
        - Use the formula:  
        - **AIR:** (cost_for_air - cost_for_road) * requested_quantity  
        - **RAIL:** (cost_for_rail - cost_for_road) * requested_quantity  
        - Replace `cost_for_rail`, `cost_for_air`, `cost_for_road`, and `requested_quantity` with their actual values from the DataFrame.  
        - Ensure output format strictly follows this structure:

        ```
        For order number nnn, the lead time and cost per unit for alternate vendors are as follows:
        AIR: Lead time is x weeks, Cost per unit is $x, Additional cost is $x
        ROAD: Lead time is x weeks, Cost per unit is $x
        RAIL: Lead time is x weeks, Cost per unit is $x, Additional cost is $x
        ```

    5. **Revenue Analysis:**  
    - If revenue is mentioned, express values in **millions of dollars**.  

    6. **Strict Output Constraints:**  
    - Do not perform **any calculations** unless explicitly asked for cost impact.  
    - Do not include any unnecessary analysis—keep responses **short, crisp, and directly relevant** to the question.  
    - Do not include **step-by-step calculations** in the response, only the final numbers.  

    **Conversation History:**
    {history_text}

    **Data:**  
    {result_df.head(10).to_json()}  

    **User Question:**  
    "{user_question}"

    Provide only the relevant response as per the rules above.
    """
    
    # Get response
    explanation_response = llm.invoke(explanation_prompt).content.strip()
    
    # Update memory with this interaction
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(explanation_response)
    
    return explanation_response


def create_connection():
    try:
        conn = sql.connect(
            server_hostname=DATABRICKS_SERVER_HOSTNAME,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_ACCESS_TOKEN
        )
        return conn
    except Exception as e:
        print(f"❌ Databricks Connection Failed: {e}")
        raise

import pandas as pd

def get_databricks_metadata(conn, catalog_name, schema_name):
    metadata_query = f"""
        SELECT table_name, column_name, data_type
        FROM {catalog_name}.information_schema.columns
        WHERE table_catalog = '{catalog_name}' AND table_schema = '{schema_name}';
    """
    try:
        cursor = conn.cursor()
        cursor.execute(metadata_query)
        metadata_rows = cursor.fetchall()
        cursor.close()

        if not metadata_rows:
            raise ValueError("⚠️ No metadata retrieved! Check database permissions or schema name.")

        metadata_df = pd.DataFrame(metadata_rows, columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
        metadata_dict = (
            metadata_df.drop(columns=["TABLE_NAME"])
            .groupby(metadata_df["TABLE_NAME"], group_keys=False)
            .apply(lambda x: {col: dtype for col, dtype in zip(x["COLUMN_NAME"], x["DATA_TYPE"])})
            .to_dict()
        )
        return metadata_dict
    except Exception as e:
        print(f"❌ Error fetching metadata: {str(e)}")
        return None


def query_databricks(conn, sql_query):
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        return pd.DataFrame(result, columns=columns)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def visual_generate(llm, query, data, response):
    try:
        prompt = f"""
        Give me Python code that can generate an insightful graph or plot on the given dataframe for the query based on response using Plotly.
        Dataframe: {data}.
        Query: {query}
        Response: {response}

        Follow below color schema for the plot:
        - Background: black (#2B2C2E)
        - Text and labels: white (#FFFFFF)
        - Bars/Lines: either #4BC884 or #22A6BD
        Also, add data labels.

        DO NOT include any additional text other than the Python code in the response.
        Save the plot at the end using: fig.write_image('graph.png', engine='kaleido')

        If it's not possible to generate a graph, return 'No graph generated' as a response.
        """

        code = llm.invoke(prompt).content
        if "No graph generated" in code:
            encoded_image = ''
        else:
            code = re.sub(r'```python', '', code)
            code = re.sub(r'`', '', code)
            exec(code)
            # plt.savefig('graph.png', dpi=300, bbox_inches='tight', facecolor='black')
            with open('graph.png', 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print("⚠️ Graph generation error:", e)
        encoded_image = ''
    return encoded_image
def classify_query(llm, user_question):
    """
    Determines if a question is data-specific or general knowledge
    
    Parameters:
    llm: The language model
    user_question (str): The user's question
    
    Returns:
    dict: Classification with query_type and confidence
    """

    classification_prompt = f"""
    You are a query classifier that determines if a user question requires data analysis or general knowledge.
    
    Question: "{user_question}"
    
    Analyze this question and determine if it requires:
    1. DATA_ANALYSIS - Questions about orders, revenue, costs, deliveries, metrics, or anything that would need database access
    2. GENERAL_KNOWLEDGE - General questions, factual information, advice, or anything not related to company data
    
    Output your answer in JSON format with these fields:
    - query_type: Either "DATA_ANALYSIS" or "GENERAL_KNOWLEDGE"
    - confidence: A number between 0 and 1 indicating your confidence in this classification
    - reasoning: A brief explanation of your reasoning
    
    JSON format only with no additional text:
    """
    
    parser = JsonOutputParser()
    
    try:
        # First get the raw response from the LLM
        response = llm.invoke(classification_prompt)
        
        # Then parse the JSON
        try:
            # Try to parse directly
            import json
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to use the parser
            result = parser.invoke(response.content)
            
        return result
    except Exception as e:
        print(f"❌ Classification error: {str(e)}")
        # Default to data analysis if classification fails
        return {"query_type": "DATA_ANALYSIS", "confidence": 0.5, "reasoning": "Classification failed"}
def get_general_knowledge_response(llm, user_question):
    """
    Generates a response for general knowledge questions using Tavily search
    
    Parameters:
    llm: The language model
    user_question (str): The user's question
    
    Returns:
    str: The generated response
    """
    if not tavily_search:
        return "I'm having trouble accessing my knowledge tools right now. Please try again later."
    
    try:
        # Get search results
        search_results = tavily_search.invoke(user_question)
        
        # Create prompt with search results
        search_prompt = f"""
        You need to answer the user's question based on the search results provided.
        
        User question: "{user_question}"
        
        Search results:
        {search_results}
        
        Provide a helpful, accurate, and concise response based on these search results.
        If the search results don't provide sufficient information, acknowledge this
        and provide the best answer you can based on your knowledge.
        """
        
        # Get response
        response = llm.invoke(search_prompt).content.strip()
        return response
    except Exception as e:
        print(f"❌ General knowledge response error: {str(e)}")
        return "I'm having trouble finding information about that right now. Please try again later."
def handle_database_info_query(conn, user_question, llm):
    """
    Handle queries about database structure, tables, or metadata
    
    Parameters:
    conn: Snowflake connection
    user_question (str): User's question about database
    llm: Language model
    
    Returns:
    dict: Response with table information
    """
    try:
        # Fetch all table names and their descriptions
        metadata_query = """
        SELECT 
            TABLE_NAME, 
            CASE 
                WHEN TABLE_TYPE = 'BASE TABLE' THEN 'Standard table'
                WHEN TABLE_TYPE = 'VIEW' THEN 'View'
                ELSE TABLE_TYPE 
            END AS TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'sop_da';
        """
        
        cursor = conn.cursor()
        cursor.execute(metadata_query)
        tables = cursor.fetchall()
        cursor.close()
        
        # Prepare table information
        table_info = [
            {
                "name": table[0], 
                "type": table[1]
            } for table in tables
        ]
        
        # Generate a human-readable description
        description_prompt = f"""
        Given the following database tables:
        {json.dumps(table_info, indent=2)}
        
        Create a concise, informative description that explains the purpose and contents of these tables.
        Focus on how these tables might be related and what kind of business insights they could provide.
        """
        
        # Get description from LLM
        description = llm.invoke(description_prompt).content.strip()
        
        return {
            "table_count": len(tables),
            "tables": table_info,
            "description": description
        }
    
    except Exception as e:
        print(f"❌ Database info query error: {str(e)}")
        return {
            "table_count": 0,
            "tables": [],
            "description": "Unable to retrieve database information."
        }


from langchain_openai import AzureChatOpenAI
# def generate_sql_query(user_question, metadata, catalog_name, schema_name):
#     """
#     Generates an SQL query using LLM with schema context.

#     Parameters:
#     user_question (str): The natural language question from the user.
#     metadata (dict): A dictionary containing table names,their column names along with their datatypes.
#     catalog_name (str): The catalog name in Unity Catalog.
#     schema_name (str): The schema name in Unity Catalog.

#     Returns:
#     str: The generated SQL query.
#     """
#     try:
#         # log_event("SQL_Generation", "INFO", f"Generating SQL query for user question: {user_question}")
#         os.environ["AZURE_OPENAI_API_KEY"] = ""
#         os.environ["AZURE_OPENAI_ENDPOINT"] = ""
#         llm = AzureChatOpenAI(
#         azure_deployment="",
#         api_version="2024-08-01-preview",
#         temperature=0,
#         max_tokens=None,
#         timeout=None)

#         metadata_context = "\n".join([f"{catalog_name}.{schema_name}.{table}: {', '.join(columns)}" for table, columns in metadata.items()])
#         with open("instructions.txt","r") as file:
#             terminlogical_instruction = file.read()
#         context = f"""Here are the available tables and their columns with catalog and schema:\n{metadata_context}
#         Strictly follow these instructions to generate the SQL query for respected queries:\n{terminlogical_instruction}
#         Strictly give ONLY the SQL query as response. DO NOT include any additional text.
#         """
#         messages = [
#             ("system", "You are a SQL expert for Databricks Unity Catalog."),
#             ("system", f"""Here are the available tables and their columns with catalog and schema:\n{metadata_context}
#             Strictly follow these instructions to generate the SQL query for respected queries:{context}
#             """),
#             ("human", user_question),
#         ]
#         # print(messages)
#         # log_event("SQL_Generation", "INFO", "Invoking LLM for SQL generation.")
#         response = llm.invoke(messages).content
#         return response
#     except Exception as e:
#         # log_event("SQL_Generation", "ERROR", f"Failed to generate SQL query: {str(e)}")
#         return None

def generate_sql_query_with_memory(user_question, metadata, catalog_name, schema_name, session_id, memory):
    """
    Generates an SQL query using LLM with schema context and session memory.

    Parameters:
    user_question (str): The natural language question from the user.
    metadata (dict): A dictionary containing table names, their column names along with their datatypes.
    catalog_name (str): The catalog name in Unity Catalog.
    schema_name (str): The schema name in Unity Catalog.
    session_id (str): Unique identifier for the user session.
    memory: The memory object to store conversation history.

    Returns:
    str: The generated SQL query.
    """
    try:
        memory = get_or_create_memory(session_id)
        # Get conversation history
        conversation_history = memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []
        
        # Format conversation history for context
        history_text = ""
        if conversation_history:
            for message in conversation_history:
                if isinstance(message, HumanMessage):
                    history_text += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_text += f"AI: {message.content}\n"

        # Prepare LLM configuration
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
        llm = AzureChatOpenAI(
            azure_deployment="",
            api_version="2024-08-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None
        )

        # Prepare metadata and context
        metadata_context = "\n".join([f"{catalog_name}.{schema_name}.{table}: {', '.join(columns)}" for table, columns in metadata.items()])
        with open("instructions.txt", "r") as file:
            terminological_instruction = file.read()

        # Create a comprehensive context-aware prompt
        sql_generation_prompt = f"""
        You are an expert SQL query generator for Databricks Unity Catalog. Your task is to generate a precise SQL query based on the user's question, considering the conversation history and available schema.

        **Conversation History:**
        {history_text}

        **Available Tables and Columns:**
        {metadata_context}

        **SQL Generation Instructions:**
        {terminological_instruction}

        **Important Guidelines:**
        1. Use the conversation history to understand context and potential references to previous queries.
        2. Ensure the SQL query is directly relevant to the user's current question.
        3. Strictly adhere to the SQL generation instructions provided.
        4. Return ONLY the SQL query without any additional text or explanation.

        **User Question:**
        {user_question}

        Generate the most appropriate SQL query.
        """

        # Invoke LLM to generate SQL query
        messages = [
            ("system", "You are a SQL expert for Databricks Unity Catalog."),
            ("system", sql_generation_prompt),
            ("human", user_question),
        ]
        
        response = llm.invoke(messages).content

        # Update memory with this interaction
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(response)

        return response
    except Exception as e:
        # Log the error (replace with your preferred logging mechanism)
        print(f"Error in SQL query generation: {str(e)}")
        return None
os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_ENDPOINT"] = ""
os.environ["AZURE_API_VERSION"] = ""
os.environ["AZURE_DEPLOYMENT_NAME"] = ""

try:
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        temperature=0.7
    )
except Exception as e:
    print(f"❌ Failed to Initialize Azure OpenAI Model: {e}")
    raise

app = Flask(__name__)

@app.route('/getdata', methods=['POST'])
def query_api():
    data = request.get_json()
    user_question = data.get("user_question")
    session_id = data.get("session_id", "default_session")

    if not user_question:
        return jsonify({"message": "No user_question provided", "result": {}}), 400
    classification = classify_query(llm, user_question)
    print(f"Query classification: {classification}")
    
    
    # Check for database information queries
    database_keywords = [
        "database","structure", 
        "information","description of tables"
    ]
    
    if any(keyword.lower() in user_question.lower() for keyword in database_keywords):
        conn = create_connection()
        try:
            db_info = handle_database_info_query(conn, user_question, llm)
            
            # Generate a response
            response_prompt = f"""
            Create a helpful summary about the database based on this information:
            {json.dumps(db_info, indent=2)}
            
            User's original question: {user_question}
            """
            
            response = llm.invoke(response_prompt).content.strip()
            
            # Update memory
            memory = get_or_create_memory(session_id)
            memory.chat_memory.add_user_message(user_question)
            memory.chat_memory.add_ai_message(response)
            
            return jsonify({
                "message": response,
                "result": db_info,
                "session_id": session_id
            })
        
        finally:
            conn.close()
    # Route to appropriate handler based on classification
    if classification["query_type"] == "GENERAL_KNOWLEDGE" and classification["confidence"] > 0.6:
        # Handle general knowledge query with Tavily search
        response = get_general_knowledge_response(llm, user_question)
        
        # Update memory
        memory = get_or_create_memory(session_id)
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(response)
        result_list=pd.DataFrame({"tavily response": [response]})
        # Return response with only message key
        return jsonify({
            "message": response,
            "result": result_list.to_dict(orient="records"),
            "session_id": session_id
        })
    else:
        conn = create_connection()
        catalog = "sop_da"
        schema = "sop_da"

        table_metadata = get_databricks_metadata(conn,catalog, schema)
        if not table_metadata:
            conn.close()
            return jsonify({"message": "Metadata retrieval failed.", "result": {}}), 500
        # with open("instructions.txt", "r", encoding="utf-8") as file:
        #     system_prompt = file.read().strip()
        # metadata_prompt = f"{system_prompt}\n\nUser Question:\n{user_question}"
        # try:
        #     llm_response = llm.invoke(metadata_prompt).content.strip()

        # sql_query = generate_sql_query(user_question, table_metadata, catalog, schema)
        memory = get_or_create_memory(session_id)
        sql_query = generate_sql_query_with_memory(
        user_question, 
        table_metadata, 
        catalog, 
        schema, 
        session_id,
        memory
        )
        print("sql_query: ",sql_query)
        llm_response=sql_query.strip()
        sql_match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            conn.close()
            return jsonify({"message": "LLM did not return a valid SQL query format.", "result": {}}), 500

        result_df = query_databricks(conn,sql_query)
        explanation_response = generate_response_with_memory(llm, user_question, result_df, session_id)

    #     explanation_prompt = f"""
    #     You need to generate a concise, relevant explanation based on the query results provided. Follow these strict guidelines:

    #     1. **General Summary:**  
    #     - Provide a **one-sentence summary** of the query results.  
    #     - If the column `average_delay` exists in the query result, interpret it in **days**, not weeks.  

    #     2. **Yes/No Response:**  
    #     - **Only include 'yes' or 'no'** at the beginning of your response **if and only if** the user's question (`{user_question}`) contains the word **'can'**.  
    #     - If the question does not contain 'can', **do not include 'yes' or 'no'** in your response.  

    #     3. **Cost Impact Calculation (Strictly When Asked):**  
    #     - If and only if the user question explicitly asks for **cost impact**, follow this structured format:  
    #         - **List available Modes of Transport (MOT).**  
    #         - Assume the **initial MOT is ROAD** and calculate additional costs for other MOTs (AIR, RAIL).  
    #         - Use the formula:  
    #         - **AIR:** (cost_for_air - cost_for_road) * requested_quantity  
    #         - **RAIL:** (cost_for_rail - cost_for_road) * requested_quantity  
    #         - Replace `cost_for_rail`, `cost_for_air`, `cost_for_road`, and `requested_quantity` with their actual values from the DataFrame.  
    #         - Ensure output format strictly follows this structure:

    #         ```
    #         For order number nnn, the lead time and cost per unit for alternate vendors are as follows:
    #         AIR: Lead time is x weeks, Cost per unit is $x, Additional cost is $x
    #         ROAD: Lead time is x weeks, Cost per unit is $x
    #         RAIL: Lead time is x weeks, Cost per unit is $x, Additional cost is $x
    #         ```

    #     4. **Revenue Analysis:**  
    #     - If revenue is mentioned, express values in **millions of dollars**.  

    #     5. **Strict Output Constraints:**  
    #     - Do not perform **any calculations** unless explicitly asked for cost impact.  
    #     - Do not include any unnecessary analysis—keep responses **short, crisp, and directly relevant** to the question.  
    #     - Do not include **step-by-step calculations** in the response, only the final numbers.  

    #     **Data:**  
    #     {result_df.head(10).to_json()}  

    #     **User Question:**  
    #     "{user_question}"

    #     Provide only the relevant response as per the rules above.
    # """
    #     explanation_response = llm.invoke(explanation_prompt).content.strip()
        conn.close()

        result_list = result_df.to_dict(orient="records")
        graph_png = visual_generate(llm, sql_query, result_list, explanation_response)
        graph_png_url = f"data:image/png;base64,{graph_png}" if graph_png else ""

        return jsonify({
            "message": explanation_response,
            "result": result_list,
            "image": graph_png_url,
            "session_id": session_id 
        })
    # also you strictly shouldnt include 'yes' or 'no' in every response, i only need them if our {user_question} is containing 'can', based on query results {result_df.head(10).to_json()} you give me 'yes' or 'no' followed by your summary response above.
    # if the query is asking to calculate cost impacted strictly only then use the below intructions to calcualte or else just summary response above is enough.
    #     You have the following data in a Pandas DataFrame:
    #     {result_df} 
    #     The user asks: "{user_question}" 
    #     Terminology instructions based on user questions:
    #     1.The lead time is in weeks and nnn is order number from query.
    #     2. for cost impact for different mode of transport(MOT) the response should be of following format. First list the MOT available. 
    #     consider intial MOT as ROAD: Respone- AIR - (cost_for_air - cost_for_road)*(requested_quantity) additional cost for (ship_from_loc_id)
    #                                             RAIL - (cost_for_rail - cost_for_road)*(requested_quantity) additional cost (ship_from_loc_id)
    #     give the subtracted value for (cost_for_rail/air - cost_for_road)
    #     always the substitue the actual value from dataframe for cost_for_rail, cost_for_road and requested_quantity
    #     output should be strictly in this example format no above calculations should be in response only i need numbers -
    #     For order number nnn, the lead time and cost per unit for alternate vendor are as follows:
    #     AIR: Lead time is x weeks, Cost per unit is $x, Additional cost is $x
    #     ROAD: Lead time is x weeks, Cost per unit is $x
    #     RAIL: Lead time is x weeks, Cost per unit is $x, Additional cost is $x

    #     get these x values from datafame and do calculations accordinglt and give me output in this format.
    #     Analyse the revenue number in million dollars
    #     Please provide a short and crisp answer based on the data above.dont give all analysis as output ,only give me very short,crisp and relevant answer to question.
# 9️⃣ Run Flask in a separate thread
def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


import streamlit as st
import base64
import time
from io import BytesIO
from PIL import Image
import pandas as pd
import html
import requests
import json  # for JSON serialization if needed
    

# Set page config first
st.set_page_config(layout="wide", page_title="Supply Chain Digital Assistant")

# Session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'initial_messages_shown' not in st.session_state:
    st.session_state.initial_messages_shown = False

# Your CSS code here (keep your existing CSS)
st.markdown("""
<style>
/* Your existing CSS */
</style>
""", unsafe_allow_html=True)

# Functions for pages
def show_home_page():
    logo = Image.open("Images/Bristlecone Logo.png")
    ai_chip = Image.open("Images/microchip-ai.png")
    check_icon = Image.open("Images/checkmark.png")
    threads_icon = Image.open("Images/chat.png")
    hand_icon = Image.open("Images/hand.png")
 
     
    # tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png")
    # check_icon.save(tmp.name)
    # check_icon_path=tmp.name
    buffer = BytesIO()
    logo.save(buffer, format="PNG")
    logo_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logo_path = f"data:image/png;base64,{logo_str}"
    
    buffer = BytesIO()
    threads_icon.save(buffer, format="PNG")
    threads_icon_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    threads_icon_url = f"data:image/png;base64,{threads_icon_str}"
    
    buffer = BytesIO()
    check_icon.save(buffer, format="PNG")
    check_icon_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    check_icon_path = f"data:image/png;base64,{check_icon_str}"
     
    buffer = BytesIO()
    ai_chip.save(buffer, format="PNG")
    ai_chip_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    ai_chip_data_url = f"data:image/png;base64,{ai_chip_str}"

    buffer = BytesIO()
    hand_icon.save(buffer, format="PNG")
    hand_icon_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    hand_icon_url = f"data:image/png;base64,{hand_icon_str}"
     
    # st.set_page_config(layout="wide", page_title="Supply Chain Digital Assistant")  
    # Add this to your existing CSS in st.markdown
    st.markdown("""
    <style>
    /* Hide the default sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Hide the sidebar content when expanded */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Additional style to ensure the sidebar is fully hidden */
    .css-1d391kg, .css-1vq4p4l {
        visibility: hidden;
    }
    </style>
    
    <script>
    // JavaScript to ensure sidebar stays hidden
    document.addEventListener('DOMContentLoaded', function() {
        // Hide any sidebar elements that might appear
        const sidebarElements = document.querySelectorAll('[data-testid="stSidebar"]');
        sidebarElements.forEach(function(element) {
            element.style.display = 'none';
        });
        
        // Also hide the collapse control button
        const collapseControls = document.querySelectorAll('[data-testid="collapsedControl"]');
        collapseControls.forEach(function(control) {
            control.style.display = 'none';
        });
    });
    </script>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
    /* Hide default Streamlit header */
    header {{
      visibility: hidden;
    }}
    
    /* Custom header wrapper spanning full width */
    .custom-header {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: space-between;
      z-index: 10000;
      height: 90px;
      margin: 0;
    }}
    
    /* Left column (black background) */
    .custom-header-left {{
      background-color: #000000;
      width: 25%;
      padding: 0.5rem 1.5rem;
      display: flex;
      align-items: center;
      height: 90px;
    }}
    
    /* Right column (#2B2B2B background) */
    .custom-header-right {{
      background-color: #2B2B2B;
      width: 75%;
      padding: 0.5rem 1.5rem;
      display: flex;
      align-items: right;
      justify-content: flex-start;
      height:90px
    }}
    
    /* Title styling */
    .custom-header-right h1 {{
      color: white;
      font-size: 1.5rem;
      font-weight: 600;
      margin: 0;
      padding: 0;
    }}
    
    /* Push the main app content below the new header */
    .stApp {{
      padding-top: 60px;
    }}
    </style>
    
    <div class="custom-header">
      <!-- Left col: black background + Bristlecone logo -->
      <div class="custom-header-left">
        <img src="{logo_path}" width="280"/>
      </div>
    
      <!-- Right col: #2B2B2B background + title text -->
      <div class="custom-header-right">
        <h1>SUPPLY CHAIN DIGITAL ASSISTANT</h1>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    ##css code
    st.markdown(
        """
        <style>
        /* Hide default Streamlit header and footer
        #MainMenu, header, footer {visibility: hidden;}
        html,body{
          margin : 0;
          padding :0;
          height : 100%;
        }
        */
        /* Overall body background can be set if desired, but we mainly use columns */
        .main, .block-container {
            padding:0;
            height:100vh;
            max-width:100%;
        }
     
        /* Left column styling (thick black) */
        .left-col {
            background-color: #000000;  /* pure black */
            padding: 1rem;
            height:100vh;
            position:fixed;
            left:0;
            width:25%;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
     
        /* Right column styling (light black / dark gray) */
        .right-col {
            background-color: #2B2B2B;
            padding: 1rem 2rem;
            height:200vh;
            position:fixed;
            overflow-y: auto;
            margin-left:25%;
            width:75%;
            padding:1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .content-wrapper {
            display: flex;
            height:100%;
            flex-direction: row;
        }
     
        /* Bristlecone logo */
     
     
        /* "Threads" area */
        .threads-container {
            margin-top: 2rem;
            display: flex;
            align-items: center;
            color: white;
            font-size: 1rem;
        }
        .threads-icon {
            width: 24px;
            margin-right: 0;
        }
     
        /* Title in right column */
        .title-text {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
            padding: 0;
            margin-bottom: 1rem;
        }
     
        .ai-icon {
            width: 25px;
            margin-right: 0.5rem;
        }
     
        /* Container for the three feature boxes */
        .features-container {
            display: flex;
            justify-content: center;
            gap: 0.5rem !important;
            margin-left: 2cm;
            max-width:900px;
            margin: 2rem auto 0;
            margin-top: 4rem;
        }
     
        /* Individual feature box */
        .feature-box {
            background-color: #1E1E1E;
            margin:0 !important;
            padding:1rem;
            width: 250px;
            min-height: 190px;
            border-radius: 10px;
            text-align: left;
            margin-bottom:2rem;
        }
        .feature-title {
            color: #00FF00; /* Green text */
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature-desc {
            color: white; /* White text */
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .check-icon {
            width: 20px;
        }
        /* Top row inside right column */
        .top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
     
        </style>
        """,
        unsafe_allow_html=True
    )
     
     
    st.markdown("""
    <div class="content-wrapper">
        <div class="left-col">
            <!-- Left column content -->
        </div>
        <div class="right-col">
            <!-- Right column content -->
        </div>
    </div>
    """, unsafe_allow_html=True)
     
    # ----- LEFT COLUMN -----
    left_col = st.container()
    with left_col:
        st.markdown('<div class="left-col" style="position:fixed;">', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="position:fixed;">
              <img src="{threads_icon_url}" width="24"/>
              <span style="color:gray;font-size:1rem;position:fixed;margin-left: 10px;">THREADS</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
     
    # ----- RIGHT COLUMN  -----
     
    with st.container():
        st.markdown(
            '<h3 class="title-text" style="margin-left: 10cm; margin-top: -3.5cm;margin-bottom: 1rem;color: white;">SUPPLY CHAIN DIGITAL ASSISTANT</h3>',
            unsafe_allow_html=True
        )
     
        # Get Started button below title
    #     st.markdown(f'''
    #     <div style="text-align: center; margin-top: 20px;">
    #         <button class="get-started-btn" id="get-started-btn">
    #             <img src="{ai_chip_data_url}" style="width: 20px; vertical-align: middle; margin-right: 10px;"/>
    #             Get Started
    #         </button>
    #     </div>
    # ''', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .get-started-btn {
            background-color: #1E1E1E;
            border: 1px solid #1E1E1E;
            padding: 1rem 1.5rem;
            color: white;
            font-size: 1rem;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            text-decoration: none;
            margin-top: 1rem;
        }
        .get-started-btn:hover {
            background-color: #444444;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Target Streamlit button elements */
    div.stButton > button {
        background-color: #1E1E1E;  /* Dark background matching the image */
        color: #00C4CC;            /* Light teal text */
        border: 1px solid #1E1E1E; /* Border matching background */
        padding: 1rem 1.5rem;    /* Increased padding for larger size */
        font-size: 1.1rem;         /* Slightly larger text */
        border-radius: 10px;       /* Rounded corners */
        display: inline-flex;      /* Flexbox for alignment */
        align-items: center;       /* Center items vertically */
        position: relative;        /* For positioning the image */
    }
    div.stButton > button:hover {
        background-color: #444444; /* Hover effect */
    }
    /* Style for the image next to the button */
    .button-image {
        margin-right: 0.5rem;     /* Space between image and text */
        vertical-align: middle;   /* Align with text */
        margin-top: 0cm;
        margin-left:1.3cm;/* Shift image down by 0.2 cm */
        width: 30px;              /* Match image size from your design */
        height: 30px;             /* Match image size from your design */
    }
    .hand-pointer {
        position: absolute;
        left: -50px; /* Adjust position to move left of the button */
        top: 70px;
        transform: translateY(-50%);
        width: 40px; /* Adjust the hand size */
        animation: bounce 1s infinite; /* Apply animation */
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(-50%) translateX(0); }
        50% { transform: translateY(-50%) translateX(-10px); } /* Small movement */
    }
    </style>
""", unsafe_allow_html=True)
    
    # Hidden Streamlit Button for Navigation with Image
    cols = st.columns([2.8, 1, 3])  # Increased first column width to push right
    with cols[2]:
        # Display the image and button side by side
        st.markdown(f'<img src="{hand_icon_url}" class="hand-pointer">', unsafe_allow_html=True)
        st.markdown(
            f'<img src="{ai_chip_data_url}" class="button-image">',
            unsafe_allow_html=True
        )
        # Button with original functionality
        if st.button("Get Started", key="hidden_get_started"):
            st.session_state.page = "sample"
            st.rerun()

    # JavaScript to trigger the hidden button when styled button is clicked
        # st.markdown("""
        #     <script>
        #         document.getElementById('styled-button').addEventListener('click', function() {
        #             document.querySelector('button[kind="secondaryFormSubmit"]').click();
        #         });
        #     </script>
        #     """, unsafe_allow_html=True
        # )
     
    # JavaScript to Trigger the Hidden Streamlit Button
    # st.markdown("""
    #     <script>
    #         document.getElementById('get-started-btn').addEventListener('click', function() {
    #             var hiddenBtn = window.parent.document.querySelector('[aria-label="Hidden Get Started"]');
    #             if (hiddenBtn) hiddenBtn.click();
    #         });
    #     </script>
    # """, unsafe_allow_html=True)
        # Create feature boxes with custom gap and right shift
    st.markdown('''
            <div style="display: flex; gap: 2rem; margin-left: 12cm;margin-top: 1cm;">
                <div class="feature-box style="width: 150px; min-height: 250px;">
                    <div class="feature-title">Information Retrieval</div>
                    <div class="feature-desc">
                        This feature will retrieve the required information and provide visualization.
                    </div>
                    <img src="{check_icon_path}" class="check-icon" />
                </div>
                <div class="feature-box style="width: 150px; min-height: 250px;">
                    <div class="feature-title">Smart Insights</div>
                    <div class="feature-desc">
                        This feature will generate insights by analysing the underlying data.
                    </div>
                    <img src="{check_icon_path}" class="check-icon" />
                </div>
                <div class="feature-box style="width: 150px; min-height: 250px;">
                    <div class="feature-title">Simulation</div>
                    <div class="feature-desc">
                        This feature allows users to run simulations to understand various business what-if scenarios.
                    </div>
                    <img src="{check_icon_path}" class="check-icon" />
                </div>
            </div>
        '''.format(check_icon_path=check_icon_path), unsafe_allow_html=True)

def show_sample_page():
  
  
    # ----------------------------------------------------
    #               IMAGE SETUP (LOGO, THREADS)
    # ----------------------------------------------------
    logo = Image.open("Images/Bristlecone Logo.png")
    threads_icon = Image.open("Images/chat.png")
    
    buffer = BytesIO()
    logo.save(buffer, format="PNG")
    logo_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logo_path = f"data:image/png;base64,{logo_str}"
    
    buffer = BytesIO()
    threads_icon.save(buffer, format="PNG")
    threads_icon_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    threads_icon_url = f"data:image/png;base64,{threads_icon_str}"
    
    # ----------------------------------------------------
    #           STREAMLIT PAGE CONFIG
    # ----------------------------------------------------
    # st.set_page_config(layout="wide", page_title="Supply Chain Digital Assistant")
    
    # ----------------------------------------------------
    #      SESSION STATE INITIALIZATION
    # ----------------------------------------------------
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'initial_messages_shown' not in st.session_state:
        st.session_state.initial_messages_shown = False
    
    # ----------------------------------------------------
    #                  CUSTOM CSS
    # ----------------------------------------------------
    st.markdown("""
    <style>
    /* Hide the default sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Hide the sidebar content when expanded */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Additional style to ensure the sidebar is fully hidden */
    .css-1d391kg, .css-1vq4p4l {
        visibility: hidden;
    }
    </style>
    
    <script>
    // JavaScript to ensure sidebar stays hidden
    document.addEventListener('DOMContentLoaded', function() {
        // Hide any sidebar elements that might appear
        const sidebarElements = document.querySelectorAll('[data-testid="stSidebar"]');
        sidebarElements.forEach(function(element) {
            element.style.display = 'none';
        });
        
        // Also hide the collapse control button
        const collapseControls = document.querySelectorAll('[data-testid="collapsedControl"]');
        collapseControls.forEach(function(control) {
            control.style.display = 'none';
        });
    });
    </script>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <style>
    /* Hide default Streamlit header */
    header {{
      visibility: hidden;
    }}
    
    /* Custom header wrapper spanning full width */
    .custom-header {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: space-between;
      z-index: 10000;
      height: 90px;
      margin: 0;
    }}
    
    /* Left column (black background) */
    .custom-header-left {{
      background-color: #000000;
      width: 25%;
      padding: 0.5rem 1.5rem;
      display: flex;
      align-items: center;
      height: 90px;
    }}
    
    /* Right column (#2B2B2B background) */
    .custom-header-right {{
      background-color: #2B2B2B;
      width: 75%;
      padding: 0.5rem 1.5rem;
      display: flex;
      align-items: right;
      justify-content: flex-start;
      height:90px
    }}
    
    /* Title styling */
    .custom-header-right h1 {{
      color: white;
      font-size: 1.5rem;
      font-weight: 600;
      margin: 0;
      padding: 0;
    }}
    
    /* Push the main app content below the new header */
    .stApp {{
      padding-top: 60px;
    }}
    </style>
    
    <div class="custom-header">
      <!-- Left col: black background + Bristlecone logo -->
      <div class="custom-header-left">
        <img src="{logo_path}" width="280"/>
      </div>
    
      <!-- Right col: #2B2B2B background + title text -->
      <div class="custom-header-right">
        <h1>SUPPLY CHAIN DIGITAL ASSISTANT</h1>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown(
        """
        <style>
        .main, .block-container {
            padding:0;
            height:100vh;
            max-width:100%;
        }
        .left-col {
            background-color: #000000;
            padding: 1rem;
            height:100vh;
            position:fixed;
            left:0;
            width:25%;
            overflow-y: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .right-col {
            background-color: #2B2B2B;
            padding: 1.5rem;
            height:100vh;
            position:fixed;
            overflow-x: auto;
            margin-left:25%;
            width:75%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .content-wrapper {
            display: flex;
            height:100%;
            flex-direction: row;
        }
        .title-text {
            color: white;
            position: fixed;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
            padding: 0;
            margin-bottom: 1rem;
        } 
         .chat-container {
           margin-top: 60px;
           height: calc(100vh - 230px);
           overflow-y: auto !important;
           position: relative;
           z-index: 1;
           padding-bottom: 100px !important;
        }
        .assistant-message {
            display: flex;
            align-items: flex-start;
            margin-left: 380px;
            margin-bottom: 20px;
            z-index:1;
        }
        .assistant-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #1cd8d2 0%, #93edc7 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 10px;
            color: #FFFFFF;
            z-index:1;
        }
        .assistant-text {
            background-color: #3A3A3A;
            padding: 10px;
            border-radius: 5px;
            max-width: 60%;
            color: #FFFFFF;
            z-index:1;
        }
        .user-message {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 20px;
            z-index:1;
        }
        .user-text {
            background-color: #444444;
            padding: 10px;
            border-radius: 5px;
            max-width: 60%;
            color: #FFFFFF;
            z-index:1;
        }
        .typing {
            color: #AAAAAA;
            font-style: italic;
            margin-left: 380px;
            margin-bottom: 10px;
            z-index:1;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            margin-left: 25%;
            width: 75%;
            background-color: #2B2B2B;
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .input-box {
            width: 80%;
            padding: 8px;
            border-radius: 4px;
            border: none;
            outline: none;
            margin-right: 10px;
            background-color: #000000 !important;
            color: #FFFFFF;
        }
        .input-box::placeholder {
            color: gray;
        }
        .send-button {
            padding: 8px 16px;
            border-radius: 4px;
            background-color: #000000;
            color: white;
            border: none;
            cursor: pointer;
        }
        .stForm {
          position: fixed !important;
          bottom: 0 !important;
          left: 25% !important;
          width: 75% !important;
          z-index: 1001 !important;
          background-color: #2B2B2B !important;
          padding: 20px !important;
          border-top: 2px solid #444 !important;
        }
        .table-container {
          margin-left: 380px;
          max-width: calc(100% - 450px);
          overflow-x: auto;
          margin-bottom: 20px;
          background-color: #3A3A3A;
          border-radius: 5px;
          padding: 10px;
        }
        .table-container table {
            width: auto;
            min-width: 100%;
            background-color: #000000 !important;
            color: white !important;
            border-color: #000000 !important;
            border-collapse: collapse;
        }
        .table-container th, .table-container td {
            background-color: #000000 !important;
            color: white !important;
            border: 1px solid #000000 !important;
            padding: 8px;
            text-align: left;
        }
        .table-container td {
            background-color: #2B2B2B !important;
            color: white !important;
            border: 1px solid #000000 !important;
            padding: 8px;
            text-align: left;
        }
        .table-container tr:nth-child(odd) td {
          background-color: #333333 !important;
        }
        .table-container tr:hover td {
          background-color: #444444 !important;
        }
        .stMarkdown {
           all: unset;
        }
        .element-container .stMarkdown {
           all: unset;
        }
        .stApp {
          padding-top: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # ----------------------------------------------------
    #                LAYOUT STRUCTURE
    # ----------------------------------------------------
    st.markdown("""
    <div class="content-wrapper">
        <div class="left-col">
        </div>
        <div class="right-col">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------- LEFT COLUMN --------------------
    left_col = st.container()
    with left_col:
        st.markdown('<div class="left-col" style="position:fixed;">', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="position:fixed;">
              <img src="{threads_icon_url}" width="24"/>
              <span style="color:gray;font-size:1rem;position:fixed;margin-left: 10px;">THREADS</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # -------------------- CHAT CONTAINER -----------------
    right_col = st.container()
    with right_col:
        st.markdown('<div class="right-col" style="position:fixed; margin-left:25%; width:75%;">', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container" id="chat-container" style="margin-top: -20cm;">', unsafe_allow_html=True)
    
            # Display existing messages
            for idx, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "assistant":
                    if isinstance(message["content"], dict):
                        text = message["content"].get("text", "")
                        table_data = message["content"].get("table_data", [])
                        # Here is your base64 from the backend's "image" field.
                        # We'll store it as "graph_base64" in the combined_content below.
                        graph_base64 = message["content"].get("graph_base64", None)
    
                        plain_text = str(text).replace('<', '&lt;').replace('>', '&gt;')
    
                        # Display the text message
                        st.markdown(
                            f"""
                            <div class="assistant-message" style="margin-top: 5px;">
                                <div class="assistant-avatar">S</div>
                                <div class="assistant-text">{plain_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
                        # Display table if present
                        if table_data and isinstance(table_data, (list, dict)):
                            try:
                                if isinstance(table_data, dict):
                                    table_data = [table_data]
                                table_data = [item for item in table_data if isinstance(item, dict)]
                                if table_data:
                                    df = pd.DataFrame(table_data)
                                    table_id = f"table_{hash(str(table_data))}"
                                    html_table = f"""
                                    <div class="table-container">
                                        <table id="{table_id}" class="display">
                                            <thead>
                                                <tr>
                                                    {"".join(f"<th>{col}</th>" for col in df.columns)}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {"".join(
                                                    f"<tr>{''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row)}</tr>"
                                                    for _, row in df.iterrows()
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                    """
                                    st.markdown(html_table, unsafe_allow_html=True)
                                    st.markdown("""
                                      <style>
                                      .stButton > button {
                                          background-color: #2B2B2B !important;  /* Dark gray background */
                                          color: #0000FF !important;            /* Blue text */
                                          border: none !important;              /* No border */
                                          border-radius: 5px;                   /* Maintains rounded edges */
                                      }
                                      </style>
                                  """, unsafe_allow_html=True)
                                    if graph_base64:
                                    
                                        graph_button_key = f"view_graph_button_{idx}"
                                        cols = st.columns([1, 4])
                                        with cols[1]:
                                          sub_cols = st.columns([1, 9])
                                          with sub_cols[1]:
                                              if st.button("View Graph", key=graph_button_key):
                                                  with st.container():
                                                      st.image(f"data:image/png;base64,{graph_base64}",width=500)
                            except Exception as e:
                                st.markdown(
                                    f"""
                                    <div class="assistant-message" style="margin-top: 5px;">
                                        <div class="assistant-avatar">S</div>
                                        <div class="assistant-text">Error rendering table: {str(e).replace('<','&lt;').replace('>','&gt;')}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
    
                    else:
                        # Just text
                        plain_content = str(message["content"]).replace('<', '&lt;').replace('>', '&gt;')
                        st.markdown(
                            f"""
                            <div class="assistant-message" style="margin-top: 5px;">
                                <div class="assistant-avatar">S</div>
                                <div class="assistant-text">{plain_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:  # user message
                    plain_content = str(message["content"]).replace('<', '&lt;').replace('>', '&gt;')
                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="user-text">{plain_content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
            # Show initial messages only once
            if not st.session_state.initial_messages_shown:
                initial_messages = [
                    "I am your Supply Chain Digital Assistant!",
                    "I work closely with Joule",
                    "Please let me know how can I help you"
                ]
                for msg in initial_messages:
                    typing_placeholder = st.empty()
                    typing_placeholder.markdown(
                        "<div class='typing' style='margin-left: 380px;position: relative; z-index: 10;'>Assistant is typing...</div>",
                        unsafe_allow_html=True
                    )
                    time.sleep(1.0)
                    typing_placeholder.empty()
                    st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                    st.markdown(
                        f"""
                        <div class="assistant-message" style="margin-top: 5px;">
                            <div class="assistant-avatar">S</div>
                            <div class="assistant-text">{msg}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.session_state.initial_messages_shown = True
    
            st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)
    
        # ----------------- USER INPUT FORM -------------------
        with st.form(key='chat_form', clear_on_submit=True):
            col1, col2 = st.columns([8, 1])
            with col1:
                user_input = st.text_input(
                    "",
                    placeholder="Start a Conversation...",
                    label_visibility="collapsed",
                    key="user_input"
                )
            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)
    
        # -------------------- HANDLE SUBMISSION --------------
        if submit_button and user_input.strip():
            with chat_container:
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.markdown(
                    f"""
                    <div class="user-message">
                        <div class="user-text">{user_input}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
                # Show typing animation
                typing_placeholder = st.empty()
                typing_placeholder.markdown(
                    "<div class='typing' style='margin-left: 380px;'>Assistant is typing...</div>",
                    unsafe_allow_html=True
                )
                time.sleep(1.0)
    
                try:
                    response = requests.post('http://localhost:5000/getdata', json={'user_question': user_input})
                    if response.status_code == 200:
                        data = response.json()
                        message = data.get('message', 'No message returned')
                        result = data.get('result', {})
                        
                        # Here is the base64 string the backend returned
                        # e.g. "data:image/png;base64,ABC..."
                        # or sometimes just the raw base64
                        image_url = data.get('image', "")  
    
                        # We remove the "data:image/png;base64," prefix if it exists (just to keep it consistent in our code)
                        # because we'll add it ourselves inside st.image(...) above
                        graph_base64 = image_url.replace("data:image/png;base64,", "")
    
                        plain_message = str(message).replace('<', '&lt;').replace('>', '&gt;')
    
                        if isinstance(result, dict):
                            result = [result]
                        elif not isinstance(result, list):
                            result = []
                        result = [item for item in result if isinstance(item, dict)]
    
                        # Store everything in session
                        combined_content = {
                            "text": plain_message,
                            "table_data": result,
                            "graph_base64": graph_base64  # <= This is key
                        }
                        st.session_state.chat_messages.append({"role": "assistant", "content": combined_content})
    
                        # Clear typing
                        typing_placeholder.empty()
    
                        # Show the assistant text
                        st.markdown(
                            f"""
                            <div class="assistant-message" style="margin-top: 5px;">
                                <div class="assistant-avatar">S</div>
                                <div class="assistant-text">{plain_message}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
                        # Show the table if present
                        if result:
                            try:
                                df = pd.DataFrame(result)
                                table_id = f"table_{hash(str(result))}"
                                html_table = f"""
                                <div class="table-container">
                                    <table id="{table_id}" class="display">
                                        <thead>
                                            <tr>
                                                {"".join(f"<th>{col}</th>" for col in df.columns)}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {"".join(
                                                f"<tr>{''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row)}</tr>"
                                                for _, row in df.iterrows()
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                                """
                                st.markdown(html_table, unsafe_allow_html=True)
                                st.markdown("""
                                      <style>
                                      .stButton > button {
                                          background-color: #2B2B2B !important;  /* Dark gray background */
                                          color: #0000FF !important;            /* Blue text */
                                          border: none !important;              /* No border */
                                          border-radius: 5px;                   /* Maintains rounded edges */
                                      }
                                      </style>
                                  """, unsafe_allow_html=True)
                                # If there's a graph, show a button that uses the newly stored base64
                                if graph_base64:
                                  cols = st.columns([1, 4])
                                  with cols[1]:
                                      sub_cols = st.columns([1, 9])
                                      with sub_cols[1]:
                                          if st.button("View Graph"):
                                              with st.container(): # Use a container to create a dedicated space
                                                  st.image(f"data:image/png;base64,{graph_base64}", width=500)
    
                            except Exception as e:
                                st.markdown(
                                    f"""
                                    <div class="assistant-message" style="margin-top: 5px;">
                                        <div class="assistant-avatar">S</div>
                                        <div class="assistant-text">Error rendering table: {str(e).replace('<','&lt;').replace('>','&gt;')}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
    
                    else:
                        typing_placeholder.empty()
                        error_message = f"Error: Received status code {response.status_code}"
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_message
                        })
                        st.markdown(
                            f"""
                            <div class="assistant-message" style="margin-top: 5px;">
                                <div class="assistant-avatar">S</div>
                                <div class="assistant-text">{error_message}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    typing_placeholder.empty()
                    error_message = f"Error connecting to API: {str(e).replace('<','&lt;').replace('>','&gt;')}"
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    st.markdown(
                        f"""
                        <div class="assistant-message" style="margin-top: 5px;">
                            <div class="assistant-avatar">S</div>
                            <div class="assistant-text">{error_message}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
            # Auto-scroll
            st.markdown(
                """
                <script>
                function scrollChatToBottom() {
                    var chatContainer = document.getElementById('chat-container');
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
                window.addEventListener('load', function() {
                    setTimeout(scrollChatToBottom, 500);
                });
                </script>
                """,
                unsafe_allow_html=True
            )
            st.rerun()


# Main app logic
def main():
    # Common header
    st.markdown(f"""
    <div class="custom-header">
        <!-- Your header HTML -->
    </div>
    """, unsafe_allow_html=True)
    
    # Display the appropriate page
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "sample":
        show_sample_page()

if __name__ == "__main__":
    main()