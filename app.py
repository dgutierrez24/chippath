import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.adapters.openai import convert_openai_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END

import re
import json
import time
import functools
from typing import Annotated, Sequence, TypedDict
import operator

# Load environment variables
load_dotenv()

# Set up API keys using environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("API keys are missing. Please set OPENAI_API_KEY and TAVILY_API_KEY as environment variables.")
    st.stop()

# Use the API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Initialize LLM and embedding model
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Tavily tool
tavily_tool = TavilySearchResults(max_results=5)

def display_message(mess):
    st.caption(mess)
    #st.subheader(mess, divider=True)

# Global variables for retrievers
qdrant_retriever_resume = None
qdrant_retriever_semiconductor = None

semiconductor_directory ='/Users/danielgutierrez/Jupyter_pad/new_final/semi_reports'
resume_directory ='/Users/danielgutierrez/Jupyter_pad/new_final/resume'

def dict_to_flat_string(input_data):
    """
    Convert a dictionary to a flat string. Each key-value pair is converted
    to 'key: value' format and combined into a single string.

    Args:
    input_data (dict): The dictionary to convert.

    Returns:
    str: A flat string representation of the dictionary.
    """
    if isinstance(input_data, dict):
        # Create a list of 'key: value' strings for each item in the dictionary
        items = [f"{key}: {value}" for key, value in input_data.items()]
        # Join all items into a single string, separated by ', '
        return ', '.join(items)
    elif isinstance(input_data, str):
        # Return string as is
        return input_data
    else:
        # Optionally handle other types, or raise an error
        raise TypeError("Input must be a dictionary or a string")


def load_and_split_pdf(directory, chunk_size=1000, chunk_overlap=25):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyMuPDFLoader(file_path)
            pdf_documents = loader.load()

            for page_num, page in enumerate(pdf_documents):
                splits = text_splitter.split_text(page.page_content)
                for i, split in enumerate(splits):
                    documents.append(Document(
                        page_content=split,
                        metadata={
                            "filename": filename,
                            "page_number": page_num + 1,
                            "chunk_number": i + 1
                        }
                    ))
    return documents

@tool
def Semi_Job_Researcher(state):

    """Use this to use Tavily and LLM to write a report on Semi Job/Career Research"""
    print('-> Calling Semi_Job_Researcher->')
    display_message("-> Calling Semi_Job_Researcher->")

    print(state)
    question = state
    print('Question:', question)
    
    query = dict_to_flat_string(question)
    print(query)

    content = tavily_tool(query)
    
    prompt = [{
        "role": "system",
        "content": f'You are a great job and career advice counselor who specializes in the Semiconductor Industry. '
                   f'Your sole purpose is to write a well written summary about the given job or career of interest within the Semiconductor industry.'
                   f'Include the following in the report: Job Summary, Education Requirements, Learning Resources, Companies Hiring, Salary Range in dollars, and Job Growth'
                   f'Give as much detail as possible in the report'
    }, {
        "role": "user",
        "content": f'Information: """{content}"""\n\n'
                   f'Using the above information, answer the following'
                   f'query: "{query}" in a detailed report about the semiconductor industry --'
                   f'Please use MLA format and markdown syntax.'
    }]
    
    lc_messages = convert_openai_messages(prompt)
    result = ChatOpenAI(model='gpt-4').invoke(lc_messages).content
    return result

@tool
def RAG_Semi(state): 
    """Use this to execute RAG. If the question is related to the state and outlook of the Semiconductor Industry, using this tool retrieve the results."""
    print('-> Calling RAG_Semi ->')
    display_message("-> Calling RAG_Semi ->")
    question = state
    print('Question:', question)
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Be as detailed as possible and feel free to elaborate from the context.  
    The header should be the concise version of the question.
    Please use MLA format, markdown syntax and bullet points to make a clear report.
    Also show where you got the context from.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    retrieval_chain = (
        {"context": qdrant_retriever_semiconductor, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = retrieval_chain.invoke(question)
    return result


@tool 
def RAG_Resume(state): 

    """Use this to execute RAG. If the question is related to my resume, using this tool retrieve the results."""
    print('-> Calling RAG_Resume ->')
    display_message("-> Calling RAG_Resume ->")
    #question = state
    question = "analyze my resume"
    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question} 

    Always output a detailed summary of the Resume.

    Extract any skills, education, and job titles mentioned.
    
    Here is the Final Answer Format:
    1. Resume Summary: summary
    2. Skills: list of skills
    3. Education: list of education
    4. Job Title: list of job titles

    Using the above information, what jobs can the user apply for in the semiconductor industry (some companies hiring may include: Nvidia, Intel, AMD, Marvell, Texas Instruments...)? Please provide a detailed summary.
    
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain1 = (
        {"context": qdrant_retriever_resume, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
     
    result = retrieval_chain1.invoke(question)

    
    
    # Extract information for JSON
    resume_summary = re.search(r'1\. Resume Summary: (.*?)(?=2\.)', result, re.DOTALL)
    skills = re.search(r'2\. Skills: (.*?)(?=3\.)', result, re.DOTALL)
    education = re.search(r'3\. Education: (.*?)(?=4\.)', result, re.DOTALL)
    job_title = re.search(r'4\. Job Title: (.*?)(?=Using)', result, re.DOTALL)
    
    json_output = {
        "Resume Summary": resume_summary.group(1).strip() if resume_summary else "",
        "Skills": [skill.strip() for skill in skills.group(1).split(',')] if skills else [],
        "Education": [edu.strip() for edu in education.group(1).split(',')] if education else [],
        "Job Title": [title.strip() for title in job_title.group(1).split(',')] if job_title else []
    }
    print(result)
    print(json_output)


    # Extract and format the 'Skills' list into a single string
    skills_string = ', '.join(json_output['Skills'])

    # Extract and format the 'Education' list into a single string
    education_string = ', '.join(json_output['Education'])

    # Extract and format the 'Education' list into a single string
    job_string = ', '.join(json_output['Job Title'])

    # Combine both into one string
    combined_string = f"Summary: {result}. Skills: {skills_string}. Education: {education_string}. Title: {job_string}."

    print(combined_string)

    return combined_string
    # return {
    #     "text_output": result,
    #     "json_output": combined_string
    # }


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Setsup WorkFlow Graph
def setup_workflow():

    members = ["RAG_Semi", "RAG_Resume", "Semi_Job_Researcher"]
    
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Use RAG_Semi tool when questions "
        "are related to the outlook or State of the US Semiconductor Industry or asking about how the semiconductor industry is doing."
        "Use Semi_Job_Researcher when questions are related to careers or jobs interest or advice."
        "Use RAG_Resume tool when questions are related a resume or analysis of a resume, or resume guidance. "
        " Each worker will perform a task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    
    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": options,
                }
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    
    supervisor_chain = (
        prompt
        | llm.bind(functions=[function_def], function_call={"name": "route"})
        | JsonOutputFunctionsParser()
    )
    
    # SEMI JOB RESEARCH AGENT
    research_agent = create_agent(llm, [Semi_Job_Researcher], "You are a Semiconductor Career and Job Researcher")
    semi_job_research_node = functools.partial(agent_node, agent=research_agent, name="Semi_Job_Researcher")
    
    # RAG_SEMI AGENT
    RAG_Semi_agent = create_agent(llm, [RAG_Semi], "Use this tools when questions are related to the State or Outlook of the Semiconductor Industry")
    rag_semi_node = functools.partial(agent_node, agent=RAG_Semi_agent, name="RAG_Semi")
    
    # RAG_RESUME AGENT
    RAG_Resume_agent = create_agent(llm, [RAG_Resume], "Use this tools when questions are related to uploaded resume")
    rag_resume_node = functools.partial(agent_node, agent=RAG_Resume_agent, name="RAG_Resume")

    # JOB_SEARCHER AGENT
    Job_Searcher_agent = create_agent(llm,[tavily_tool],"Use this tool to search for jobs based off resume",)
    job_searcher_node = functools.partial(agent_node, agent=Job_Searcher_agent, name="Job_Searcher")
    
    workflow = StateGraph(AgentState)
    workflow.add_node("Semi_Job_Researcher", semi_job_research_node)
    workflow.add_node("RAG_Semi", rag_semi_node)
    workflow.add_node("RAG_Resume", rag_resume_node)
    workflow.add_node("supervisor", supervisor_chain)
    workflow.add_node("Job_Searcher", job_searcher_node)

    # Define edges
    workflow.add_edge('RAG_Resume', 'Job_Searcher')
    workflow.add_edge('RAG_Resume', 'supervisor')
    workflow.add_edge('Job_Searcher', 'supervisor')
    workflow.add_edge('Semi_Job_Researcher', 'supervisor')
    workflow.add_edge('RAG_Semi', 'supervisor')
    
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    
    workflow.set_entry_point("supervisor")

    # graph = workflow.compile()

    # from IPython.display import Image, display

    # try:
    #     # Generate the image data (in binary format, like PNG)
    #     image_data = graph.get_graph(xray=True).draw_mermaid_png()

    #     # Define the file name where the image will be saved
    #     image_filename = "graph_image.png"

    #     # Save the image to a file
    #     with open(image_filename, "wb") as f:
    #         f.write(image_data)

    #     # Display the image (optional, especially useful in Jupyter notebooks)
    #     display(Image(image_filename))

    #     print(f"Image saved as {image_filename}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    return workflow.compile()

def process_pdf(file):
    with st.spinner(f"Processing {file.name}..."):
        pdf_content = file.read()
        with open(file.name, "wb") as f:
            f.write(pdf_content)
        loader = PyMuPDFLoader(file.name)
        documents = loader.load()
        os.remove(file.name)  # Clean up the temporary file
    return documents

def main():
    st.set_page_config(
        page_title="CHIPpath", page_icon=":robot_face:",
)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("chip.png")

    with col3:
        st.write(' ')

    st.markdown("<h1 style='text-align: center; color: green;'>CHIPpath:</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Uncover your future in the Semiconductor Industry</h1>", unsafe_allow_html=True)

    st.text("-----------------------------------------------------------------------------------------------------------------")
    st.markdown(" :blue[Mission Statement:] To foster economic opportunity and drive the microelectronics  ")
    st.markdown(" industry’s growth, we create pathways and opportunities for job seekers and provide  ")
    st.markdown(" tools and systems for semiconductor companies to attract, develop, retain, and advance")
    st.markdown(" a diverse and skilled workforce. and skilled workforce.")
    st.text("-----------------------------------------------------------------------------------------------------------------")

    # progress_text = "Operation in progress. Please wait."
    # my_bar = st.progress(0, text=progress_text)

    with st.sidebar:
        st.markdown("***Application Instructions:***")
        st.text("-----------------------------------------------------------------------------------------------------------------")
        st.markdown("First Step :page_facing_up:")

        resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
        if resume_file is not None:
            progress = st.progress(10)
            # Process the file
            count = 0
            while count < 3:
                    count += 1
                    time.sleep(1)
            progress.progress(100)  # Complete the progress bar
            st.success('File processing complete!')
        st.text("-----------------------------------------------------------------------------------------------------------------")
        st.markdown("**Three options to interact with Agents**")
        st.markdown("1. :page_with_curl:  Ask to analyze Resume for Job recommendations")
        st.markdown("2. :student:  Ask about a specific job of interest e.g. lithography engineer")
        st.markdown("3. :books:  Ask about the current state of US Semiconductor Industry (2024 reports)")
        st.text("-----------------------------------------------------------------------------------------------------------------")
    
    # Display a warning if running locally without API keys
    if not openai_api_key or not tavily_api_key:
        st.warning("API keys are not set. The app may not function correctly.")
    
    # File uploaders
    # resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    #semiconductor_files = st.file_uploader("Upload semiconductor industry reports (PDF)", type="pdf", accept_multiple_files=True)

    if resume_file:
        # Process uploaded files
        resume_docs = process_pdf(resume_file)
       
        
        # Create vector stores
        resume_vectorstore = Qdrant.from_documents(resume_docs, embedding_model, location=":memory:", collection_name="User Resume")
        # resume_vectorstore = Qdrant.from_documents(load_and_split_pdf(resume_directory), embedding_model, location=":memory:", collection_name="User Resume")
        semiconductor_vectorstore = Qdrant.from_documents(load_and_split_pdf(semiconductor_directory), embedding_model, location=":memory:", collection_name="Outlook of Semiconductor Industry PDFs")
        
        # Set up retrievers
        global qdrant_retriever_resume, qdrant_retriever_semiconductor
        qdrant_retriever_semiconductor = semiconductor_vectorstore.as_retriever(collection_name='Outlook of Semiconductor Industry PDFs')
        qdrant_retriever_resume = resume_vectorstore.as_retriever(collection_name='User Resume')
        
        
        # Set up the workflow
        graph = setup_workflow()
        
        # User input
        user_input = st.text_input("Ask a question about the semiconductor industry:", "Lithography engineer")



        if user_input:
            with st.spinner("Processing your question..."):
                result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

            st.write("Message:")
            
            if isinstance(result, dict) and 'messages' in result:
                for index, message in enumerate(result['messages']):
                    if isinstance(message.content, str):
                        st.markdown(message.content)
                        # st.download_button(
                        #     label="Download Results",
                        #     data=message.content,
                        #     file_name=f"detailed_analysis_{index}.txt",
                        #     mime="text/plain",
                        #     key=f"download_button_{index}"  # Ensuring each button has a unique key
                        # )
            
            # # Handle the result correctly
            # if isinstance(result, dict) and 'messages' in result:
            #     for message in result['messages']:
            #         print(message)
            #         if isinstance(message, HumanMessage):
            #             print("done")
            #             st.markdown(message.content)
                        
            #             # # Check if the message content is a dictionary and has a 'json_output' key
            #             # if isinstance(message.content, dict) and 'json_output' in message.content:
            #             #     st.write("Resume Information:")
            #             #     st.json(message.content['json_output'])


                        
                        
            else:
                st.write("Unexpected result format. Please check the LangGraph output structure.")
        

if __name__ == "__main__":
    main()
