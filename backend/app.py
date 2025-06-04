import pandas as pd
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.responses import StreamingResponse, JSONResponse
from backend.agent_workflow import run_agent, stream_pre_agent_summary, stream_qna_response, RouteQuery, llm as agent_llm
from backend.models import QueryRequest, AgentResponse, FileInfo  
from langchain.prompts import ChatPromptTemplate
import traceback
import zipfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

data_store: dict[str, pd.DataFrame] = {}
file_info_store: dict[str, FileInfo] = {}

@app.post("/uploadfile/", response_model=FileInfo)
async def create_upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    filename = file.filename
    contents = await file.read() # Read contents once

    df = None # Initialize df

    try:
        if filename is None:
            raise HTTPException(status_code=400, detail="File has no name.")

        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            except zipfile.BadZipFile as bzf_error: # Catch specific error from openpyxl
                print(f"BadZipFile error for {filename} with openpyxl: {bzf_error}")
                # Attempt to read with xlrd as a fallback ONLY IF you suspect it might be an XLS disguised as XLSX
                # This is generally NOT a good practice for true .xlsx files.
                # For true .xlsx, the issue is likely file corruption or incorrect format.
                # Consider removing this xlrd fallback for .xlsx if it causes confusion.
                # For now, we keep it for broader compatibility testing.
                try:
                    print(f"Attempting fallback to xlrd for {filename} due to BadZipFile error.")
                    df = pd.read_excel(io.BytesIO(contents), engine='xlrd') # xlrd might read some malformed xlsx as xls
                    print(f"Successfully read {filename} with xlrd after openpyxl failed.")
                except Exception as xlrd_err:
                    print(f"Fallback to xlrd also failed for {filename}: {xlrd_err}")
                    raise HTTPException(status_code=400, detail=f"File '{filename}' appears to be corrupted or not a valid Excel file. It's not a proper ZIP archive (required for .xlsx). Error: {bzf_error}")
            except Exception as openpyxl_err: # Catch other openpyxl errors
                print(f"Other openpyxl error for {filename}: {openpyxl_err}")
                raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}' with openpyxl engine. Ensure it's a valid .xlsx file. Error: {openpyxl_err}")

        elif filename.endswith('.xls'):
            try:
                df = pd.read_excel(io.BytesIO(contents), engine='xlrd')
            except Exception as xlrd_err: # Catch xlrd specific errors
                 print(f"xlrd error for {filename}: {xlrd_err}")
                 raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}' with xlrd engine. Ensure it's a valid .xls file. Error: {xlrd_err}")
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV, XLSX, or XLS.")
        
        if df is None or df.empty : # Check if df was successfully read and is not empty
            raise HTTPException(status_code=400, detail=f"The file '{filename}' was empty or could not be parsed into a DataFrame by any available engine.")

        # ... (rest of your code: data_store, file_info_data, etc.)
        data_store[session_id] = df
        file_info_data = FileInfo(
            session_id=session_id,
            filename=filename,
            columns=df.columns.tolist(),
            df_head=df.head().to_string()
        )
        file_info_store[session_id] = file_info_data
        return file_info_data

    except pd.errors.EmptyDataError:
        print(f"Pandas EmptyDataError for file: {filename}")
        raise HTTPException(status_code=400, detail="The uploaded file is empty (pandas error).")
    except HTTPException as e:
        raise e # Re-raise our own HTTPExceptions
    except Exception as e:
        print(f"Unexpected error processing file in /uploadfile/ for {filename}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error while processing file '{filename}'.")
    finally:
        if file and hasattr(file, 'file') and not file.file.closed:
            await file.close()
            
@app.post("/process_query/") # Will always return StreamingResponse now
async def process_query(request: QueryRequest): # Return type annotation removed for StreamingResponse flexibility
    session_id = request.session_id
    user_query = request.query

    if session_id not in data_store or session_id not in file_info_store:
        # This error should also be streamed if we are to be consistent
        async def error_stream_gen():
            yield json.dumps({"type": "error", "content": "Session ID not found or data not uploaded."}) + "\n"
        return StreamingResponse(error_stream_gen(), media_type="application/x-ndjson", status_code=404)


    df = data_store[session_id]
    current_file_info = file_info_store[session_id]
    
    action_type_for_routing = "query_data" # Default
    try:
        if agent_llm:
            # ... (FastAPI router logic as before to determine route_result.action)
            router_prompt_template_messages = [
                ("system", """You are an expert query router. Based on the user's query, dataset columns, and a sample of the data, decide the primary action to take.
                The available actions are:
                - 'visualize': If the user is asking for a chart, graph, plot, or any kind of visual representation of the data.
                - 'query_data': If the user is asking a direct question about the data that can likely be answered by looking at the sample or general data properties, or requires a textual summary/calculation.
                - 'fallback': If the query is conversational, a greeting, off-topic, or too complex for direct visualization or simple Q&A from the sample.

                Dataset columns: {columns}
                Data sample (first 5 rows):
                {df_head}
                User query: {user_query}
                Provide your routing decision and a brief reasoning.
                """),
                ("human", "{user_query}")
            ]
            router_prompt = ChatPromptTemplate.from_messages(router_prompt_template_messages)
            structured_router = agent_llm.with_structured_output(RouteQuery, method="function_calling", include_raw=False)
            router_chain = router_prompt | structured_router
            route_result: RouteQuery = router_chain.invoke({
                "columns": current_file_info.columns, "df_head": current_file_info.df_head, "user_query": user_query
            })
            action_type_for_routing = route_result.action
            print(f"FastAPI Router decision: {action_type_for_routing}, Reasoning: {route_result.reasoning}")
        else:
            print("FastAPI Router: LLM not available, making broad assumption for query type.")
            if any(kw in user_query.lower() for kw in ["plot", "chart", "graph", "visualize", "show me"]):
                action_type_for_routing = "visualize"

    except Exception as e:
        print(f"Error in FastAPI router invocation: {e}")
        action_type_for_routing = "fallback"


    async def combined_stream_generator():
        # 1. Stream the initial "plan" or "pre-summary"
        # This summary is generated BEFORE the main agent graph for "visualize" or "fallback"
        # For "query_data", the stream_qna_response itself is the summary and content.
        if action_type_for_routing != "query_data": # Only stream pre-summary if not direct Q&A
            yield json.dumps({"type": "system", "message": f"Router decided action: {action_type_for_routing}. Starting pre-summary stream..."}) + "\n"
            async for summary_chunk in stream_pre_agent_summary(
                user_query, current_file_info.df_head, current_file_info.columns, action_type_for_routing
            ):
                yield summary_chunk # Already formatted as JSON string with newline
            yield json.dumps({"type": "system", "message": "Pre-summary stream ended."}) + "\n"

        # 2. Execute the main action and stream its results
        if action_type_for_routing == "query_data":
            print("Streaming Q&A response from FastAPI...")
            yield json.dumps({"type": "system", "message": "Starting Q&A stream..."}) + "\n"
            async for content_chunk in stream_qna_response(
                user_query, current_file_info.df_head, current_file_info.columns
            ):
                yield content_chunk # Already formatted as JSON string with newline
            yield json.dumps({"type": "system", "message": "Q&A stream ended."}) + "\n"
        
        else: # "visualize" or "fallback" -> call the main agent graph
            print(f"Processing '{action_type_for_routing}' using full agent graph...")
            try:
                agent_result_dict = run_agent(user_query=user_query, df=df)
                # Send the entire agent_result_dict as a single JSON object in the stream
                yield json.dumps({"type": "final_agent_response", "data": agent_result_dict}) + "\n"
            except Exception as e:
                print(f"Error during agent processing for {action_type_for_routing}: {e}")
                traceback.print_exc()
                error_response = AgentResponse(
                    response_type="error",
                    content=f"An unexpected error occurred in the agent: {str(e)}",
                    error=str(e),
                    thinking_log_str="Error during agent execution."
                ).model_dump() # Use .model_dump() for Pydantic v2
                yield json.dumps({"type": "final_agent_response", "data": error_response }) + "\n" # send error as final_agent_response
        
        yield json.dumps({"type": "system", "message": "Full processing stream ended."}) + "\n"

    return StreamingResponse(combined_stream_generator(), media_type="application/x-ndjson")
