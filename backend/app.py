# backend/app.py

import pandas as pd
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request # Removed Form as it was only for /upload_screencast/
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.responses import JSONResponse, StreamingResponse
# Assuming your agent_workflow and models are correctly set up
# For a minimal example, stubs are used. Replace with your actual imports.
try:
    from backend.agent_workflow import run_agent, stream_pre_agent_summary, stream_qna_response, RouteQuery, llm as agent_llm
    from langchain.prompts import ChatPromptTemplate # If used by router logic
except ImportError:
    print("Warning: agent_workflow or langchain.prompts not fully found. Using stubs for agent logic.")
    class RouteQuery: pass # Stub
    agent_llm = None # Stub
    async def stream_pre_agent_summary(*args, **kwargs): yield json.dumps({"type":"system", "message":"Pre-summary stub"}) + "\n" # Stub
    async def stream_qna_response(*args, **kwargs): yield json.dumps({"type":"content", "chunk":"Q&A stub"}) + "\n" # Stub
    def run_agent(*args, **kwargs): return {"response_type":"fallback", "content":"Agent stub", "error":None, "thinking_log_str": "Stub run"} # Stub
    ChatPromptTemplate = None # Stub


from backend.models import QueryRequest, AgentResponse, FileInfo

import traceback
import zipfile
import os
# from datetime import datetime # Not strictly needed if not using for screencast timestamps on server
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Agent Backend", version="1.0.1") # Minor version bump for clarity

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Exception Handlers ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred on the server.", "error_type": type(exc).__name__},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(log_level, f"HTTPException for request {request.method} {request.url}: Status {exc.status_code}, Detail: {exc.detail}", exc_info=(exc.status_code >= 500))
    content = {"detail": exc.detail, "error_type": "HTTPException"}
    return JSONResponse(status_code=exc.status_code, content=content)

# --- Data Stores ---
data_store: dict[str, pd.DataFrame] = {}
file_info_store: dict[str, FileInfo] = {}

# SCREENCAST_DIR and its creation are REMOVED as screencasts are not saved on the server.

# --- Endpoints ---
@app.get("/", tags=["General"])
async def read_root():
    logger.info("Root path '/' accessed.")
    return {"message": "Welcome to the Data Agent Backend API! Screencast upload endpoint removed (client-side download only)."}

@app.post("/uploadfile/", response_model=FileInfo, tags=["Data File"])
async def create_upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    filename = file.filename
    logger.info(f"Received file upload request for: '{filename}' (Session to be: {session_id})")

    if not filename:
        logger.error("File upload attempt with no filename.")
        raise HTTPException(status_code=400, detail="File has no name.")

    contents = await file.read()
    if not contents:
        logger.error(f"Uploaded file '{filename}' is empty.")
        raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")

    df = None
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            except zipfile.BadZipFile as bzf_error:
                logger.warning(f"BadZipFile error for '{filename}' with openpyxl: {bzf_error}. Trying xlrd.")
                df = pd.read_excel(io.BytesIO(contents), engine='xlrd')
            except Exception as openpyxl_err:
                logger.error(f"Openpyxl error for '{filename}': {openpyxl_err}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}'. Ensure it's a valid .xlsx file.")
        elif filename.endswith('.xls'):
            try:
                df = pd.read_excel(io.BytesIO(contents), engine='xlrd')
            except Exception as xlrd_err:
                logger.error(f"xlrd error for '{filename}': {xlrd_err}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}'. Ensure it's a valid .xls file.")
        else:
            logger.error(f"Invalid file type uploaded: {filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV, XLSX, or XLS.")

        if df is None or df.empty:
            logger.error(f"File '{filename}' resulted in an empty DataFrame after parsing.")
            raise HTTPException(status_code=400, detail=f"The file '{filename}' was empty or could not be parsed correctly.")

        data_store[session_id] = df
        file_info_data = FileInfo(
            session_id=session_id,
            filename=str(filename),
            columns=df.columns.tolist(),
            df_head=df.head().to_string()
        )
        file_info_store[session_id] = file_info_data
        logger.info(f"File '{filename}' processed successfully. Session ID: {session_id}")
        return file_info_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_upload_file for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during file processing.")


@app.post("/process_query/", tags=["Agent"])
async def process_query(request: QueryRequest):
    session_id = request.session_id
    user_query = request.query
    logger.info(f"Processing query for session '{session_id}': '{user_query[:100]}...'")

    if session_id not in data_store or session_id not in file_info_store:
        logger.warning(f"Session ID '{session_id}' not found for process_query.")
        async def error_stream_gen_404():
            yield json.dumps({"type": "error", "content": "Session ID not found or data not uploaded."}) + "\n"
        return StreamingResponse(error_stream_gen_404(), media_type="application/x-ndjson", status_code=404)

    df = data_store[session_id]
    current_file_info = file_info_store[session_id]
    action_type_for_routing = "query_data"

    try:
        if agent_llm and ChatPromptTemplate and RouteQuery: # Check if stubs are replaced
            router_prompt_template_messages = [
                ("system", "You are an expert query router... Columns: {columns}, Head: {df_head}, Query: {user_query}"),
                ("human", "{user_query}")
            ]
            router_prompt = ChatPromptTemplate.from_messages(router_prompt_template_messages)
            structured_router = agent_llm.with_structured_output(RouteQuery, method="function_calling", include_raw=False) # type: ignore
            router_chain = router_prompt | structured_router
            route_result = router_chain.invoke({ # type: ignore
                "columns": current_file_info.columns, "df_head": current_file_info.df_head, "user_query": user_query
            })
            action_type_for_routing = route_result.action # type: ignore
            logger.info(f"Router decision: {action_type_for_routing}, Reasoning: {route_result.reasoning if hasattr(route_result, 'reasoning') else 'N/A'}") # type: ignore
        else:
            logger.warning("LLM/Router components not fully available. Using basic keyword matching for routing.")
            if any(kw in user_query.lower() for kw in ["plot", "chart", "graph", "visualize", "show me"]):
                action_type_for_routing = "visualize"
    except Exception as e_router:
        logger.error(f"Error in query routing for session {session_id}: {e_router}", exc_info=True)
        action_type_for_routing = "fallback"

    async def combined_stream_generator():
        try:
            if action_type_for_routing != "query_data":
                yield json.dumps({"type": "system", "message": f"Router decided action: {action_type_for_routing}. Starting pre-summary stream..."}) + "\n"
                async for summary_chunk in stream_pre_agent_summary(user_query, current_file_info.df_head, current_file_info.columns, action_type_for_routing):
                    yield summary_chunk
                yield json.dumps({"type": "system", "message": "Pre-summary stream ended."}) + "\n"

            if action_type_for_routing == "query_data":
                logger.info(f"Streaming Q&A response for session '{session_id}'.")
                yield json.dumps({"type": "system", "message": "Starting Q&A stream..."}) + "\n"
                async for content_chunk in stream_qna_response(user_query, current_file_info.df_head, current_file_info.columns):
                    yield content_chunk
                yield json.dumps({"type": "system", "message": "Q&A stream ended."}) + "\n"
            else:
                logger.info(f"Running full agent for '{action_type_for_routing}' on session '{session_id}'.")
                agent_result_dict = run_agent(user_query=user_query, df=df)
                yield json.dumps({"type": "final_agent_response", "data": agent_result_dict}) + "\n"
        except Exception as e_stream_gen:
            logger.error(f"Error during agent response generation/streaming for session '{session_id}': {e_stream_gen}", exc_info=True)
            error_payload = {
                "response_type": "error",
                "content": f"An internal error occurred during agent processing: {str(e_stream_gen)}",
                "error": str(e_stream_gen)
            }
            yield json.dumps({"type": "final_agent_response", "data": error_payload}) + "\n"
        finally:
            logger.info(f"Full processing stream ended for session '{session_id}'.")
            yield json.dumps({"type": "system", "message": "Full processing stream ended."}) + "\n"
    return StreamingResponse(combined_stream_generator(), media_type="application/x-ndjson")


# Endpoint /upload_screencast/ is intentionally REMOVED.
# The screencast download functionality is handled entirely client-side by the
# Streamlit component using the 'simple-screen-recorder' NPM package.

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn dev server for Data Agent Backend on http://0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)