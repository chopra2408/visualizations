# backend/app.py
import sys
import asyncio
if sys.platform == "win32" and sys.version_info >= (3, 8):
    try:
        current_policy = asyncio.get_event_loop_policy()
        if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("INFO: Switched asyncio event loop policy to WindowsSelectorEventLoopPolicy.")
    except Exception as e_policy:
        print(f"WARNING: Could not forcefully set asyncio event loop policy: {e_policy}")
elif sys.platform == "win32": # For Python < 3.8
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("INFO: Set asyncio event loop policy to WindowsSelectorEventLoopPolicy (Python < 3.8 method).")

import pandas as pd
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from backend.models import FileInfo, QueryRequest

try:
    from backend.agent_workflow import (
        run_agent, 
        stream_pre_agent_summary, 
        stream_qna_response, 
        RouteQuery, 
        openai_client as agent_openai_client,  
        _convert_lc_messages_to_openai_format,  
        get_structured_output_from_openai  
    )
    from langchain_core.prompts import ChatPromptTemplate  
    if agent_openai_client is None:
        print("WARNING: agent_workflow.openai_client is None. Agent features might be limited.")
        agent_llm_for_router = None 
    else:
        agent_llm_for_router = agent_openai_client  

except ImportError as e:
    print(f"WARNING: Could not import from agent_workflow or langchain_core.prompts: {e}. Using stubs for agent logic.")
    class RouteQuery(BaseModel): action: str; reasoning: Optional[str] = None
    agent_openai_client = None
    agent_llm_for_router = None
    ChatPromptTemplate = None
    _convert_lc_messages_to_openai_format = lambda x: []
    async def get_structured_output_from_openai(*args, **kwargs): return None, []
    async def stream_pre_agent_summary(*args, **kwargs): yield json.dumps({"type":"system", "chunk":"Pre-summary stub"}) + "\n"
    async def stream_qna_response(*args, **kwargs): yield json.dumps({"type":"content", "chunk":"Q&A stub"}) + "\n"
    async def run_agent(*args, **kwargs): return {"response_type":"fallback", "content":"Agent stub", "error":None, "thinking_log_str": "Stub run"}



import traceback
import zipfile
import os
import logging
import shutil
from pathlib import Path
# from starlette.concurrency import run_in_threadpool # Only if truly needed for blocking calls

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Agent Backend", version="1.2.0 (Async Agent)")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
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
    if hasattr(exc, "headers"):
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)
    return JSONResponse(status_code=exc.status_code, content=content)

# --- Data Stores ---
data_store: dict[str, pd.DataFrame] = {}
file_info_store: dict[str, FileInfo] = {}

# --- Directory for Storing Client-Side Converted Screencasts (Optional) ---
UPLOADED_SCREENCASTS_DIR = Path(__file__).parent / "uploaded_screencasts"
UPLOADED_SCREENCASTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Directory for (optionally) storing client-converted screencasts: {UPLOADED_SCREENCASTS_DIR.resolve()}")


# --- Screencast Endpoints (Simplified for Client-Side Conversion) ---
@app.post("/screencast/upload/", tags=["Screencast (Client-Converted)"])
async def upload_client_converted_screencast(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    logger.info(f"Received client-converted screencast upload. Session: {session_id}, Filename: {file.filename}")
    if not file.filename:
        logger.warning("Client-converted screencast upload attempt with no filename.")
        raise HTTPException(status_code=400, detail="Uploaded screencast file has no name.")
    original_stem = Path(file.filename).stem
    original_ext = Path(file.filename).suffix or ".bin"
    unique_id = uuid.uuid4().hex[:8]
    stored_filename = f"{original_stem}_{session_id}_{unique_id}{original_ext}"
    stored_file_path = UPLOADED_SCREENCASTS_DIR / stored_filename
    try:
        with stored_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Client-converted screencast '{file.filename}' saved as '{stored_filename}'")
        return JSONResponse(
            status_code=201,
            content={
                "message": "Client-converted file uploaded successfully.",
                "original_filename": file.filename,
                "uploaded_filename_on_server": stored_filename,
                "download_url_relative_path": f"/screencast/download/{stored_filename}"
            }
        )
    except Exception as e:
        logger.error(f"Error saving client-converted screencast: {e}", exc_info=True)
        if stored_file_path.exists():
            try: os.remove(stored_file_path)
            except Exception as e_clean: logger.error(f"Failed to cleanup {stored_file_path} on error: {e_clean}")
        raise HTTPException(status_code=500, detail=f"Server error saving uploaded screencast: {str(e)}")
    finally:
        await file.close()

@app.get("/screencast/download/{filename}", tags=["Screencast (Client-Converted)"])
async def download_client_converted_screencast(filename: str):
    logger.info(f"Download requested for client-converted screencast: {filename}")
    safe_filename = Path(filename).name
    file_path = UPLOADED_SCREENCASTS_DIR / safe_filename
    if not UPLOADED_SCREENCASTS_DIR.resolve() in file_path.resolve().parents:
        logger.error(f"Attempted download outside of designated screencast dir: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename for download.")
    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Client-converted screencast '{safe_filename}' not found. Path: {file_path}")
        raise HTTPException(status_code=404, detail="File not found or was already cleaned up.")
    media_type_map = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm"}
    file_ext = file_path.suffix.lower()
    media_type = media_type_map.get(file_ext, "application/octet-stream")
    logger.info(f"Serving file {file_path} (media type: {media_type}) for download.")
    return FileResponse(path=str(file_path), filename=safe_filename, media_type=media_type)

@app.post("/screencast/cleanup_upload/{filename}", tags=["Screencast (Client-Converted)"])
async def cleanup_uploaded_screencast_file(filename: str, background_tasks: BackgroundTasks):
    logger.info(f"Explicit cleanup requested for uploaded screencast: {filename}")
    safe_filename = Path(filename).name
    file_path = UPLOADED_SCREENCASTS_DIR / safe_filename
    if not UPLOADED_SCREENCASTS_DIR.resolve() in file_path.resolve().parents:
        logger.error(f"Attempted cleanup outside of designated screencast dir: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename for cleanup.")
    if file_path.exists() and file_path.is_file():
        logger.info(f"Scheduling cleanup for uploaded file: {file_path.name}")
        background_tasks.add_task(os.remove, file_path)
        return JSONResponse(status_code=200, content={"message": f"Cleanup initiated for {safe_filename}."})
    else:
        logger.warning(f"Uploaded file '{safe_filename}' not found for explicit cleanup. Path: {file_path}")
        return JSONResponse(status_code=200, content={"message": "File not found or already cleaned up."})

# --- Existing Endpoints (Data File and Agent Processing) ---
@app.get("/", tags=["General"])
async def read_root():
    logger.info("Root path '/' accessed.")
    return {"message": "Welcome to the Data Agent Backend API! (Async Agent)"}

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
        elif filename.endswith(('.xlsx', '.xls')):
            excel_engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
            file_ext_display = Path(filename).suffix
            try:
                df = pd.read_excel(io.BytesIO(contents), engine=excel_engine)
            except zipfile.BadZipFile as bzf_error:
                # ... (existing BadZipFile handling logic) ...
                if excel_engine == 'openpyxl':
                    logger.warning(f"BadZipFile error for '{filename}' with openpyxl: {bzf_error}. Trying xlrd as fallback.")
                    try:
                        contents_io = io.BytesIO(contents); contents_io.seek(0)
                        df = pd.read_excel(contents_io, engine='xlrd')
                    except Exception as xlrd_fallback_err:
                        logger.error(f"xlrd fallback for .xlsx '{filename}' also failed: {xlrd_fallback_err}", exc_info=True)
                        raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}'. File might be corrupted or unsupported.")
                else:
                    logger.error(f"BadZipFile error for .xls '{filename}' with xlrd: {bzf_error}", exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}'. File might be corrupted.")

            except Exception as excel_err:
                logger.error(f"Error reading Excel file '{filename}' with {excel_engine}: {excel_err}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Error reading Excel file '{filename}'. Ensure it's a valid {file_ext_display} file.")
        else:
            file_ext = Path(filename).suffix
            logger.error(f"Invalid file type uploaded: {filename} (extension: {file_ext})")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or Excel (XLSX, XLS).")
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
            error_payload = {"type": "error", "content": "Session ID not found or data not uploaded. Please upload a file first."}
            yield json.dumps(error_payload) + "\n"
        return StreamingResponse(error_stream_gen_404(), media_type="application/x-ndjson", status_code=404)

    df = data_store[session_id]
    current_file_info = file_info_store[session_id]
    action_type_for_routing = "query_data" # Default

    try:
        # Using the direct get_structured_output_from_openai from agent_workflow
        if agent_openai_client and ChatPromptTemplate and RouteQuery and get_structured_output_from_openai:
            # This router logic is now simplified to directly use the helper from agent_workflow
            # This matches how agent_workflow.py's router_node works.
            router_prompt_spec = ChatPromptTemplate.from_messages([
                ("system", """You are an expert query router. Based on the user's query, data columns, and a sample of the data, determine the primary action required.
                The available actions are: 'visualize', 'query_data', or 'fallback'.
                Data Context: Columns: {router_df_columns}, Data Sample (head): {router_df_head}
                User Query: "{router_user_query}"
                Provide your routing decision and a brief reasoning.
                """),
                ("human", "Determine the action: 'visualize', 'query_data', or 'fallback'.")
            ])
            router_prompt_input = {
                "router_df_columns": current_file_info.columns,
                "router_df_head": current_file_info.df_head,
                "router_user_query": user_query
            }
            # The thinking_log for this specific router call in app.py isn't strictly necessary
            # unless you want to capture it separately here.
            app_router_log = [f"--- App.py Router for Session {session_id} ---"]
            
            route_result_instance, _ = await get_structured_output_from_openai(
                RouteQuery, router_prompt_spec, router_prompt_input, app_router_log, "APP_ROUTER"
            )

            if route_result_instance:
                action_type_for_routing = route_result_instance.action
                reasoning = getattr(route_result_instance, 'reasoning', 'N/A')
                logger.info(f"Router decision (app.py): {action_type_for_routing}, Reasoning: {reasoning}")
            else:
                logger.warning("App.py Router: get_structured_output_from_openai returned None. Falling back.")
                action_type_for_routing = "fallback_due_to_router_failure"
        else:
            logger.warning("LLM/Router components (agent_openai_client, ChatPromptTemplate, RouteQuery, get_structured_output_from_openai) not fully available. Using basic keyword matching for routing in app.py.")
            if any(kw in user_query.lower() for kw in ["plot", "chart", "graph", "visualize", "show me", "draw"]):
                action_type_for_routing = "visualize"
    except Exception as e_router:
        logger.error(f"Error in app.py query routing for session {session_id}: {e_router}", exc_info=True)
        action_type_for_routing = "fallback_due_to_router_error"


    async def combined_stream_generator():
        try:
            yield json.dumps({"type": "system", "message": f"Router decided action: {action_type_for_routing}."}) + "\n"
            
            if action_type_for_routing in ["fallback_due_to_router_error", "fallback_due_to_router_failure"]:
                fallback_payload = {
                    "response_type": "error",
                    "content": "There was an issue determining how to handle your query. Please try rephrasing.",
                    "error": "Query routing failed internally."}
                yield json.dumps({"type": "final_agent_response", "data": fallback_payload}) + "\n"
            elif action_type_for_routing == "query_data":
                async for content_chunk in stream_qna_response(user_query, current_file_info.df_head, current_file_info.columns):
                    yield content_chunk
            else: # Visualize or other agent actions (including 'visualize' and 'fallback' determined by router)
                  # The 'fallback' here means the agent_workflow decided to fallback, not the app.py router
                async for summary_chunk in stream_pre_agent_summary(user_query, current_file_info.df_head, current_file_info.columns, action_type_for_routing):
                    yield summary_chunk
                
                # run_agent is async, so it must be awaited
                agent_result_dict = await run_agent(user_query=user_query, df=df) # CORRECTED
                
                yield json.dumps({"type": "final_agent_response", "data": agent_result_dict}) + "\n"
        except Exception as e_stream_gen:
            logger.error(f"Error during agent response streaming for session '{session_id}': {e_stream_gen}", exc_info=True)
            error_payload = {
                "response_type": "error", "content": f"Internal error: {str(e_stream_gen)}",
                "error": str(e_stream_gen), "thinking_log_str": traceback.format_exc() if logger.level == logging.DEBUG else "Internal error details logged."
            }
            yield json.dumps({"type": "final_agent_response", "data": error_payload}) + "\n"
        finally:
            logger.info(f"Full processing stream generation ended for session '{session_id}'.")

    return StreamingResponse(combined_stream_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn dev server (Async Agent Mode) on http://0.0.0.0:8000")
    # For Windows with potential asyncio issues when reload=True, you might need to run uvicorn differently:
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1) # if reload=True causes issues.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")