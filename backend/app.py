import pandas as pd
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.responses import JSONResponse, StreamingResponse # Import JSONResponse and StreamingResponse
from backend.agent_workflow import run_agent, stream_pre_agent_summary, stream_qna_response, RouteQuery, llm as agent_llm
from backend.models import QueryRequest, AgentResponse, FileInfo
import traceback
import zipfile
import os
from datetime import datetime
import logging # For better logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Exception Handler (Optional but Recommended for API-wide JSON errors) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error_type": type(exc).__name__},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException for request {request.url}: Status {exc.status_code}, Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "HTTPException"}, # Ensure detail is always present
    )


# --- Data Stores ---
data_store: dict[str, pd.DataFrame] = {}
file_info_store: dict[str, FileInfo] = {}

# Directory to store screencasts
SCREENCAST_DIR = "screencasts_storage"
os.makedirs(SCREENCAST_DIR, exist_ok=True)


# --- Endpoints ---
@app.post("/uploadfile/", response_model=FileInfo)
async def create_upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload request for: {file.filename}")
    session_id = str(uuid.uuid4())
    filename = file.filename
    
    if not filename: # Added check for filename
        logger.error("File upload attempt with no filename.")
        raise HTTPException(status_code=400, detail="File has no name.")

    contents = await file.read()
    df = None

    try:
        # ... (your existing pandas DataFrame reading logic for .csv, .xlsx, .xls) ...
        # (This part remains the same as your working version)
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            except zipfile.BadZipFile as bzf_error:
                logger.warning(f"BadZipFile error for {filename} with openpyxl: {bzf_error}. Trying xlrd.")
                df = pd.read_excel(io.BytesIO(contents), engine='xlrd') # Fallback
            # Add more specific error handling for Excel if needed
        elif filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(contents), engine='xlrd')
        else:
            logger.error(f"Invalid file type uploaded: {filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV, XLSX, or XLS.")

        if df is None or df.empty:
            logger.error(f"File '{filename}' was empty or could not be parsed.")
            raise HTTPException(status_code=400, detail=f"The file '{filename}' was empty or could not be parsed.")

        data_store[session_id] = df
        file_info_data = FileInfo(
            session_id=session_id,
            filename=filename,
            columns=df.columns.tolist(),
            df_head=df.head().to_string()
        )
        file_info_store[session_id] = file_info_data
        logger.info(f"File '{filename}' processed successfully. Session ID: {session_id}")
        return file_info_data

    except pd.errors.EmptyDataError:
        logger.error(f"Pandas EmptyDataError for file: {filename}")
        raise HTTPException(status_code=400, detail="The uploaded file is empty (pandas error).")
    except HTTPException as e: # Re-raise HTTPExceptions to be handled by the handler
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing file {filename}: {e}", exc_info=True)
        # This will be caught by the global_exception_handler now
        raise HTTPException(status_code=500, detail=f"Internal server error while processing file '{filename}'.")
    # finally block for closing file removed as `await file.read()` consumes it.
    # If you were reading in chunks, you'd manage file closing.


@app.post("/process_query/")
async def process_query(request: QueryRequest):
    session_id = request.session_id
    user_query = request.query
    logger.info(f"Processing query for session {session_id}: '{user_query}'")

    if session_id not in data_store or session_id not in file_info_store:
        logger.warning(f"Session ID {session_id} not found for process_query.")
        # This custom error stream is fine for StreamingResponse.
        # For non-streaming, an HTTPException would be caught by the handler.
        async def error_stream_gen():
            yield json.dumps({"type": "error", "content": "Session ID not found or data not uploaded."}) + "\n"
        return StreamingResponse(error_stream_gen(), media_type="application/x-ndjson", status_code=404)

    # ... (your existing agent routing and streaming logic for process_query) ...
    # (This part remains the same as your working version)
    df = data_store[session_id]
    current_file_info = file_info_store[session_id]
    action_type_for_routing = "query_data"
    try:
        if agent_llm:
            # Your router logic (ensure any errors here are also handled or raise HTTPException)
            from langchain.prompts import ChatPromptTemplate # Moved import here as it's specific to this path
            router_prompt_template_messages = [
                ("system", "You are an expert query router... Columns: {columns}, Head: {df_head}, Query: {user_query}"),
                ("human", "{user_query}")
            ] # Simplified for brevity
            router_prompt = ChatPromptTemplate.from_messages(router_prompt_template_messages)
            structured_router = agent_llm.with_structured_output(RouteQuery, method="function_calling", include_raw=False)
            router_chain = router_prompt | structured_router
            route_result: RouteQuery = router_chain.invoke({
                "columns": current_file_info.columns, "df_head": current_file_info.df_head, "user_query": user_query
            })
            action_type_for_routing = route_result.action
            logger.info(f"FastAPI Router decision: {action_type_for_routing}, Reasoning: {route_result.reasoning}")
        else:
            logger.warning("FastAPI Router: LLM not available, making broad assumption for query type.")
            if any(kw in user_query.lower() for kw in ["plot", "chart", "graph", "visualize", "show me"]):
                action_type_for_routing = "visualize"
    except Exception as e:
        logger.error(f"Error in FastAPI router invocation for session {session_id}: {e}", exc_info=True)
        action_type_for_routing = "fallback" # Or potentially raise an HTTPException to be caught

    async def combined_stream_generator():
        # ... (your existing combined_stream_generator logic)
        # Ensure any exceptions within this generator that aren't caught and yielded as JSON error
        # might lead to the connection being cut prematurely.
        try:
            if action_type_for_routing != "query_data":
                yield json.dumps({"type": "system", "message": f"Router decided action: {action_type_for_routing}. Starting pre-summary stream..."}) + "\n"
                async for summary_chunk in stream_pre_agent_summary(user_query, current_file_info.df_head, current_file_info.columns, action_type_for_routing):
                    yield summary_chunk
                yield json.dumps({"type": "system", "message": "Pre-summary stream ended."}) + "\n"

            if action_type_for_routing == "query_data":
                yield json.dumps({"type": "system", "message": "Starting Q&A stream..."}) + "\n"
                async for content_chunk in stream_qna_response(user_query, current_file_info.df_head, current_file_info.columns):
                    yield content_chunk
                yield json.dumps({"type": "system", "message": "Q&A stream ended."}) + "\n"
            else: # "visualize" or "fallback"
                agent_result_dict = run_agent(user_query=user_query, df=df) # This call needs to be robust
                yield json.dumps({"type": "final_agent_response", "data": agent_result_dict}) + "\n"
        except Exception as e_stream:
            logger.error(f"Error during combined_stream_generator for session {session_id}: {e_stream}", exc_info=True)
            # Yield a structured error if possible
            error_response_stream = AgentResponse(
                response_type="error",
                content=f"An unexpected error occurred while streaming: {str(e_stream)}",
                error=str(e_stream),
                thinking_log_str="Error during agent streaming."
            ).model_dump()
            yield json.dumps({"type": "final_agent_response", "data": error_response_stream}) + "\n"
        finally: # Ensure stream ends cleanly
            yield json.dumps({"type": "system", "message": "Full processing stream ended."}) + "\n"


    return StreamingResponse(combined_stream_generator(), media_type="application/x-ndjson")


@app.post("/upload_screencast/")
async def upload_screencast(
    session_id: str = Form(...),
    screencast_file: UploadFile = File(...)
):
    logger.info(f"--- Received request for /upload_screencast/ for session: {session_id} ---")
    logger.info(f"Uploaded screencast filename: {screencast_file.filename}, content_type: {screencast_file.content_type}")

    if not session_id:
        logger.warning("Screencast upload attempt with no session_id.")
        # This will be caught by the HTTPException handler and returned as JSON
        raise HTTPException(status_code=400, detail="Session ID is required for screencast upload.")

    # Basic filename validation (though frontend might send 'blob')
    # if not screencast_file.filename:
    #     logger.warning("Screencast file received with no filename by server.")
        # Consider if this should be an error or if you generate a name regardless

    session_screencast_dir = os.path.join(SCREENCAST_DIR, session_id)
    try:
        os.makedirs(session_screencast_dir, exist_ok=True)
    except OSError as e_mkdir:
        logger.error(f"Could not create directory {session_screencast_dir}: {e_mkdir}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error: Could not create storage directory for screencast.")


    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_base, original_ext = os.path.splitext(screencast_file.filename if screencast_file.filename else "screencast")

    # Determine a safe extension
    content_type = screencast_file.content_type
    if content_type == "video/webm":
        final_ext = ".webm"
    elif content_type == "video/mp4":
        final_ext = ".mp4"
    elif original_ext.lower() in ['.webm', '.mp4', '.mkv', '.mov']:
        final_ext = original_ext.lower()
    else:
        final_ext = ".webm" # Default if unsure

    sane_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in original_base)[:50] # Sanitize
    saved_filename = f"{timestamp_str}_{sane_base}{final_ext}"
    file_path = os.path.join(session_screencast_dir, saved_filename)

    try:
        contents = await screencast_file.read()
        if not contents:
            logger.warning(f"Screencast file '{screencast_file.filename}' for session {session_id} is empty.")
            raise HTTPException(status_code=400, detail="Received empty screencast file.")

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        file_size_kb = len(contents) / 1024
        logger.info(f"Screencast '{saved_filename}' (Session: {session_id}, Size: {file_size_kb:.2f} KB) saved to '{file_path}'")

        return JSONResponse( # Explicitly return JSONResponse
            status_code=200, # Or 201 Created
            content={
                "message": "Screencast uploaded successfully.",
                "session_id": session_id,
                "filename": saved_filename,
                "path_on_server": file_path,
                "size_kb": round(file_size_kb, 2)
            }
        )
    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        logger.error(f"Error saving screencast for session {session_id}: {e}", exc_info=True)
        if os.path.exists(file_path): # Attempt to clean up partially saved file
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up partially saved file: {file_path}")
            except Exception as e_remove:
                logger.error(f"Error removing partially saved screencast {file_path}: {e_remove}")
        # This will be caught by the global_exception_handler
        raise HTTPException(status_code=500, detail=f"Could not save screencast. Error: {str(e)}")
    finally:
        # UploadFile from FastAPI is an in-memory file or spooled to disk,
        # its .close() method should be called if you manually manage its file pointer.
        # Here, `await screencast_file.read()` consumes it.
        # If you were using `screencast_file.file`, you'd need to close it.
        # For safety, an explicit close can be added if there are concerns.
        # await screencast_file.close() # Generally good practice if not fully consumed by .read()
        pass

if __name__ == "__main__":
    import uvicorn
    # This is for direct execution of this file, e.g., python app.py
    # Your Procfile or docker command would typically use `uvicorn app:app --host 0.0.0.0 --port $PORT`
    logger.info("Starting Uvicorn server directly from app.py for development.")
    uvicorn.run(app, host="0.0.0.0", port=8000)