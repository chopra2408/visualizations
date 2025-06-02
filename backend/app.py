import pandas as pd
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi.responses import StreamingResponse
from agent_workflow import run_agent, stream_qna_response, RouteQuery, llm as agent_llm
from models import QueryRequest, AgentResponse, FileInfo 
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store: dict[str, pd.DataFrame] = {}
file_info_store: dict[str, FileInfo] = {}

@app.post("/uploadfile/", response_model=FileInfo)  
async def create_upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    filename = file.filename
    
    try:
        contents = await file.read()
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or XLSX.")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file is empty or could not be parsed into a DataFrame.")

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
        raise HTTPException(status_code=400, detail="The uploaded file is empty (pandas error).")
    except HTTPException as e:  
        raise e
    except Exception as e:
        print(f"Error processing file in /uploadfile/: {e}")
        import traceback
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=f"Internal server error while processing the file.")
    finally:
        if file and not file.file.closed: 
            await file.close()

@app.post("/process_query/") 
async def process_query(request: QueryRequest):
    session_id = request.session_id
    user_query = request.query

    if session_id not in data_store or session_id not in file_info_store:
        raise HTTPException(status_code=404, detail="Session ID not found or data not uploaded.")

    df = data_store[session_id]
    current_file_info = file_info_store[session_id] 
    router_prompt_template_messages = [
        ("system",
         """You are an expert query router...
         Dataset columns: {columns}
         Data sample (first 5 rows):
         {df_head}
         User query: "{user_query}"
         Based on the user's query, determine the type of action: 'visualize', 'query_data', or 'fallback'.
         """),
        ("human", "{user_query}")
    ]
    router_prompt = ChatPromptTemplate.from_messages(router_prompt_template_messages)
    # agent_llm is the ChatOpenAI instance from agent_workflow
    structured_router = agent_llm.with_structured_output(RouteQuery, method="function_calling")
    router_chain = router_prompt | structured_router

    try:
        route_result: RouteQuery = router_chain.invoke({
            "columns": current_file_info.columns,
            "df_head": current_file_info.df_head,
            "user_query": user_query
        })
        action_type = route_result.action
        print(f"FastAPI Router decision: {action_type}, Reasoning: {route_result.reasoning}")

    except Exception as e:
        print(f"Error in FastAPI router invocation: {e}")
        # Fallback to a generic error structure Streamlit can parse
        error_response = AgentResponse(
            response_type="error",
            content=f"Error in routing your query: {str(e)}",
            error=str(e)
        ).model_dump_json() # Use model_dump_json for Pydantic v2
        # This isn't ideal for streaming, but necessary if router fails before decision
        async def single_error_chunk():
            yield json.dumps({"type": "error", "data": error_response}) # Send as a JSON chunk
        return StreamingResponse(single_error_chunk(), media_type="application/x-ndjson")


    # 2. Conditional logic based on router's decision
    if action_type == "query_data":
        print("Streaming Q&A response from FastAPI...")
        # Define an async generator that FastAPI's StreamingResponse will use
        async def stream_generator():
            # Yield a start marker (optional, but can be useful for frontend)
            # We need to send JSON chunks if we mix types of messages (control, data, error)
            yield json.dumps({"type": "system", "message": "Starting Q&A stream..."}) + "\n"
            async for content_chunk in stream_qna_response(
                user_query, current_file_info.df_head, current_file_info.columns
            ):
                yield json.dumps({"type": "content", "chunk": content_chunk}) + "\n" # Send content as JSON chunks
            yield json.dumps({"type": "system", "message": "Q&A stream ended."}) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson") # Use ndjson

    elif action_type == "visualize" or action_type == "fallback":
        print(f"Processing '{action_type}' using full agent graph...")
        try:
            agent_result_dict = run_agent(user_query=user_query, df=df)
            return agent_result_dict 

        except Exception as e:
            print(f"Error during agent processing for {action_type}: {e}")
            import traceback
            traceback.print_exc()
            error_response_obj = AgentResponse(
                response_type="error",
                content=f"An unexpected error occurred: {str(e)}",
                error=str(e)
            )
            return error_response_obj.model_dump() 

    else: 
        error_response_obj = AgentResponse(response_type="error", content="Unknown action type from router.")
        return error_response_obj.model_dump()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)