import os
import pandas as pd
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from backend.utils import generate_plot_from_config 
from backend.models import PlotConfig 
import numpy as np

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

llm = ChatOpenAI(model="gpt-4.1", temperature=0)  

class AgentState(TypedDict):
    user_query: str
    df_head_str: str
    df_columns: List[str]
    df_full: Optional[pd.DataFrame]
    action_type: Optional[str]
    plot_config_json: Optional[str]
    plot_image_bytes: Optional[bytes]
    llm_response: Optional[str]
    error_message: Optional[str]

class RouteQuery(BaseModel):
    action: str = Field(description="Must be 'visualize', 'query_data', or 'fallback'.")
    reasoning: str = Field(description="Brief explanation for the chosen action.")

def router_node(state: AgentState) -> AgentState:
    print("Router Node")
    prompt_template_messages = [
        ("system",
         """You are an expert query router for a data analysis application.
         The user has uploaded a dataset.
         Dataset columns: {columns}
         Data sample (first 5 rows):
         {df_head}

         Based on the user's query, determine the type of action required.
         Possible actions are:
         1. "visualize": If the user is asking to create a plot, chart, graph (e.g., bar, line, scatter, histogram, boxplot, normal distribution, KDE), or visualize data.
         2. "query_data": If the user is asking a general question that can be answered by looking at, summarizing, or analyzing the provided data (e.g., "what is the average age?", "how many unique categories are there?", "calculate the median of sales").
         3. "fallback": If the query is a general greeting, off-topic, or cannot be answered.
         
         User query: "{user_query}"
         Provide your decision and reasoning.
         """),
        ("human", "{user_query}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")
    router_chain = prompt | structured_llm_router

    try:
        route_result = router_chain.invoke({
            "columns": state["df_columns"],
            "df_head": state["df_head_str"],
            "user_query": state["user_query"]
        })
        print(f"Router decision: {route_result.action}, Reasoning: {route_result.reasoning}")
        return {"action_type": route_result.action, "llm_response": f"Routing decision: {route_result.action}. Reasoning: {route_result.reasoning}"}
    except Exception as e:
        print(f"Error in router_node: {e}")
        return {"action_type": "fallback", "error_message": f"Router error: {e}", "llm_response": f"Router error: {e}"}


def visualization_node(state: AgentState) -> AgentState:
    print("Visualization Node")
    prompt_template_messages = [
        ("system",
         f"""You are a data visualization expert. Based on the user's request, determine the parameters for a plot.
         The DataFrame is available.
         Dataset columns: {state["df_columns"]}
         Data sample (first 5 rows):
         {state["df_head_str"]}

         User request: "{state["user_query"]}"

         You MUST select a NUMERICAL column for the `x_column` when the `plot_type` is 'histogram' or 'kde'. These plots show the distribution of numerical data.
         If the user asks for a general distribution plot (e.g., "show me a bell curve", "plot the distribution of data") and does not specify a column, you MUST pick ONE suitable NUMERICAL column from the available columns for the `x_column`.
         If multiple numerical columns exist, prioritize columns like 'Age', 'Score', 'Value', 'Measurement', 'Height_cm', 'TestScore', etc., or the first available numerical column.
         
         Available plot types: "bar", "line", "scatter", "histogram", "boxplot", "kde".
         
         Guidelines for choosing columns:
         - If the user specifies a column (e.g., "histogram of Age"), use that column for `x_column`.
         - If the user asks for a general distribution plot (e.g., "show me a bell curve", "plot the distribution") AND there are multiple numerical columns, you MUST pick ONE suitable numerical column for the `x_column`.
         - A good heuristic for picking a column for a general distribution plot, if not specified, might be the first numerical column, or a column that sounds like a primary measure (e.g., 'Age', 'Score', 'Value', 'Measurement', 'Height_cm').
         - For "histogram" or "kde" (e.g., "distribution of Age"), only `x_column` is typically needed. The output plot will show the distribution of this single column.
         - For "bar" chart of counts (e.g., "bar chart of Genders"), only `x_column` is needed.
         - For "bar" chart of values (e.g., "average Salary by Department"), `x_column` and `y_column` are needed.
         - For "scatter" (e.g., "plot Salary vs Experience"), `x_column` and `y_column` are needed.
         - For "line" (e.g., "Sales over Time"): `x_column` (often time-based or sequential) and `y_column` (numerical) are needed. A line plot of a single categorical column is usually not meaningful without a corresponding numerical y-value.
         - For "boxplot" (e.g., "distribution of Salary by Department"), `x_column` (categorical) and `y_column` (numerical) are common, or just `y_column` for a single series.

         If a column name in the user query is slightly different but clearly refers to an available column, use the correct available column name.
         Think step-by-step to determine the plot configuration. If the user's query is very vague about the column for a distribution plot, and multiple numerical columns exist, select the most prominent or first numerical column from the list: {state["df_columns"]}.
         Provide the configuration as a JSON object matching the PlotConfig schema.
         Ensure `x_column` is correctly chosen based on the plot type and user request. For 'histogram' and 'kde', it must be numerical.
         If no suitable numerical column is available for a requested histogram/KDE, you should indicate that in your reasoning (though the task is to provide a PlotConfig).
         Provide the configuration as a JSON object matching the PlotConfig schema.         """),
        ("human", "User query: {user_query}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    structured_llm_viz = llm.with_structured_output(PlotConfig, method="function_calling")
    viz_chain = prompt | structured_llm_viz

    try:
        plot_config_obj: PlotConfig = viz_chain.invoke({
            "columns": state["df_columns"],
            "df_head": state["df_head_str"],
            "user_query": state["user_query"]
        })
        print(f"LLM generated PlotConfig: {plot_config_obj.model_dump_json(indent=2)}")

        if state["df_full"] is not None and plot_config_obj.x_column and plot_config_obj.plot_type in ["histogram", "kde"]:
            if plot_config_obj.x_column in state["df_full"].columns:
                column_dtype = state["df_full"][plot_config_obj.x_column].dtype
                if not pd.api.types.is_numeric_dtype(column_dtype):
                    original_x_col = plot_config_obj.x_column
                    print(f"Warning: LLM chose categorical column '{original_x_col}' for {plot_config_obj.plot_type}. Attempting to find a numerical column.")
                    numerical_cols = state["df_full"].select_dtypes(include=np.number).columns.tolist()
                    if numerical_cols:
                        plot_config_obj.x_column = numerical_cols[0]
                        print(f"Fallback: Switched x_column from '{original_x_col}' to numerical column '{plot_config_obj.x_column}' for {plot_config_obj.plot_type}.")
                        if plot_config_obj.title and original_x_col in plot_config_obj.title:
                            plot_config_obj.title = plot_config_obj.title.replace(original_x_col, plot_config_obj.x_column)
                    else:
                        error_msg = f"Cannot generate {plot_config_obj.plot_type}: Column '{original_x_col}' is categorical, and no other numerical columns are available."
                        return {
                            "plot_config_json": plot_config_obj.model_dump_json(),
                            "error_message": error_msg,
                            "llm_response": f"I tried to create a {plot_config_obj.plot_type} for '{original_x_col}', but it's not a numerical column. No other numerical columns were found to plot instead."
                        }
            else:  
                error_msg = f"Cannot generate {plot_config_obj.plot_type}: Column '{plot_config_obj.x_column}' chosen by AI does not exist in the data."
                numerical_cols = state["df_full"].select_dtypes(include=np.number).columns.tolist()
                if numerical_cols and ("distribution" in state["user_query"].lower() or "bell curve" in state["user_query"].lower()):
                    plot_config_obj.x_column = numerical_cols[0]
                    print(f"Fallback: LLM chose non-existent column. Switched to numerical column '{plot_config_obj.x_column}' for {plot_config_obj.plot_type}.")
                else:
                    return {
                        "plot_config_json": plot_config_obj.model_dump_json(),
                        "error_message": error_msg,
                        "llm_response": f"I tried to create a {plot_config_obj.plot_type}, but the column '{plot_config_obj.x_column}' doesn't seem to exist."
                    }

        if state["df_full"] is not None:
            plot_bytes = generate_plot_from_config(state["df_full"], plot_config_obj)
            if plot_bytes:
                return {
                    "plot_config_json": plot_config_obj.model_dump_json(),
                    "plot_image_bytes": plot_bytes,
                    "llm_response": f"Here is your {plot_config_obj.plot_type} plot."
                }
            else:
                error_detail = f"Failed to generate plot with config: type='{plot_config_obj.plot_type}', x='{plot_config_obj.x_column}', y='{plot_config_obj.y_column}'. Check logs in utils.py for more details on plotting failure."
                return {
                    "plot_config_json": plot_config_obj.model_dump_json(),
                    "error_message": error_detail,
                    "llm_response": f"I tried to generate a {plot_config_obj.plot_type} plot but encountered an issue. {error_detail}"
                }
        else:
             return {"error_message": "DataFrame not available for visualization.", "llm_response": "DataFrame not found for plotting."}

    except Exception as e:
        print(f"Error in visualization_node: {e}")
        import traceback
        traceback.print_exc()
        return {"error_message": f"Visualization setup error: {e}", "llm_response": "Sorry, I encountered an error trying to set up the visualization."}

async def stream_qna_response(user_query: str, df_head_str: str, df_columns: List[str]):
    """
    Streams the Q&A response directly using Langchain's astream.
    This function will be called by FastAPI for the Q&A streaming path.
    """
    print("--- Streaming Q&A Response Directly ---")
    prompt_template_messages = [
        ("system",
         """You are a helpful AI assistant. The user has uploaded a dataset and has a question.
         Dataset columns: {columns}
         Data sample (first 5 rows):
         {df_head}
         User question: "{user_query}"
         Answer concisely. If asked for specific calculations like mean, median, mode for a column,
         and you can infer it or it's simple, provide an answer.
         For complex calculations or if the full data is needed, state that more processing would be required.
         """),
        ("human", "{user_query}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    qna_chain = prompt | llm 

    try:
        async for chunk in qna_chain.astream({
            "columns": df_columns,
            "df_head": df_head_str,
            "user_query": user_query
        }):
            if chunk.content is not None:
                yield chunk.content # Yield content chunks
    except Exception as e:
        print(f"Error during Q&A streaming: {e}")
        yield f"\n\n[Error in Q&A: {str(e)}]"

def qna_node(state: AgentState) -> AgentState:
    print("--- Q&A Node (Streaming with Langchain) ---")
    prompt_template_messages = [
        ("system",
         """You are a helpful AI assistant. The user has uploaded a dataset and has a question.
         Dataset columns: {columns}
         Data sample (first 5 rows):
         {df_head}

         User question: "{user_query}"
         Answer concisely.
         """),
        ("human", "{user_query}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    qna_chain = prompt | llm
    try:
        response = qna_chain.invoke({
            "columns": state["df_columns"],
            "df_head": state["df_head_str"],
            "user_query": state["user_query"]
        })
        return {"llm_response": response.content}
    except Exception as e:
        print(f"Error in sync qna_node: {e}")
        return {"error_message": f"Q&A error: {e}", "llm_response": "Error answering."}
def fallback_node(state: AgentState) -> AgentState:
    print("--- Fallback Node ---")
    return {"llm_response": "I'm sorry, I can only help with visualizations or questions directly related to the uploaded data. Your query seems to be outside this scope."}

def decide_next_node(state: AgentState) -> str:
    if state.get("error_message"):
        print(f"Error detected ('{state.get('error_message')}'), ending early.")
        return END
        
    action = state.get("action_type")
    if action == "visualize":
        return "visualization_agent"
    elif action == "query_data":
        return "qna_agent"
    elif action == "fallback": 
        return "fallback_agent"
    else: 
        print(f"Unknown action_type: {action}. Routing to fallback.")
        return "fallback_agent"

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("visualization_agent", visualization_node)
workflow.add_node("qna_agent", qna_node)
workflow.add_node("fallback_agent", fallback_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "visualization_agent": "visualization_agent",
        "qna_agent": "qna_agent",
        "fallback_agent": "fallback_agent",
        END: END
    }
)
workflow.add_edge("visualization_agent", END)
workflow.add_edge("qna_agent", END)
workflow.add_edge("fallback_agent", END)
app_graph = workflow.compile()

def run_agent(user_query: str, df: pd.DataFrame) -> dict:
    df_head_str = df.head().to_string()
    df_columns = df.columns.tolist()

    initial_state = AgentState(
        user_query=user_query,
        df_head_str=df_head_str,
        df_columns=df_columns,
        df_full=df,
        action_type=None,
        plot_config_json=None,
        plot_image_bytes=None,
        llm_response=None,
        error_message=None
    )

    final_state = app_graph.invoke(initial_state)

    plot_bytes = final_state.get("plot_image_bytes")
    plot_bytes_b64 = None
    if plot_bytes:
        import base64
        plot_bytes_b64 = base64.b64encode(plot_bytes).decode('utf-8')

    response = {
        "response_type": final_state.get("action_type", "fallback"),
        "content": final_state.get("llm_response"),
        "plot_image_bytes": plot_bytes_b64, # Send as base64 string
        "plot_config_json": final_state.get("plot_config_json"),
        "error": final_state.get("error_message")
    }

    if final_state.get("error_message") and not final_state.get("action_type"):
        response["response_type"] = "error" # Mark as general error
        if not response["content"]: # If no specific llm_response for the error
            response["content"] = final_state.get("error_message")
    elif final_state.get("error_message") and final_state.get("action_type"):
        pass

    return response