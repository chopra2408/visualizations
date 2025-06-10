# --- START OF FILE agent_workflow.py ---

import os
import pandas as pd
from typing import TypedDict, List, Optional, Tuple, Type
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from backend.utils import generate_plot_from_config, calculate_age_from_dob # Assuming these are in your project
from backend.models import PlotConfig # Assuming this is in your project
import numpy as np
import traceback
import json
from openai import AsyncAzureOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage
import httpx # For custom HTTP client with proxy
import asyncio # For running async code if needed from sync context

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Get Azure OpenAI connection details from environment
llm_azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
llm_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
llm_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview") # Default if not set
llm_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

# Get Proxy details from environment (optional)
# Prefers HTTPS_PROXY, then HTTP_PROXY
# PROXY_URI = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("TARGET_URI")

openai_client: Optional[AsyncAzureOpenAI] = None
custom_http_client_for_openai: Optional[httpx.AsyncClient] = None

try:
    if not all([llm_azure_endpoint, llm_api_key, llm_deployment_name]):
        missing_vars = [
            var_name for var_name, var_val in {
                "AZURE_OPENAI_ENDPOINT": llm_azure_endpoint,
                "AZURE_OPENAI_API_KEY": llm_api_key,
                "AZURE_OPENAI_DEPLOYMENT_NAME": llm_deployment_name
            }.items() if not var_val
        ]
        raise ValueError(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")

    client_args = {
        "azure_endpoint": llm_azure_endpoint,
        "api_key": llm_api_key,
        "api_version": llm_api_version,
        "azure_deployment": llm_deployment_name, # Default deployment for this client
    }

    # if PROXY_URI:
    #     print(f"INFO: Configuring OpenAI client to use proxy: {PROXY_URI}")
    #     proxies = {
    #         "http://": PROXY_URI,
    #         "https://": PROXY_URI,
    #     }
    #     custom_http_client_for_openai = httpx.AsyncClient(proxies=proxies, timeout=30.0) # Added timeout
    #     client_args["http_client"] = custom_http_client_for_openai
    
    openai_client = AsyncAzureOpenAI(**client_args)
    print("INFO: AsyncAzureOpenAI Client Initialized successfully in agent_workflow.py.")

except Exception as e:
    print(f"FATAL ERROR initializing AsyncAzureOpenAI Client in agent_workflow.py: {e}")
    openai_client = None
    # If a custom client was created, it should be closed on application shutdown.
    # For simplicity in this script, we're not managing its explicit closure here on init failure.
    # In a long-running app, ensure custom_http_client_for_openai.aclose() is called.

class AgentState(TypedDict):
    user_query: str
    df_head_str: str
    df_columns: List[str]
    df_full: Optional[pd.DataFrame]
    action_type: Optional[str]
    plot_config_json: Optional[str]
    plotly_fig_json: Optional[str]
    plot_insights: Optional[str]
    llm_response: Optional[str]
    error_message: Optional[str]
    thinking_log: List[str]

class RouteQuery(BaseModel):
    action: str = Field(description="Must be 'visualize', 'query_data', or 'fallback'.")
    reasoning: str = Field(description="Brief explanation for the chosen action.")

class PlotInsights(BaseModel):
    insights: str = Field(description="A concise textual summary of what the plot shows and any key observations or conclusions that can be drawn from it.")
    suggestions: Optional[List[str]] = Field(None, description="Optional: 1-2 follow-up questions or related plots.")

def _convert_lc_messages_to_openai_format(lc_messages: List[BaseMessage]) -> List[dict]:
    """Converts Langchain BaseMessage objects to OpenAI API message format."""
    out_messages = []
    for msg in lc_messages:
        role = "system" # Default
        if msg.type == "human": role = "user"
        elif msg.type == "ai": role = "assistant"
        elif msg.type == "system": role = "system"
        elif msg.type == "function": role = "function" # For function/tool calls
        elif msg.type == "tool": role = "tool" # For tool results
        else: # Fallback for older or different structures, might need adjustment
            if msg.__class__.__name__ == "SystemMessage": role = "system"
            elif msg.__class__.__name__ == "HumanMessage": role = "user"
            elif msg.__class__.__name__ == "AIMessage": role = "assistant"
            
        content = msg.content
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('function_call'):
            # Handle AIMessage with function_call for older Langchain versions if needed
            # For new openai lib, tool_calls is preferred.
            pass # This path might need refinement if you hit it with specific Langchain versions
        
        out_messages.append({"role": role, "content": content})
    return out_messages


async def get_structured_output_from_openai(
    pydantic_model: Type[BaseModel],
    prompt_template: ChatPromptTemplate,
    prompt_input: dict,
    thinking_log: List[str],
    log_prefix: str
) -> Tuple[Optional[BaseModel], List[str]]:
    current_log = list(thinking_log)
    if openai_client is None or llm_deployment_name is None: # Also check deployment name
        current_log.append(f"{log_prefix}_ERROR: OpenAI client or deployment name not available.")
        return None, current_log

    tool_schema = convert_to_openai_tool(pydantic_model)
    
    try:
        prompt_value = prompt_template.invoke(prompt_input)
        messages = _convert_lc_messages_to_openai_format(prompt_value.to_messages())

        current_log.append(f"{log_prefix}_PROMPT_MESSAGES (first 500 chars): {json.dumps(messages, indent=0)[:500]}...")

        response = await openai_client.chat.completions.create(
            model=llm_deployment_name, # Explicitly pass deployment name
            messages=messages,
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": tool_schema['function']['name']}}
        )
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_args = tool_call.function.arguments
            current_log.append(f"{log_prefix}_RAW_ARGS: {function_args}")
            try:
                instance = pydantic_model.model_validate_json(function_args)
                current_log.append(f"{log_prefix}_SUCCESS: Parsed to {pydantic_model.__name__}")
                return instance, current_log
            except ValidationError as e:
                current_log.append(f"{log_prefix}_PARSE_ERROR: Pydantic validation for {pydantic_model.__name__} failed: {e}. Args: {function_args}")
                traceback.print_exc()
                return None, current_log
        else:
            error_detail = f"No tool_calls in LLM response. Content: {message.content}"
            current_log.append(f"{log_prefix}_ERROR: {error_detail}")
            if message.content:
                 current_log.append(f"{log_prefix}_FALLBACK_CONTENT: {message.content}")
            return None, current_log
    except Exception as e:
        current_log.append(f"{log_prefix}_CALL_ERROR: LLM call for {pydantic_model.__name__} failed: {type(e).__name__} {str(e)}")
        traceback.print_exc()
        return None, current_log

async def generate_insights_for_plot(
    plot_config: PlotConfig, df_columns_list: List[str], df_head_sample: str,
    user_query_str: str, current_thinking_log: List[str]
) -> Tuple[Optional[str], List[str]]:
    thinking_log = list(current_thinking_log)
    if openai_client is None:
        thinking_log.append("INSIGHTS_ERROR: OpenAI client not available.")
        return "Could not generate insights: OpenAI client not available.", thinking_log

    desc = (f"An interactive '{plot_config.plot_type}' plot titled '{plot_config.title or 'N/A'}' was generated. "
            f"X-axis: '{plot_config.xlabel or plot_config.x_column or 'N/A'}'. "
            f"Y-axis: '{plot_config.ylabel or plot_config.y_column or 'N/A'}'. "
            f"Colored by: '{plot_config.color_by_column or 'N/A'}'.")
    if "Age_Derived_From_DOB" in desc: desc += " 'Age' was derived from Date of Birth."
    
    insights_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data analyst. An interactive plot was generated.
         User Query: "{user_query_for_insights}"
         Plot Description: {plot_description_for_insights}
         Dataset Columns: {dataset_columns_for_insights}
         Data Sample (head): {data_sample_for_insights}
         Provide concise insights based ONLY on the plot description and data context.
         What key takeaways, patterns, or distributions might this plot reveal?
         Do NOT hallucinate specific data values. Focus on typical interpretations.
         """),
        ("human", "Please provide insights for the generated plot.")
    ])

    prompt_input = {
        "user_query_for_insights": user_query_str,
        "plot_description_for_insights": desc,
        "dataset_columns_for_insights": df_columns_list,
        "data_sample_for_insights": df_head_sample
    }

    plot_insights_instance, thinking_log = await get_structured_output_from_openai(
        PlotInsights, insights_prompt_template, prompt_input, thinking_log, "INSIGHTS"
    )

    if plot_insights_instance:
        txt = plot_insights_instance.insights
        if plot_insights_instance.suggestions:
            txt += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in plot_insights_instance.suggestions)
        thinking_log.append(f"INSIGHTS_RESULT: Generated (first 100 chars): {txt[:100]}...")
        return txt, thinking_log
    else:
        thinking_log.append("INSIGHTS_ERROR: Failed to get structured PlotInsights.")
        return "Could not generate insights due to an internal error.", thinking_log


async def router_node(state: AgentState) -> AgentState:
    print("\n--- Router Node ---")
    current_log = state.get("thinking_log", [])
    current_log.extend(["--- Router Node: Initiated ---", f"Query: '{state['user_query']}'"])

    if openai_client is None:
        current_log.append("Router_ERROR: OpenAI client not available.")
        return {**state, "action_type": "fallback", "error_message": "OpenAI client unavailable.", "llm_response": "System error: LLM client is down.", "thinking_log": current_log}

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert query router. Based on the user's query, data columns, and a sample of the data, determine the primary action required.
        The available actions are: 'visualize', 'query_data', or 'fallback'.
        Data Context: Columns: {router_df_columns}, Data Sample (head): {router_df_head}
        User Query: "{router_user_query}"
        Provide your routing decision and a brief reasoning.
        """),
        ("human", "Determine the action: 'visualize', 'query_data', or 'fallback'.")
    ])
    
    prompt_input = {
        "router_df_columns": state["df_columns"],
        "router_df_head": state["df_head_str"],
        "router_user_query": state["user_query"]
    }

    route_query_instance, current_log = await get_structured_output_from_openai(
        RouteQuery, router_prompt, prompt_input, current_log, "ROUTER"
    )

    if route_query_instance:
        current_log.append(f"Router Decision: Action='{route_query_instance.action}', Reasoning='{route_query_instance.reasoning}'")
        return {**state, "action_type": route_query_instance.action, "llm_response": f"Okay, I will try to {route_query_instance.action.replace('_', ' ')} for your query.", "thinking_log": current_log}
    else:
        current_log.append("Router_ERROR: Failed to get structured RouteQuery from LLM.")
        return {**state, "action_type": "fallback", "error_message": "Error in routing: Could not determine action from LLM.", "llm_response": "Error in routing decision.", "thinking_log": current_log}


async def stream_pre_agent_summary(
    user_query: str, df_head_str: str, df_columns: List[str], intended_action: str
):
    print(f"\n--- Streaming Pre-Agent Summary for action: {intended_action} ---")
    if openai_client is None or llm_deployment_name is None:
        yield json.dumps({"type": "error", "chunk": "OpenAI client or deployment name not available for pre-summary."}) + "\n"; return

    summary_prompt_system = f"""You are a helpful AI assistant providing a quick plan.
    Dataset columns: {df_columns}, Data sample (first 5 rows): {df_head_str}
    User query: "{user_query}", Your current intended action is: '{intended_action}'.
    Provide a concise, user-friendly summary of your plan (1-2 sentences).
    """
    pre_summary_prompt_template = ChatPromptTemplate.from_messages([("system", summary_prompt_system), ("human", "Briefly, what's your plan?")])
    prompt_value = pre_summary_prompt_template.invoke({})
    messages = _convert_lc_messages_to_openai_format(prompt_value.to_messages())

    try:
        stream = await openai_client.chat.completions.create(
            model=llm_deployment_name, # Explicitly pass deployment name
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield json.dumps({"type": "thinking_process_update", "chunk": chunk.choices[0].delta.content}) + "\n"
    except Exception as e:
        print(f"Error during pre-agent summary streaming: {e}"); traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in pre-summary stream: {str(e)}"}) + "\n"

async def stream_qna_response(user_query: str, df_head_str: str, df_columns: List[str]):
    print("\n--- Streaming Q&A Response Directly ---")
    if openai_client is None or llm_deployment_name is None:
        yield json.dumps({"type": "error", "chunk": "OpenAI client or deployment name not available for Q&A."}) + "\n"; return
    
    prompt_template_messages_spec = [
        ("system", f"""You are a helpful AI assistant.
         Dataset columns: {df_columns}, Data sample (first 5 rows): {df_head_str}
         User question: "{user_query}"
         Answer concisely based *only* on the provided data sample and column names.
         If asked for simple calculations from the sample, try to provide an answer.
         For complex calculations or if full data is needed, state that. Do not make up data.
         """),
        ("human", "{user_query_for_qna}")
    ]
    prompt_template = ChatPromptTemplate.from_messages(prompt_template_messages_spec)
    prompt_value = prompt_template.invoke({"user_query_for_qna": user_query})
    messages = _convert_lc_messages_to_openai_format(prompt_value.to_messages())
    
    try:
        stream = await openai_client.chat.completions.create(
            model=llm_deployment_name, # Explicitly pass deployment name
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield json.dumps({"type": "content", "chunk": chunk.choices[0].delta.content}) + "\n"
    except Exception as e:
        print(f"Error during Q&A streaming in agent_workflow: {e}"); traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in Q&A stream: {str(e)}"}) + "\n"


async def visualization_node(state: AgentState) -> AgentState:
    print("\n--- Visualization Node ---")
    thinking_log = state.get("thinking_log", [])
    thinking_log.append("--- Visualization Node: Initiated ---")

    user_query = state["user_query"]
    user_query_lower = user_query.lower()
    df_columns_list = state["df_columns"]
    df_full = state["df_full"]
    df_head_sample = state["df_head_str"]

    plot_config_obj: Optional[PlotConfig] = None
    derived_age_col_name = "Age_Derived_From_DOB"

    if df_full is None or df_full.empty:
        thinking_log.append("VIZ_ERROR: DataFrame is empty or None at node start.")
        return {**state, "error_message": "No data available to visualize.", "llm_response": "Cannot create a plot because the data is missing or empty.", "thinking_log": thinking_log, "action_type": "visualize"}

    df_to_plot = df_full.copy()
    thinking_log.append(f"VIZ_INFO: User Query: '{user_query}'")
    thinking_log.append(f"VIZ_INFO: Available columns: {df_columns_list}")

    # 1. Programmatic Age Distribution Check (remains synchronous logic)
    age_distribution_keywords = ["age distribution", "distribution of age", "histogram of age", "age histogram", "kde of age", "age kde", "bell curve of age"]
    user_wants_age_distribution = any(kw in user_query_lower for kw in age_distribution_keywords)
    thinking_log.append(f"VIZ_CHECK: User wants age distribution? {user_wants_age_distribution}")

    if user_wants_age_distribution:
        log_prefix = "VIZ_AGEDIST_"
        thinking_log.append(f"{log_prefix}Attempting programmatic age distribution plot.")
        actual_dob_col = next((c for c in df_columns_list if c.lower() in ["date of birth", "dob", "birth date", "birthdate"]), None)
        existing_age_col = next((c for c in df_columns_list if c.lower() == "age"), None)
        age_col_to_use = None
        is_derived_age = False

        if existing_age_col and existing_age_col in df_to_plot.columns and pd.api.types.is_numeric_dtype(df_to_plot[existing_age_col]):
            age_col_to_use = existing_age_col
            thinking_log.append(f"{log_prefix}Using existing numeric 'Age' column: '{existing_age_col}'.")
        elif actual_dob_col:
            thinking_log.append(f"{log_prefix}Attempting to derive age from DOB column: '{actual_dob_col}'.")
            age_series = calculate_age_from_dob(df_to_plot, actual_dob_col)
            if age_series is not None and not age_series.isnull().all() and pd.api.types.is_numeric_dtype(age_series):
                df_to_plot[derived_age_col_name] = age_series
                age_col_to_use = derived_age_col_name
                is_derived_age = True
                thinking_log.append(f"{log_prefix}Successfully derived '{derived_age_col_name}'. Dtype: {age_series.dtype}.")
            else:
                thinking_log.append(f"{log_prefix}ERROR: Failed to derive numeric age from '{actual_dob_col}'. Age series was None, all null, or not numeric.")
        else:
            thinking_log.append(f"{log_prefix}No suitable existing 'Age' or 'DOB' column found for programmatic age plot.")

        if age_col_to_use:
            plot_type_for_age = "kde" if "bell curve" in user_query_lower or "kde" in user_query_lower else "histogram"
            plot_config_obj = PlotConfig(
                plot_type=plot_type_for_age, x_column=age_col_to_use,
                title=f"Age Distribution{' (Derived)' if is_derived_age else ''}{' (Bell Curve)' if plot_type_for_age == 'kde' else ''}",
                xlabel=f"Age{' (Derived)' if is_derived_age else ''}",
                ylabel="Density" if plot_type_for_age == "kde" else "Frequency"
            )
            thinking_log.append(f"{log_prefix}Programmatically set PlotConfig for age: {plot_config_obj.model_dump_json(indent=0)}")
        else:
            thinking_log.append(f"{log_prefix}Could not find or derive suitable age column programmatically. Will try LLM if needed.")


    # 2. LLM for PlotConfig if not programmatically set
    if plot_config_obj is None:
        if openai_client is None:
            thinking_log.append("VIZ_LLM_ERROR: OpenAI client is not available.")
            return {**state, "error_message": "OpenAI client not available for plotting.", "llm_response": "System error: Cannot determine plot settings.", "thinking_log": thinking_log}

        thinking_log.append("VIZ_LLM_ACTION: Using LLM to determine plot configuration.")
        actual_dob_col_for_llm = next((c for c in df_columns_list if c.lower() in ["date of birth", "dob", "birth date", "birthdate"]), None)
        dob_info_for_llm = "No specific Date of Birth or Age column identified that could be used for age derivation for the LLM."
        if derived_age_col_name in df_to_plot.columns:
             dob_info_for_llm = (f"An 'Age' column ('{derived_age_col_name}') has already been derived and is available. "
                                 f"If 'Age' is requested for plotting, use '{derived_age_col_name}'.")
        elif actual_dob_col_for_llm:
            dob_info_for_llm = (f"A 'Date of Birth' column ('{actual_dob_col_for_llm}') is present. "
                                f"If 'Age' is requested for plotting, use '{derived_age_col_name}' as the column name; the system can derive it if this column is selected.")
        thinking_log.append(f"VIZ_LLM_INFO: DOB/Age info for LLM prompt: {dob_info_for_llm}")

        cardinality_info_dict = {col: df_full[col].nunique() for col in df_columns_list if col in df_full}
        cardinality_prompt_str = "\n".join([f"  - Column '{col}': {count} unique values" for col, count in cardinality_info_dict.items()])
        
        system_prompt_for_plot_config = f"""You are an expert data visualization advisor. Your goal is to choose the BEST and MOST READABLE plot configuration based on the user's query and the provided data characteristics.
Output your choice strictly using the PlotConfig schema.

Data Context:
- Dataset columns available: {{prompt_df_columns}}
- Data sample (first 5 rows):
{{prompt_df_head}}
- Column Cardinalities (number of unique values):
{{prompt_cardinalities}}
- Information about Date of Birth column for age derivation: {{prompt_dob_info}}

User request: "{{prompt_user_query}}"

Decision Guidelines for Plot Type and Configuration:
... (rest of the detailed prompt from original file, unchanged) ...
Available plot types in PlotConfig: "bar", "line", "scatter", "histogram", "boxplot", "kde", "auto_categorical", "heatmap", "dot_plot", "cumulative_curve", "lollipop", "pie", "doughnut".
Ensure all fields in PlotConfig are populated if they are relevant to the chosen plot type.
""" # Ensure your full prompt is here
        viz_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_for_plot_config),
            ("human", "Based on my request and the data context provided, suggest the most suitable and readable plot configuration using the PlotConfig schema.")
        ])
        
        prompt_input_viz = {
            "prompt_df_columns": df_columns_list,
            "prompt_df_head": df_head_sample,
            "prompt_cardinalities": cardinality_prompt_str,
            "prompt_dob_info": dob_info_for_llm,
            "prompt_user_query": user_query,
            "derived_age_col_name": derived_age_col_name
        }

        plot_config_obj_llm, thinking_log = await get_structured_output_from_openai(
            PlotConfig, viz_prompt_template, prompt_input_viz, thinking_log, "VIZ_LLM_CONFIG"
        )
        
        if plot_config_obj_llm:
            plot_config_obj = plot_config_obj_llm
            thinking_log.append(f"VIZ_LLM_CONFIG_RECEIVED: PlotConfig from LLM: {plot_config_obj.model_dump_json(indent=0)}")

            if actual_dob_col_for_llm and \
               (plot_config_obj.x_column == derived_age_col_name or plot_config_obj.y_column == derived_age_col_name) and \
               derived_age_col_name not in df_to_plot.columns:
                thinking_log.append(f"VIZ_LLM_ACTION: LLM suggested '{derived_age_col_name}'. Attempting derivation from '{actual_dob_col_for_llm}'.")
                age_series = calculate_age_from_dob(df_to_plot, actual_dob_col_for_llm)
                if age_series is not None and not age_series.isnull().all() and pd.api.types.is_numeric_dtype(age_series):
                    df_to_plot[derived_age_col_name] = age_series
                    thinking_log.append(f"VIZ_LLM_SUCCESS: Derived '{derived_age_col_name}' for LLM-suggested plot. Dtype: {age_series.dtype}.")
                else:
                    thinking_log.append(f"VIZ_LLM_ERROR: LLM suggested derived age from '{actual_dob_col_for_llm}', but derivation failed or was non-numeric. Plot may fail.")
                    if plot_config_obj.plot_type in ["histogram", "kde"] and \
                       (plot_config_obj.x_column == derived_age_col_name or plot_config_obj.y_column == derived_age_col_name):
                        plot_config_obj = None
                        thinking_log.append(f"VIZ_LLM_ERROR: Invalidated PlotConfig due to failed critical age derivation for {plot_config_obj.plot_type if plot_config_obj else 'N/A'}.")
        else:
            thinking_log.append("VIZ_LLM_ERROR: Failed to get PlotConfig from LLM (get_structured_output_from_openai returned None).")
            plot_config_obj = None


    # 3. Final check for PlotConfig and Validation (remains synchronous logic)
    if plot_config_obj is None:
        thinking_log.append("VIZ_ERROR: PlotConfig is None after all attempts (programmatic and LLM).")
        return {**state, "error_message": "Could not determine how to configure the plot for your request.", "llm_response": "I'm unable to create the visualization as I couldn't determine the necessary settings.", "thinking_log": thinking_log}

    thinking_log.append(f"VIZ_VALIDATE_INPUT: Validating final PlotConfig: {plot_config_obj.model_dump_json(indent=0)}")
    thinking_log.append(f"VIZ_VALIDATE_INPUT: Columns in df_to_plot for validation: {df_to_plot.columns.tolist()}")

    required_cols_for_plot = [col for col in [plot_config_obj.x_column, plot_config_obj.y_column, plot_config_obj.color_by_column] if col]
    missing_cols = [col for col in required_cols_for_plot if col not in df_to_plot.columns]
    if missing_cols:
        msg = f"Column(s) selected for plotting not found in the available data: {', '.join(missing_cols)}. Available: {df_to_plot.columns.tolist()}"
        thinking_log.append(f"VIZ_VALIDATE_ERROR: {msg}")
        return {**state, "error_message": msg, "llm_response": msg, "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json()}

    if plot_config_obj.plot_type not in ["pie", "doughnut"] and plot_config_obj.plot_type in ["bar", "histogram", "kde", "line", "scatter", "auto_categorical", "boxplot", "heatmap", "dot_plot", "cumulative_curve", "lollipop"]:
        if not plot_config_obj.x_column and not (plot_config_obj.plot_type == "boxplot" and plot_config_obj.y_column):
            msg = f"X-axis column is required for plot type '{plot_config_obj.plot_type}' (unless boxplot of single Y) but was not specified."
            thinking_log.append(f"VIZ_VALIDATE_ERROR: {msg}")
            return {**state, "error_message": msg, "llm_response": msg, "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json()}

    final_plot_df = df_to_plot

    # 4. Generate plot (Plotly JSON string) - synchronous
    thinking_log.append(f"VIZ_PLOT_ATTEMPT: Generating interactive plot with final config: {plot_config_obj.model_dump_json(indent=0)}")
    try:
        plotly_json_string = generate_plot_from_config(final_plot_df, plot_config_obj)

        if plotly_json_string:
            thinking_log.append("VIZ_PLOT_SUCCESS: Interactive Plotly JSON string generated successfully.")
            llm_user_message = f"Here is your interactive {plot_config_obj.plot_type} plot"
            if plot_config_obj.title: llm_user_message += f" titled: '{plot_config_obj.title}'."
            else: llm_user_message += "."

            insights_text, thinking_log_after_insights = await generate_insights_for_plot(
                plot_config_obj, df_columns_list, df_head_sample, user_query, thinking_log
            )
            return {
                **state,
                "plot_config_json": plot_config_obj.model_dump_json(),
                "plotly_fig_json": plotly_json_string,
                "llm_response": llm_user_message,
                "plot_insights": insights_text,
                "thinking_log": thinking_log_after_insights,
                "error_message": None
            }
        else:
            thinking_log.append(f"VIZ_PLOT_ERROR: generate_plot_from_config returned None. Config was: {plot_config_obj.model_dump_json(indent=0)}")
            return {**state, "error_message": "Plot generation function failed to produce an interactive plot.", "llm_response": f"I tried to create an interactive {plot_config_obj.plot_type} plot, but there was an issue generating the plot data.", "plot_config_json": plot_config_obj.model_dump_json(), "thinking_log": thinking_log}
    except Exception as e_plotting:
        thinking_log.append(f"VIZ_PLOT_CRITICAL_ERROR: Exception during interactive plot generation: {str(e_plotting)}")
        traceback.print_exc()
        return {**state, "error_message": f"A critical error occurred while creating the interactive plot: {str(e_plotting)}", "llm_response": "Sorry, an unexpected error stopped me from creating the interactive plot.", "plot_config_json": plot_config_obj.model_dump_json(), "thinking_log": thinking_log}


async def qna_node(state: AgentState) -> AgentState:
    current_log = state.get("thinking_log", [])
    current_log.append("--- QnA Node: Initiated ---")
    if openai_client is None or llm_deployment_name is None:
        current_log.append("QnA_ERROR: OpenAI client or deployment name not available.")
        return {**state, "error_message": "OpenAI client/deployment unavailable.", "llm_response": "LLM error: Client not available.", "thinking_log": current_log}

    qna_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the provided column names and data sample.\nColumns: {qna_df_cols}\nSample Data:\n{qna_df_head}"),
        ("human", "{qna_user_query}")
    ])
    
    prompt_input = {
        "qna_df_cols": state["df_columns"],
        "qna_df_head": state["df_head_str"],
        "qna_user_query": state["user_query"]
    }
    prompt_value = qna_prompt_template.invoke(prompt_input)
    messages = _convert_lc_messages_to_openai_format(prompt_value.to_messages())

    try:
        response = await openai_client.chat.completions.create(
            model=llm_deployment_name, # Explicitly pass deployment name
            messages=messages
        )
        llm_content = response.choices[0].message.content
        current_log.append("QnA_SUCCESS: Response generated.")
        return {**state, "llm_response": llm_content, "thinking_log": current_log, "error_message": None}
    except Exception as e:
        current_log.append(f"QnA_ERROR: {str(e)}"); traceback.print_exc()
        return {**state, "error_message": str(e), "llm_response": "Error in QnA processing.", "thinking_log": current_log}


def fallback_node(state: AgentState) -> AgentState:
    log = state.get("thinking_log", [])
    log.append("--- Fallback Node: Initiated ---")
    user_msg = "I'm sorry, I could not fully process that request."
    if state.get("error_message"): user_msg += f" Details: {state.get('error_message')}"; log.append(f"Fallback_REASON: Error: {state.get('error_message')}")
    else: log.append("Fallback_REASON: Query off-topic, router default, or unhandled action.")
    return_state = {**state, "llm_response": user_msg, "action_type": "fallback", "thinking_log": log}
    if not return_state.get("error_message"): return_state["error_message"] = "Fell back due to unhandled query or internal issue."
    return return_state

def decide_next_node(state: AgentState) -> str:
    log = state.get("thinking_log", [])
    log.append(f"--- Deciding Next Node: Action='{state.get('action_type')}', Error='{state.get('error_message')}' ---")
    action = state.get("action_type")
    if action == "visualize": return "visualization_agent"
    if action == "query_data": return "qna_agent"
    return "fallback_agent"


# Workflow setup
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("visualization_agent", visualization_node)
workflow.add_node("qna_agent", qna_node)
workflow.add_node("fallback_agent", fallback_node) # Sync

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next_node, {
    "visualization_agent": "visualization_agent", "qna_agent": "qna_agent", "fallback_agent": "fallback_agent"
})
workflow.add_edge("visualization_agent", END)
workflow.add_edge("qna_agent", END)
workflow.add_edge("fallback_agent", END)

app_graph = None
try:
    app_graph = workflow.compile()
    print("INFO: LangGraph compiled successfully in agent_workflow.py.")
except Exception as e_compile:
    print(f"FATAL ERROR compiling LangGraph in agent_workflow.py: {e_compile}")
    traceback.print_exc()
    app_graph = None

async def run_agent(user_query: str, df: pd.DataFrame) -> dict:
    print(f"\n--- Running Agent Graph for Query: '{user_query}' ---")
    if app_graph is None:
        return {"response_type": "error", "content": "Agent graph not compiled.", "thinking_log_str": "Graph not compiled.", "error": "Graph compilation failed."}
    if openai_client is None: # Check if client itself is None
        return {"response_type": "error", "content": "OpenAI client not initialized.", "thinking_log_str": "OpenAI client None.", "error": "OpenAI client None."}
    if df is None or df.empty:
        return {"response_type": "error", "content": "No data provided to agent.", "thinking_log_str": "DataFrame empty.", "error": "DataFrame empty."}

    initial_thinking_log = [f"--- Agent Run Initiated for Query: '{user_query}' ---"]
    initial_state = AgentState(
        user_query=user_query,
        df_head_str=df.head().to_string(),
        df_columns=df.columns.tolist(),
        df_full=df,
        action_type=None,
        plot_config_json=None,
        plotly_fig_json=None,
        plot_insights=None,
        llm_response=None,
        error_message=None,
        thinking_log=initial_thinking_log
    )

    final_state: AgentState = initial_state
    try:
        final_state = await app_graph.ainvoke(initial_state, {"recursion_limit": 15})
    except Exception as e_invoke:
        log_so_far = final_state.get("thinking_log", initial_thinking_log)
        log_so_far.append(f"AGENT_CRITICAL_ERROR: Graph invoke failed: {type(e_invoke).__name__} - {str(e_invoke)}")
        traceback.print_exc()
        final_state["error_message"] = f"Graph execution error: {str(e_invoke)}"
        final_state["action_type"] = "error"
        final_state["llm_response"] = f"An internal error occurred while processing your request."
        final_state["thinking_log"] = log_so_far


    resp_type = final_state.get("action_type", "error")
    user_content = final_state.get("llm_response", "No specific response generated.")
    err_msg = final_state.get("error_message")

    if err_msg and resp_type != "fallback" and resp_type != "error":
        resp_type = "error"
        if not user_content or user_content == "No specific response generated.": user_content = err_msg
    elif not user_content and resp_type == "error" and not err_msg:
        user_content = "An unspecified error occurred."; err_msg = "Unspecified error."

    log_list = final_state.get("thinking_log", ["Log not available."])
    if not log_list: log_list = ["Log is empty."]

    response = {
        "response_type": resp_type,
        "content": user_content,
        "plotly_fig_json": final_state.get("plotly_fig_json"),
        "plot_config_json": final_state.get("plot_config_json"),
        "plot_insights": final_state.get("plot_insights"),
        "thinking_log_str": "\n".join(log_list),
        "error": err_msg
    }
    return response

# Example of how to run this if called directly (for testing)
# This part would typically not be in the agent_workflow.py if it's a module.
# It's here for completeness of a runnable example.
async def main_test():
    print("Starting main_test for agent_workflow.py")
    if openai_client is None:
        print("OpenAI client not initialized. Exiting test.")
        # Clean up custom http client if it was created
        if custom_http_client_for_openai:
            print("Closing custom HTTP client...")
            await custom_http_client_for_openai.aclose()
        return

    # Create a sample DataFrame
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Date of Birth': pd.to_datetime(['1990-01-01', '1985-05-15', '1992-07-20', '2000-11-30', '1988-03-10']),
        'Score': [85, 92, 78, 88, 95],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin']
    }
    sample_df = pd.DataFrame(sample_data)

    test_query = "Show me the age distribution"
    # test_query = "What is the average score?" # For QnA node

    print(f"Running agent with test query: '{test_query}'")
    result = await run_agent(test_query, sample_df)
    
    print("\n--- Agent Test Result ---")
    print(f"Response Type: {result.get('response_type')}")
    print(f"Content: {result.get('content')}")
    if result.get('plotly_fig_json'):
        print("Plotly Fig JSON (first 100 chars):", result['plotly_fig_json'][:100] + "...")
    if result.get('plot_config_json'):
        print("Plot Config JSON:", result['plot_config_json'])
    if result.get('plot_insights'):
        print("Plot Insights:", result['plot_insights'])
    if result.get('error'):
        print(f"Error: {result.get('error')}")
    # print("\nThinking Log:")
    # print(result.get('thinking_log_str'))
    print("--- End of Agent Test Result ---")

    # Clean up: Close the custom http client if it was created
    if custom_http_client_for_openai:
        print("Closing custom HTTP client...")
        await custom_http_client_for_openai.aclose()
    # Also close the main openai_client if it has an aclose method (AsyncAzureOpenAI should)
    if openai_client and hasattr(openai_client, 'close'): # Older versions might not have close
        print("Closing main OpenAI client...")
        await openai_client.close()
    elif openai_client and hasattr(openai_client, 'aclose'): # Newer versions use aclose
        print("Closing main OpenAI client (aclose)...")
        await openai_client.aclose()


if __name__ == "__main__":
    # Ensure that .env is in the parent directory relative to this script, or adjust path.
    # For this test to run, you'd execute `python agent_workflow.py` from the directory
    # where `agent_workflow.py` is located.
    
    # Check if .env variables are loaded.
    if not all([llm_azure_endpoint, llm_api_key, llm_deployment_name]):
         print("ERROR: One or more Azure OpenAI environment variables are not set.")
         print("Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME are in your .env file.")
    else:
        asyncio.run(main_test())

# --- END OF FILE agent_workflow.py ---