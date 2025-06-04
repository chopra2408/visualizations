import os
import pandas as pd
from typing import TypedDict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from backend.utils import generate_plot_from_config, calculate_age_from_dob
from backend.models import PlotConfig
import numpy as np
import traceback
import json

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

try:
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    print("LLM Initialized successfully in agent_workflow.py.")
except Exception as e:
    print(f"FATAL ERROR initializing LLM in agent_workflow.py: {e}")
    llm = None

class AgentState(TypedDict):
    user_query: str
    df_head_str: str
    df_columns: List[str]
    df_full: Optional[pd.DataFrame]
    action_type: Optional[str]
    plot_config_json: Optional[str]
    plotly_fig_json: Optional[str]    # For Plotly JSON
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

def generate_insights_for_plot(
    plot_config: PlotConfig, df_columns_list: List[str], df_head_sample: str,
    user_query_str: str, current_thinking_log: List[str]
) -> Tuple[Optional[str], List[str]]:
    thinking_log = list(current_thinking_log)
    if llm is None:
        thinking_log.append("INSIGHTS_ERROR: LLM not available.")
        return "Could not generate insights: LLM not available.", thinking_log
    desc = (f"An interactive '{plot_config.plot_type}' plot titled '{plot_config.title or 'N/A'}' was generated. " # Added "interactive"
            f"X-axis: '{plot_config.xlabel or plot_config.x_column or 'N/A'}'. "
            f"Y-axis: '{plot_config.ylabel or plot_config.y_column or 'N/A'}'. "
            f"Colored by: '{plot_config.color_by_column or 'N/A'}'.")
    if "Age_Derived_From_DOB" in desc: desc += " 'Age' was derived from Date of Birth."
    thinking_log.append(f"INSIGHTS_PROMPT_INFO: Plot Description for insights: {desc}")
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
    structured_llm_insights = llm.with_structured_output(PlotInsights, method="function_calling", include_raw=False)
    insights_chain = insights_prompt_template | structured_llm_insights
    try:
        res = insights_chain.invoke({
            "user_query_for_insights": user_query_str,
            "plot_description_for_insights": desc,
            "dataset_columns_for_insights": df_columns_list,
            "data_sample_for_insights": df_head_sample
        })
        txt = res.insights
        if res.suggestions: txt += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in res.suggestions)
        thinking_log.append(f"INSIGHTS_RESULT: Generated (first 100 chars): {txt[:100]}...")
        return txt, thinking_log
    except Exception as e:
        thinking_log.append(f"INSIGHTS_ERROR: {str(e)}")
        traceback.print_exc()
        return f"Could not generate insights: {str(e)}", thinking_log

def router_node(state: AgentState) -> AgentState:
    # (This function remains unchanged from your previous correct version)
    print("\n--- Router Node ---")
    current_log = state.get("thinking_log", [])
    current_log.extend(["--- Router Node: Initiated ---", f"Query: '{state['user_query']}'"])
    if llm is None:
        current_log.append("Router_ERROR: LLM not available.")
        return {**state, "action_type": "fallback", "error_message": "LLM unavailable.", "llm_response": "System error.", "thinking_log": current_log}
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert query router. Based on the user's query, data columns, and a sample of the data, determine the primary action required.
        The available actions are: 'visualize', 'query_data', 'fallback'.
        Data Context: Columns: {router_df_columns}, Data Sample (head): {router_df_head}
        User Query: "{router_user_query}"
        Provide your routing decision and a brief reasoning.
        """),
        ("human", "Determine the action: 'visualize', 'query_data', or 'fallback'.")
    ])
    chain = router_prompt | llm.with_structured_output(RouteQuery, method="function_calling", include_raw=False)
    try:
        res = chain.invoke({
            "router_df_columns": state["df_columns"], "router_df_head": state["df_head_str"], "router_user_query": state["user_query"]
        })
        current_log.append(f"Router Decision: Action='{res.action}', Reasoning='{res.reasoning}'")
        return {**state, "action_type": res.action, "llm_response": f"Okay, I will try to {res.action.replace('_', ' ')} for your query.", "thinking_log": current_log}
    except Exception as e:
        current_log.append(f"Router_ERROR: {str(e)}"); traceback.print_exc()
        return {**state, "action_type": "fallback", "error_message": str(e), "llm_response": f"Error in routing: {str(e)}", "thinking_log": current_log}


async def stream_pre_agent_summary(
    user_query: str, df_head_str: str, df_columns: List[str], intended_action: str
):
    # (This function remains unchanged from your previous correct version)
    print(f"\n--- Streaming Pre-Agent Summary for action: {intended_action} ---")
    if llm is None:
        yield json.dumps({"type": "error", "chunk": "LLM not available for pre-summary."}) + "\n"; return
    summary_prompt_system = f"""You are a helpful AI assistant providing a quick plan.
    Dataset columns: {df_columns}, Data sample (first 5 rows): {df_head_str}
    User query: "{user_query}", Your current intended action is: '{intended_action}'.
    Provide a concise, user-friendly summary of your plan (1-2 sentences).
    """
    pre_summary_prompt = ChatPromptTemplate.from_messages([("system", summary_prompt_system), ("human", "Briefly, what's your plan?")])
    summary_chain = pre_summary_prompt | llm
    try:
        async for chunk in summary_chain.astream({}): # No dynamic vars needed if all in system prompt
            if chunk.content is not None:
                yield json.dumps({"type": "thinking_process_update", "chunk": chunk.content}) + "\n"
    except Exception as e:
        print(f"Error during pre-agent summary streaming: {e}"); traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in pre-summary stream: {str(e)}"}) + "\n"

async def stream_qna_response(user_query: str, df_head_str: str, df_columns: List[str]):
    # (This function remains unchanged from your previous correct version)
    print("\n--- Streaming Q&A Response Directly ---")
    if llm is None:
        yield json.dumps({"type": "error", "chunk": "LLM not available for Q&A."}) + "\n"; return
    prompt_template_messages = [
        ("system", f"""You are a helpful AI assistant.
         Dataset columns: {df_columns}, Data sample (first 5 rows): {df_head_str}
         User question: "{user_query}"
         Answer concisely based *only* on the provided data sample and column names.
         If asked for simple calculations from the sample, try to provide an answer.
         For complex calculations or if full data is needed, state that. Do not make up data.
         """),
        ("human", "{user_query_for_qna}")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    qna_chain = prompt | llm
    try:
        async for chunk in qna_chain.astream({"user_query_for_qna": user_query}):
            if chunk.content is not None: yield json.dumps({"type": "content", "chunk": chunk.content}) + "\n"
    except Exception as e:
        print(f"Error during Q&A streaming in agent_workflow: {e}"); traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in Q&A stream: {str(e)}"}) + "\n"


def visualization_node(state: AgentState) -> AgentState:
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

    df_to_plot = df_full.copy() # Operate on a copy
    thinking_log.append(f"VIZ_INFO: User Query: '{user_query}'")
    thinking_log.append(f"VIZ_INFO: Available columns: {df_columns_list}")

    # 1. Programmatic Age Distribution Check
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
                df_to_plot[derived_age_col_name] = age_series # Add derived age to df_to_plot
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
        if llm is None:
            thinking_log.append("VIZ_LLM_ERROR: LLM is not available.")
            return {**state, "error_message": "LLM not available for plotting.", "llm_response": "System error: Cannot determine plot settings.", "thinking_log": thinking_log}

        thinking_log.append("VIZ_LLM_ACTION: Using LLM to determine plot configuration.")
        actual_dob_col_for_llm = next((c for c in df_columns_list if c.lower() in ["date of birth", "dob", "birth date", "birthdate"]), None)
        dob_info_for_llm = "No specific Date of Birth or Age column identified that could be used for age derivation for the LLM."
        # If age was already derived programmatically, it's in df_to_plot.columns, so LLM can use it.
        # We still inform LLM about original DOB column for context.
        if derived_age_col_name in df_to_plot.columns:
             dob_info_for_llm = (f"An 'Age' column ('{derived_age_col_name}') has already been derived and is available. "
                                 f"If 'Age' is requested for plotting, use '{derived_age_col_name}'.")
        elif actual_dob_col_for_llm:
            dob_info_for_llm = (f"A 'Date of Birth' column ('{actual_dob_col_for_llm}') is present. "
                                f"If 'Age' is requested for plotting, use '{derived_age_col_name}' as the column name; the system can derive it if this column is selected.")
        thinking_log.append(f"VIZ_LLM_INFO: DOB/Age info for LLM prompt: {dob_info_for_llm}")

        # Use df_full for cardinalities as it's the original, df_to_plot might have new derived cols
        cardinality_info_dict = {col: df_full[col].nunique() for col in df_columns_list if col in df_full}
        cardinality_prompt_str = "\n".join([f"  - Column '{col}': {count} unique values" for col, count in cardinality_info_dict.items()])
        thinking_log.append(f"VIZ_LLM_INFO: Cardinality for LLM (passed as variable):\n{cardinality_prompt_str}")
        derived_age_col_name_in_prompt = derived_age_col_name

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

1.  Understand User Intent: What is the user trying to see? Extract key entities.
2.  X-axis (x_column): Prioritize user spec. For distributions: variable itself. Comparisons: categorical. Trends: time/ordered. Relationships: numerical.
3.  Y-axis (y_column): Prioritize user spec. Bar/line (aggregated): numerical. Bar (counts): null. Scatter: numerical. Histogram/KDE/Boxplot (distribution of): numerical.
4.  Color/Hue/Stacking (color_by_column): Categorical. Low cardinality (2-5) good for grouped/hue. Moderate (6-12) prefer STACKED bar, hue ok for scatter/line. High (>12) for bar: STACKED strongly preferred; for others: hue very challenging.
5.  Plot Type Selection ('plot_type'):
    - 'histogram'/'kde': Single NUMERICAL distribution (x_column).
    - 'bar': Compare CATEGORICAL groups (x_column). y_column (numerical) for aggregated values, null for counts.
    - 'line': Trends (x_column often time/ordered, y_column numerical).
    - 'scatter': Relationship between two NUMERICAL variables (x_column, y_column).
    - 'boxplot': Compare NUMERICAL (y_column) distributions across CATEGORICAL (x_column) groups, or single numerical (y_column only).
    - 'auto_categorical': For categorical x_column, system chooses best bar-like plot (likely 'bar').
6.  Generic Plot Requests: For 'bar'/'auto_categorical', pick low-moderate cardinality CATEGORICAL x_column (counts). For 'histogram'/'kde', pick NUMERICAL x_column (consider derived '{derived_age_col_name_in_prompt}'). Try to infer reasonable defaults.
7.  Titles and Labels: Descriptive title. xlabel/ylabel from columns or 'Count'/'Frequency'/'Density'.
Readability is KEY. If '{derived_age_col_name}' is used and derivable, it's numerical 'Age'.

Available plot types in PlotConfig: "bar", "line", "scatter", "histogram", "boxplot", "kde", "auto_categorical".
Ensure all fields in PlotConfig are populated if they are relevant to the chosen plot type.
"""
        viz_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_for_plot_config),
            ("human", "Based on my request and the data context provided, suggest the most suitable and readable plot configuration using the PlotConfig schema.")
        ])
        chain_viz = viz_prompt_template | llm.with_structured_output(PlotConfig, method="function_calling", include_raw=False)
        try:
            plot_config_obj_llm = chain_viz.invoke({
                "prompt_df_columns": df_columns_list, # Original columns
                "prompt_df_head": df_head_sample,
                "prompt_cardinalities": cardinality_prompt_str,
                "prompt_dob_info": dob_info_for_llm,
                "prompt_user_query": user_query,
                "derived_age_col_name": derived_age_col_name
            })
            plot_config_obj = plot_config_obj_llm # Assign if successful
            thinking_log.append(f"VIZ_LLM_CONFIG: PlotConfig from LLM: {plot_config_obj.model_dump_json(indent=0) if plot_config_obj else 'None'}")

            # If LLM suggests derived age AND it wasn't derived programmatically earlier AND DOB col exists
            if plot_config_obj and actual_dob_col_for_llm and \
               (plot_config_obj.x_column == derived_age_col_name or plot_config_obj.y_column == derived_age_col_name) and \
               derived_age_col_name not in df_to_plot.columns: # Check if not already in df_to_plot
                thinking_log.append(f"VIZ_LLM_ACTION: LLM suggested '{derived_age_col_name}'. Attempting derivation from '{actual_dob_col_for_llm}'.")
                age_series = calculate_age_from_dob(df_to_plot, actual_dob_col_for_llm) # Derive into df_to_plot
                if age_series is not None and not age_series.isnull().all() and pd.api.types.is_numeric_dtype(age_series):
                    df_to_plot[derived_age_col_name] = age_series
                    thinking_log.append(f"VIZ_LLM_SUCCESS: Derived '{derived_age_col_name}' for LLM-suggested plot. Dtype: {age_series.dtype}.")
                else:
                    thinking_log.append(f"VIZ_LLM_ERROR: LLM suggested derived age from '{actual_dob_col_for_llm}', but derivation failed or was non-numeric. Plot may fail.")
                    if plot_config_obj.plot_type in ["histogram", "kde"] and \
                       (plot_config_obj.x_column == derived_age_col_name or plot_config_obj.y_column == derived_age_col_name):
                        plot_config_obj = None # Invalidate if critical age derivation failed
                        thinking_log.append(f"VIZ_LLM_ERROR: Invalidated PlotConfig due to failed critical age derivation for {plot_config_obj.plot_type if plot_config_obj else 'N/A'}.")
        except Exception as e_llm_viz:
            thinking_log.append(f"VIZ_LLM_ERROR: Failed to get PlotConfig from LLM: {type(e_llm_viz).__name__}: {str(e_llm_viz)}")
            traceback.print_exc()
            plot_config_obj = None # Ensure it's None on LLM error

    # 3. Final check for PlotConfig and Validation
    if plot_config_obj is None:
        thinking_log.append("VIZ_ERROR: PlotConfig is None after all attempts (programmatic and LLM).")
        return {**state, "error_message": "Could not determine how to configure the plot for your request.", "llm_response": "I'm unable to create the visualization as I couldn't determine the necessary settings.", "thinking_log": thinking_log}

    thinking_log.append(f"VIZ_VALIDATE_INPUT: Validating final PlotConfig: {plot_config_obj.model_dump_json(indent=0)}")
    # df_to_plot now contains any derived columns
    thinking_log.append(f"VIZ_VALIDATE_INPUT: Columns in df_to_plot for validation: {df_to_plot.columns.tolist()}")

    # Validation (ensure selected columns exist in df_to_plot)
    required_cols_for_plot = [col for col in [plot_config_obj.x_column, plot_config_obj.y_column, plot_config_obj.color_by_column] if col]
    missing_cols = [col for col in required_cols_for_plot if col not in df_to_plot.columns]
    if missing_cols:
        msg = f"Column(s) selected for plotting not found in the available data: {', '.join(missing_cols)}. Available: {df_to_plot.columns.tolist()}"
        thinking_log.append(f"VIZ_VALIDATE_ERROR: {msg}")
        return {**state, "error_message": msg, "llm_response": msg, "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json()}

    if plot_config_obj.plot_type in ["bar", "histogram", "kde", "line", "scatter", "auto_categorical", "boxplot"]:
        if not plot_config_obj.x_column and not (plot_config_obj.plot_type == "boxplot" and plot_config_obj.y_column):
            msg = f"X-axis column is required for plot type '{plot_config_obj.plot_type}' (unless boxplot of single Y) but was not specified."
            thinking_log.append(f"VIZ_VALIDATE_ERROR: {msg}")
            return {**state, "error_message": msg, "llm_response": msg, "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json()}


    final_plot_df = df_to_plot # This df has the derived columns if any

    # 4. Generate plot (Plotly JSON string)
    thinking_log.append(f"VIZ_PLOT_ATTEMPT: Generating interactive plot with final config: {plot_config_obj.model_dump_json(indent=0)}")
    try:
        plotly_json_string = generate_plot_from_config(final_plot_df, plot_config_obj)

        if plotly_json_string:
            thinking_log.append("VIZ_PLOT_SUCCESS: Interactive Plotly JSON string generated successfully.")
            llm_user_message = f"Here is your interactive {plot_config_obj.plot_type} plot"
            if plot_config_obj.title: llm_user_message += f" titled: '{plot_config_obj.title}'."
            else: llm_user_message += "."

            insights_text, thinking_log_after_insights = generate_insights_for_plot(
                plot_config_obj, df_columns_list, df_head_sample, user_query, thinking_log # Pass original df_columns_list for insights context
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

# qna_node, fallback_node, decide_next_node remain the same
def qna_node(state: AgentState) -> AgentState:
    # ... (no changes from your previous version)
    current_log = state.get("thinking_log", [])
    current_log.append("--- QnA Node: Initiated ---")
    if llm is None:
        current_log.append("QnA_ERROR: LLM not available.")
        return {**state, "error_message": "LLM unavailable.", "llm_response": "LLM error.", "thinking_log": current_log}
    qna_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the provided column names and data sample.\nColumns: {qna_df_cols}\nSample Data:\n{qna_df_head}"),
        ("human", "{qna_user_query}")
    ])
    chain_qna = qna_prompt | llm
    try:
        res = chain_qna.invoke({
            "qna_df_cols": state["df_columns"], "qna_df_head": state["df_head_str"], "qna_user_query": state["user_query"]
        })
        current_log.append(f"QnA_SUCCESS: Response generated.")
        return {**state, "llm_response": res.content, "thinking_log": current_log, "error_message": None}
    except Exception as e:
        current_log.append(f"QnA_ERROR: {str(e)}"); traceback.print_exc()
        return {**state, "error_message": str(e), "llm_response": "Error in QnA processing.", "thinking_log": current_log}

def fallback_node(state: AgentState) -> AgentState:
    # ... (no changes from your previous version)
    log = state.get("thinking_log", [])
    log.append("--- Fallback Node: Initiated ---")
    user_msg = "I'm sorry, I could not fully process that request."
    if state.get("error_message"): user_msg += f" Details: {state.get('error_message')}"; log.append(f"Fallback_REASON: Error: {state.get('error_message')}")
    else: log.append("Fallback_REASON: Query off-topic, router default, or unhandled action.")
    return_state = {**state, "llm_response": user_msg, "action_type": "fallback", "thinking_log": log}
    if not return_state.get("error_message"): return_state["error_message"] = "Fell back due to unhandled query or internal issue."
    return return_state

def decide_next_node(state: AgentState) -> str:
    # ... (no changes from your previous version)
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
workflow.add_node("fallback_agent", fallback_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next_node, {
    "visualization_agent": "visualization_agent", "qna_agent": "qna_agent", "fallback_agent": "fallback_agent"
})
workflow.add_edge("visualization_agent", END)
workflow.add_edge("qna_agent", END)
workflow.add_edge("fallback_agent", END)

try:
    app_graph = workflow.compile()
    print("LangGraph compiled successfully in agent_workflow.py.")
except Exception as e_compile:
    print(f"FATAL ERROR compiling LangGraph in agent_workflow.py: {e_compile}")
    traceback.print_exc()
    app_graph = None

def run_agent(user_query: str, df: pd.DataFrame) -> dict:
    print(f"\n--- Running Agent Graph for Query: '{user_query}' ---")
    if app_graph is None:
        return {"response_type": "error", "content": "Agent graph not compiled.", "thinking_log_str": "Graph not compiled.", "error": "Graph compilation failed."}
    if df is None or df.empty:
        return {"response_type": "error", "content": "No data provided to agent.", "thinking_log_str": "DataFrame empty.", "error": "DataFrame empty."}

    initial_thinking_log = [f"--- Agent Run Initiated for Query: '{user_query}' ---"]
    initial_state = AgentState(
        user_query=user_query,
        df_head_str=df.head().to_string(),
        df_columns=df.columns.tolist(),
        df_full=df, # Pass the full DataFrame
        action_type=None,
        plot_config_json=None,
        plotly_fig_json=None, # Initialize for Plotly
        plot_insights=None,
        llm_response=None,
        error_message=None,
        thinking_log=initial_thinking_log
    )

    final_state: AgentState = initial_state
    try:
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 15})
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

    if err_msg and resp_type != "fallback":
        resp_type = "error"
        if not user_content or user_content == "No specific response generated.": user_content = err_msg
    elif not user_content and resp_type == "error" and not err_msg:
        user_content = "An unspecified error occurred."; err_msg = "Unspecified error."

    log_list = final_state.get("thinking_log", ["Log not available."])
    if not log_list: log_list = ["Log is empty."]

    response = {
        "response_type": resp_type,
        "content": user_content,
        "plotly_fig_json": final_state.get("plotly_fig_json"), # Use Plotly JSON
        "plot_config_json": final_state.get("plot_config_json"),
        "plot_insights": final_state.get("plot_insights"),
        "thinking_log_str": "\n".join(log_list),
        "error": err_msg
    }
    return response
# --- END OF FILE agent_workflow.py ---