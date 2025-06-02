# --- START OF FILE agent_workflow.py ---
import os
import pandas as pd
from typing import TypedDict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
# Ensure your utils and models can be imported.
from utils import generate_plot_from_config, calculate_age_from_dob
from models import PlotConfig # PlotConfig is used here
import numpy as np
import traceback
import json

# Load environment variables from .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Ensure LLM is initialized
try:
    llm = ChatOpenAI(model="gpt-4.1", temperature=0) # Or your preferred model
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
    plot_image_bytes: Optional[bytes]
    plot_insights: Optional[str]
    llm_response: Optional[str]      # Primary response message to user (can be different from insights)
    error_message: Optional[str]
    thinking_log: List[str]

class RouteQuery(BaseModel):
    action: str = Field(description="Must be 'visualize', 'query_data', or 'fallback'.")
    reasoning: str = Field(description="Brief explanation for the chosen action.")

class PlotInsights(BaseModel):
    insights: str = Field(description="A concise textual summary of what the plot shows and any key observations or conclusions that can be drawn from it.")
    suggestions: Optional[List[str]] = Field(None, description="Optional: 1-2 follow-up questions or related plots.")

def generate_insights_for_plot(
    plot_config: PlotConfig, df_columns: List[str], df_head_str: str,
    user_query: str, current_thinking_log: List[str]
) -> Tuple[Optional[str], List[str]]:
    thinking_log = list(current_thinking_log)
    if llm is None:
        thinking_log.append("INSIGHTS_ERROR: LLM not available.")
        return "Could not generate insights: LLM not available.", thinking_log

    desc = (f"A '{plot_config.plot_type}' plot titled '{plot_config.title or 'N/A'}' was generated. "
            f"X-axis: '{plot_config.xlabel or plot_config.x_column or 'N/A'}'. "
            f"Y-axis: '{plot_config.ylabel or plot_config.y_column or 'N/A'}'. "
            f"Colored by: '{plot_config.color_by_column or 'N/A'}'.")
    if "Age_Derived_From_DOB" in desc: desc += " 'Age' was derived from Date of Birth."
    thinking_log.append(f"INSIGHTS_PROMPT_INFO: Plot Description: {desc}")

    prompt_template_messages = [
        ("system", f"""You are an expert data analyst. A plot was generated.
         User Query: "{user_query}"
         Plot Description: {desc}
         Dataset Columns: {df_columns}
         Data Sample (head): {df_head_str}
         Provide concise insights based ONLY on the plot description and data context.
         What key takeaways, patterns, or distributions might this plot reveal?
         Do NOT hallucinate specific data values. Focus on typical interpretations.
         Example: For age distribution, mention age groups, skewness. For bar charts, compare categories.
         """), # REPLACE with your full, tested insights prompt
        ("human", "Please provide insights for the generated plot.")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    structured_llm_insights = llm.with_structured_output(PlotInsights, method="function_calling", include_raw=False)
    insights_chain = prompt | structured_llm_insights
    try:
        res = insights_chain.invoke({})
        txt = res.insights
        if res.suggestions: txt += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in res.suggestions)
        thinking_log.append(f"INSIGHTS_RESULT: Generated (first 100 chars): {txt[:100]}...")
        return txt, thinking_log
    except Exception as e:
        thinking_log.append(f"INSIGHTS_ERROR: {str(e)}")
        return f"Could not generate insights: {str(e)}", thinking_log

def router_node(state: AgentState) -> AgentState:
    print("\n--- Router Node ---")
    log = ["--- Router Node: Initiated ---", f"Query: '{state['user_query']}'"]
    if llm is None:
        log.append("Router_ERROR: LLM not available.")
        return {"action_type": "fallback", "error_message": "LLM unavailable.", "llm_response": "System error.", "thinking_log": log}
    
    prompt_msgs = [("system", f"""You are an expert query router...
        Columns: {state['df_columns']}
        Sample: {state['df_head_str']}
        Determine action: 'visualize', 'query_data', or 'fallback'.
        """), # REPLACE with your full router prompt
        ("human", "{user_query}")]
    chain = ChatPromptTemplate.from_messages(prompt_msgs) | llm.with_structured_output(RouteQuery, include_raw=False)
    try:
        res = chain.invoke({"user_query": state["user_query"]})
        log.append(f"Router Decision: Action='{res.action}', Reasoning='{res.reasoning}'")
        return {"action_type": res.action, "llm_response": f"Routing to {res.action}.", "thinking_log": log}
    except Exception as e:
        log.append(f"Router_ERROR: {str(e)}")
        return {"action_type": "fallback", "error_message": str(e), "llm_response": f"Router error: {str(e)}", "thinking_log": log}

async def stream_pre_agent_summary(
    user_query: str, df_head_str: str, df_columns: List[str], intended_action: str
):
    """
    Streams a preliminary summary/plan before the main agent graph runs.
    """
    print(f"\n--- Streaming Pre-Agent Summary for action: {intended_action} ---")
    if llm is None:
        yield json.dumps({"type": "error", "chunk": "LLM not available for pre-summary."}) + "\n"
        return

    summary_prompt_system = f"""You are a helpful AI assistant.
    The user has uploaded a dataset and made a query.
    Dataset columns: {df_columns}
    Data sample (first 5 rows):
    {df_head_str}
    User query: "{user_query}"
    Your current intended action is: '{intended_action}'.

    Based on this, provide a concise, user-friendly summary of what you plan to do or what initial insights you might look for.
    This is a preliminary step. Be brief and set expectations.
    For example, if the action is 'visualize', you might say:
    'Okay, I'll try to create a visualization for that. I'll analyze your query to determine the best chart type and data to display.'
    If the action is 'query_data' (though this function might not be called for it if QnA has its own direct stream):
    'Understood. I'll look into your data sample to answer your question about "{user_query[:30]}...".'
    If the action is 'fallback':
    'I'll try to understand your request: "{user_query[:30]}..." and respond accordingly.'
    Keep it to 1-2 sentences.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", summary_prompt_system),
        ("human", "Briefly, what's your plan?") # Placeholder, system prompt has all info
    ])
    summary_chain = prompt | llm

    try:
        async for chunk in summary_chain.astream({
            # Variables are already in the system prompt, but can be passed if structure changes
            # "user_query": user_query, # etc.
        }):
            if chunk.content is not None:
                yield json.dumps({"type": "pre_summary_chunk", "chunk": chunk.content}) + "\n"
    except Exception as e:
        print(f"Error during pre-agent summary streaming: {e}")
        traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in pre-summary stream: {str(e)}"}) + "\n"
        
async def stream_qna_response(user_query: str, df_head_str: str, df_columns: List[str]):
    """
    Streams the Q&A response directly using Langchain's astream.
    This function will be called by FastAPI for the Q&A streaming path.
    It does not participate in the graph's thinking_log state directly.
    """
    print("\n--- Streaming Q&A Response Directly (agent_workflow.stream_qna_response) ---")
    if llm is None:
        yield json.dumps({"type": "error", "chunk": "LLM not available for Q&A."}) + "\n" # Send as JSON chunk
        return

    # REPLACE with your full, tested QnA prompt for streaming
    prompt_template_messages = [
        ("system",
         f"""You are a helpful AI assistant. The user has uploaded a dataset and has a question.
         Dataset columns: {df_columns}
         Data sample (first 5 rows):
         {df_head_str}
         User question: "{user_query}"
         Answer concisely based *only* on the provided data sample and column names.
         If asked for specific calculations (mean, median, mode) for a column from the sample, try to provide an answer if it's simple.
         For complex calculations or if the full data is needed, state that more processing would be required or that you can only see a sample.
         Do not make up data. If the answer cannot be found in the sample, say so.
         """),
        ("human", "{user_query}") # user_query is passed in the .astream() call below
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_template_messages)
    qna_chain = prompt | llm

    try:
        async for chunk in qna_chain.astream({
            "df_columns": df_columns, # These are already in the system prompt but can be passed if prompt structure changes
            "df_head": df_head_str,   # Same as above
            "user_query": user_query  # This is the primary input variable for the human message
        }):
            if chunk.content is not None:
                # Yield JSON chunks for consistency with FastAPI endpoint structure
                yield json.dumps({"type": "content", "chunk": chunk.content}) + "\n"
    except Exception as e:
        print(f"Error during Q&A streaming in agent_workflow: {e}")
        traceback.print_exc()
        yield json.dumps({"type": "error", "chunk": f"Error in Q&A stream: {str(e)}"}) + "\n"

# ... (rest of your agent_workflow.py: router_node, visualization_node, etc.) ...
# ... (other imports, AgentState, PlotConfig, PlotInsights, generate_insights_for_plot, llm etc. must be defined above) ...

def visualization_node(state: AgentState) -> AgentState:
    print("\n--- Visualization Node (Comprehensive Example) ---")
    thinking_log = state.get("thinking_log", []) 
    thinking_log.append("--- Visualization Node: Initiated ---")

    user_query = state["user_query"]
    user_query_lower = user_query.lower()
    df_columns = state["df_columns"]
    df_full = state["df_full"]
    
    plot_config_obj: Optional[PlotConfig] = None
    plot_bytes: Optional[bytes] = None
    llm_user_message: str = "Plot processing started."
    final_plot_df: pd.DataFrame # DataFrame that will be passed to generate_plot_from_config

    if df_full is None or df_full.empty:
        thinking_log.append("VIZ_ERROR: DataFrame is empty or None at node start.")
        return {
            "error_message": "No data available to visualize.",
            "llm_response": "Cannot create a plot because the data is missing or empty.",
            "thinking_log": thinking_log,
            "action_type": "visualize" # Or "error"
        }
    
    df_to_plot = df_full.copy() # Work on a copy to allow modifications like adding derived age
    derived_age_col_name = "Age_Derived_From_DOB"

    thinking_log.append(f"VIZ_INFO: User Query: '{user_query}'")
    thinking_log.append(f"VIZ_INFO: Available columns: {df_columns}")

    # 1. Determine if it's a specific, programmatically handled plot (e.g., age distribution)
    age_distribution_keywords = [
        "age distribution", "distribution of age", "age bell curve", "bell curve of age",
        "histogram of age", "age histogram", "kde of age", "age kde",
        "distribution of ages", "bell curve of ages", "age density plot"
    ] # Customize these keywords
    user_wants_age_distribution = any(kw in user_query_lower for kw in age_distribution_keywords)
    thinking_log.append(f"VIZ_CHECK: User wants age distribution? {user_wants_age_distribution}")

    if user_wants_age_distribution:
        log_prefix = "VIZ_AGEDIST_"
        thinking_log.append(f"{log_prefix}Attempting programmatic age distribution plot.")
        actual_dob_col = next((c for c in df_columns if c.lower() in ["date of birth", "dob", "birth date", "birthdate"]), None)
        existing_age_col = next((c for c in df_columns if c.lower() == "age"), None)
        thinking_log.append(f"{log_prefix}DOB Column found: '{actual_dob_col}', Existing 'Age' Column: '{existing_age_col}'")
        
        age_col_to_use = None
        is_derived_age = False

        if existing_age_col and existing_age_col in df_to_plot.columns and pd.api.types.is_numeric_dtype(df_to_plot[existing_age_col]):
            age_col_to_use = existing_age_col
            thinking_log.append(f"{log_prefix}Using existing numeric 'Age' column: '{existing_age_col}'.")
        elif actual_dob_col:
            thinking_log.append(f"{log_prefix}Attempting to derive age from DOB column: '{actual_dob_col}'.")
            age_series = calculate_age_from_dob(df_to_plot, actual_dob_col) # from utils.py
            if age_series is not None and not age_series.isnull().all() and pd.api.types.is_numeric_dtype(age_series):
                df_to_plot[derived_age_col_name] = age_series # Add to the copied DataFrame
                age_col_to_use = derived_age_col_name
                is_derived_age = True
                thinking_log.append(f"{log_prefix}Successfully derived '{derived_age_col_name}'. Dtype: {age_series.dtype}.")
            else:
                thinking_log.append(f"{log_prefix}ERROR: Failed to derive numeric age from '{actual_dob_col}'. Series dtype: {age_series.dtype if age_series is not None else 'None'}.")
        
        if age_col_to_use:
            plot_type_for_age = "kde" if "bell curve" in user_query_lower or "kde" in user_query_lower else "histogram"
            plot_config_obj = PlotConfig(
                plot_type=plot_type_for_age,
                x_column=age_col_to_use,
                title=f"Age Distribution{' (Derived)' if is_derived_age else ''}{' (Bell Curve)' if plot_type_for_age == 'kde' else ''}",
                xlabel=f"Age{' (Derived)' if is_derived_age else ''}",
                ylabel="Density" if plot_type_for_age == "kde" else "Frequency"
            )
            thinking_log.append(f"{log_prefix}Programmatically set PlotConfig: {plot_config_obj.model_dump_json(indent=0)}")
        else:
            thinking_log.append(f"{log_prefix}ERROR: Could not find or derive a suitable numeric age column. Will try LLM.")
            # plot_config_obj remains None, will fall through to LLM
    
    # 2. If not a special case that set plot_config_obj, or if special case failed, use LLM
    if plot_config_obj is None:
        if llm is None:
            thinking_log.append("VIZ_LLM_ERROR: LLM is not available to determine plot configuration.")
            # Cannot proceed without a config
            return {
                "error_message": "LLM not available for plotting.", 
                "llm_response": "System error: Cannot determine plot settings.", 
                "thinking_log": thinking_log, "action_type": "visualize"
            }

        thinking_log.append("VIZ_LLM_ACTION: Using LLM to determine plot configuration.")
        actual_dob_col_for_llm = next((c for c in df_columns if c.lower() in ["date of birth", "dob", "birth date", "birthdate"]), None)
        existing_age_col_for_llm = next((c for c in df_columns if c.lower() == "age"), None)
        
        dob_info_for_llm = "No specific Date of Birth or Age column identified for LLM guidance."
        if existing_age_col_for_llm:
            dob_info_for_llm = f"An 'Age' column ('{existing_age_col_for_llm}') is present."
        elif actual_dob_col_for_llm:
            dob_info_for_llm = (f"A 'Date of Birth' column ('{actual_dob_col_for_llm}') is present. "
                                f"If 'Age' is requested for plotting, use '{derived_age_col_name}' as the column name; "
                                "the system can derive it.")
        thinking_log.append(f"VIZ_LLM_INFO: DOB/Age info for LLM prompt: {dob_info_for_llm}")

        prompt_msgs_viz = [
            ("system", f"""You are a data visualization expert. Determine plot parameters (PlotConfig schema).
             Dataset columns: {df_columns}
             Data sample (first 5 rows): {state["df_head_str"]}
             {dob_info_for_llm}
             User request: "{user_query}"
             Prioritize numerical X for histogram/KDE. If using '{derived_age_col_name}', it will be numeric.
             Avoid using ID columns for distributions unless explicitly asked.
             Available plot types: "bar", "line", "scatter", "histogram", "boxplot", "kde".
             """),
            ("human", "Suggest plot configuration based on my request.")
        ]
        chain_viz = ChatPromptTemplate.from_messages(prompt_msgs_viz) | llm.with_structured_output(PlotConfig, method="function_calling", include_raw=False)
        try:
            plot_config_obj = chain_viz.invoke({})
            thinking_log.append(f"VIZ_LLM_CONFIG: PlotConfig from LLM: {plot_config_obj.model_dump_json(indent=0) if plot_config_obj else 'None'}")

            # If LLM suggests derived age, and it's not yet in df_to_plot, derive it.
            if plot_config_obj and actual_dob_col_for_llm and \
               (plot_config_obj.x_column == derived_age_col_name or plot_config_obj.y_column == derived_age_col_name) and \
               derived_age_col_name not in df_to_plot.columns:
                thinking_log.append(f"VIZ_LLM_ACTION: LLM suggested '{derived_age_col_name}'. Attempting derivation from '{actual_dob_col_for_llm}'.")
                age_series = calculate_age_from_dob(df_to_plot, actual_dob_col_for_llm)
                if age_series is not None and not age_series.isnull().all() and pd.api.types.is_numeric_dtype(age_series):
                    df_to_plot[derived_age_col_name] = age_series
                    thinking_log.append(f"VIZ_LLM_SUCCESS: Derived '{derived_age_col_name}' for LLM-suggested plot. Dtype: {age_series.dtype}.")
                else:
                    thinking_log.append(f"VIZ_LLM_ERROR: LLM suggested derived age from '{actual_dob_col_for_llm}', but derivation failed or was non-numeric. Plot may fail.")
                    # May need to invalidate plot_config_obj or attempt fallback if this column was critical
                    if plot_config_obj.x_column == derived_age_col_name and plot_config_obj.plot_type in ["histogram", "kde"]:
                        plot_config_obj = None # Invalidate if critical derivation failed
                        thinking_log.append(f"VIZ_LLM_ERROR: Invalidated PlotConfig due to failed critical age derivation for histo/kde.")


        except Exception as e_llm_viz:
            thinking_log.append(f"VIZ_LLM_ERROR: Failed to get PlotConfig from LLM: {str(e_llm_viz)}")
            plot_config_obj = None # Ensure it's None if LLM call fails

    # 3. Final check for PlotConfig and Validation
    if plot_config_obj is None:
        thinking_log.append("VIZ_ERROR: PlotConfig is None after all attempts (programmatic and LLM).")
        return {
            "error_message": "Could not determine how to configure the plot for your request.",
            "llm_response": "I'm unable to create the visualization as I couldn't determine the necessary settings.",
            "thinking_log": thinking_log, "action_type": "visualize"
        }

    thinking_log.append(f"VIZ_VALIDATE_INPUT: Validating final PlotConfig: {plot_config_obj.model_dump_json(indent=0)}")
    thinking_log.append(f"VIZ_VALIDATE_INPUT: Columns in df_to_plot for validation: {df_to_plot.columns.tolist()}")

    # Example Validation: Ensure x_column for histogram/KDE is numeric and exists
    if plot_config_obj.plot_type in ["histogram", "kde"]:
        if not plot_config_obj.x_column:
            thinking_log.append(f"VIZ_VALIDATE_ERROR: X-column missing for {plot_config_obj.plot_type}.")
            # Attempt to find a fallback or error out
            # For simplicity, error out here. A more robust system might try to pick one.
            return {"error_message": f"X-axis column is required for {plot_config_obj.plot_type} but was not specified.",
                    "llm_response": f"I need an X-axis column to create a {plot_config_obj.plot_type}.",
                    "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json(), "action_type": "visualize"}

        if plot_config_obj.x_column not in df_to_plot.columns:
            thinking_log.append(f"VIZ_VALIDATE_ERROR: X-column '{plot_config_obj.x_column}' for {plot_config_obj.plot_type} not found in DataFrame columns: {df_to_plot.columns.tolist()}.")
            return {"error_message": f"Column '{plot_config_obj.x_column}' needed for the plot was not found in the data.",
                    "llm_response": f"The column '{plot_config_obj.x_column}' required for the {plot_config_obj.plot_type} is missing.",
                    "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json(), "action_type": "visualize"}
        
        if not pd.api.types.is_numeric_dtype(df_to_plot[plot_config_obj.x_column]):
            original_x = plot_config_obj.x_column
            thinking_log.append(f"VIZ_VALIDATE_WARNING: X-column '{original_x}' (dtype: {df_to_plot[original_x].dtype}) is not numeric for {plot_config_obj.plot_type}. Attempting fallback.")
            # Fallback: find first available numeric column not typically an ID
            numeric_cols = df_to_plot.select_dtypes(include=np.number).columns.tolist()
            preferred_numeric = [col for col in numeric_cols if not any(id_kw in col.lower() for id_kw in ['id', 'employeeid', 'number', 'index'])]
            
            if preferred_numeric:
                plot_config_obj.x_column = preferred_numeric[0]
                thinking_log.append(f"VIZ_VALIDATE_FALLBACK: Switched X-column from non-numeric '{original_x}' to '{plot_config_obj.x_column}'.")
            elif numeric_cols: # If only ID-like numerics, use one
                plot_config_obj.x_column = numeric_cols[0]
                thinking_log.append(f"VIZ_VALIDATE_FALLBACK: Switched X-column from non-numeric '{original_x}' to ID-like numeric '{plot_config_obj.x_column}'.")
            else:
                thinking_log.append(f"VIZ_VALIDATE_ERROR: No numeric columns found for fallback for {plot_config_obj.plot_type} after '{original_x}' was non-numeric.")
                return {"error_message": f"Column '{original_x}' is not numeric, and no other numeric columns are available for a {plot_config_obj.plot_type} plot.",
                        "llm_response": f"A {plot_config_obj.plot_type} requires a numeric X-axis, but '{original_x}' isn't numeric and I couldn't find a substitute.",
                        "thinking_log": thinking_log, "plot_config_json": plot_config_obj.model_dump_json(), "action_type": "visualize"}
    
    final_plot_df = df_to_plot # The DataFrame that has all necessary columns (e.g., derived age)

    # 4. Generate plot image
    thinking_log.append(f"VIZ_PLOT_ATTEMPT: Generating plot with final config: {plot_config_obj.model_dump_json(indent=0)}")
    thinking_log.append(f"VIZ_PLOT_ATTEMPT: Using DataFrame with columns: {final_plot_df.columns.tolist()}")
    try:
        plot_bytes = generate_plot_from_config(final_plot_df, plot_config_obj) # from utils.py
        
        if plot_bytes:
            thinking_log.append("VIZ_PLOT_SUCCESS: Plot image bytes generated successfully.")
            llm_user_message = f"Here is your {plot_config_obj.plot_type} plot"
            if plot_config_obj.title:
                llm_user_message += f" titled: '{plot_config_obj.title}'."
            else:
                llm_user_message += "."
            
            insights_text, thinking_log = generate_insights_for_plot(
                plot_config_obj, df_columns, state["df_head_str"], user_query, thinking_log
            )
            return {
                "plot_config_json": plot_config_obj.model_dump_json(),
                "plot_image_bytes": plot_bytes,
                "llm_response": llm_user_message,
                "plot_insights": insights_text,
                "thinking_log": thinking_log,
                "action_type": "visualize"
            }
        else: # generate_plot_from_config returned None
            thinking_log.append(f"VIZ_PLOT_ERROR: generate_plot_from_config returned None. Config was: {plot_config_obj.model_dump_json(indent=0)}")
            return {
                "error_message": "Plot generation function failed to produce an image.", 
                "llm_response": f"I tried to create a {plot_config_obj.plot_type} plot, but there was an issue generating the image.", 
                "plot_config_json": plot_config_obj.model_dump_json(), 
                "thinking_log": thinking_log, 
                "action_type": "visualize"
            }
    except Exception as e_plotting:
        thinking_log.append(f"VIZ_PLOT_CRITICAL_ERROR: Exception during plot generation (utils.generate_plot_from_config): {str(e_plotting)}")
        traceback.print_exc() # Print full traceback for server logs
        return {
            "error_message": f"A critical error occurred while creating the plot image: {str(e_plotting)}", 
            "llm_response": "Sorry, an unexpected error stopped me from creating the plot image.", 
            "plot_config_json": plot_config_obj.model_dump_json(), 
            "thinking_log": thinking_log, 
            "action_type": "visualize"
        }

def qna_node(state: AgentState) -> AgentState:
    log = state.get("thinking_log", [])
    log.append("--- QnA Node: Initiated ---")
    if llm is None: # Simplified QnA
        log.append("QnA_ERROR: LLM not available.")
        return {"error_message": "LLM unavailable.", "llm_response": "LLM error.", "thinking_log": log, "action_type": "query_data"}
    # ... (Your full QnA prompt and chain)
    try:
        # Dummy response for example
        prompt_msgs_qna = [("system", f"Answer based on cols: {state['df_columns']}, head: {state['df_head_str']}"), ("human", state['user_query'])]
        chain_qna = ChatPromptTemplate.from_messages(prompt_msgs_qna) | llm
        res = chain_qna.invoke({})
        log.append(f"QnA_SUCCESS: Response: {res.content[:50]}...")
        return {"llm_response": res.content, "thinking_log": log, "action_type": "query_data"}
    except Exception as e:
        log.append(f"QnA_ERROR: {str(e)}")
        return {"error_message": str(e), "llm_response": "Error in QnA.", "thinking_log": log, "action_type": "query_data"}

def fallback_node(state: AgentState) -> AgentState:
    log = state.get("thinking_log", [])
    log.append("--- Fallback Node: Initiated ---")
    err_msg = state.get("error_message")
    user_msg = "I'm sorry, I can't handle that request."
    if err_msg: user_msg += f" Details: {err_msg}"; log.append(f"Fallback_REASON: Error: {err_msg}")
    else: log.append("Fallback_REASON: Query off-topic or router default.")
    return_state = {"llm_response": user_msg, "action_type": "fallback", "thinking_log": log}
    if err_msg: return_state["error_message"] = err_msg
    return return_state

def decide_next_node(state: AgentState) -> str:
    log = state.get("thinking_log", []) # Get current log
    log.append(f"--- Deciding Next Node: Action='{state.get('action_type')}', Error='{state.get('error_message')}' ---")
    # state["thinking_log"] = log # Re-assign if modified, but usually not needed for just appending view
    action = state.get("action_type")
    if action == "visualize": return "visualization_agent"
    if action == "query_data": return "qna_agent"
    return "fallback_agent" # Default to fallback

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
    app_graph = None

def run_agent(user_query: str, df: pd.DataFrame) -> dict:
    print(f"\n--- Running Agent for Query: '{user_query}' ---")
    if app_graph is None:
        return {"response_type": "error", "content": "Agent not ready.", "thinking_log_str": "Graph not compiled.", "error": "Graph compilation failed."}
    if df is None or df.empty:
        return {"response_type": "error", "content": "No data.", "thinking_log_str": "DataFrame empty.", "error": "DataFrame empty."}
    
    initial_state = AgentState(
        user_query=user_query, df_head_str=df.head().to_string(), df_columns=df.columns.tolist(), df_full=df,
        action_type=None, plot_config_json=None, plot_image_bytes=None, plot_insights=None,
        llm_response=None, error_message=None, thinking_log=[]
    )
    final_state: AgentState = {} # Ensure type hinting for final_state
    try:
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 15})
    except Exception as e_invoke:
        log_so_far = final_state.get("thinking_log", initial_state["thinking_log"])
        log_so_far.append(f"AGENT_CRITICAL_ERROR: Graph invoke failed: {str(e_invoke)}")
        final_state = {"thinking_log": log_so_far, "error_message": str(e_invoke), "action_type": "error", "llm_response": f"Graph error: {str(e_invoke)}"}

    img_b64 = None
    if final_state.get("plot_image_bytes"):
        import base64
        img_b64 = base64.b64encode(final_state["plot_image_bytes"]).decode('utf-8')

    resp_type = final_state.get("action_type", "error")
    user_content = final_state.get("llm_response", "No response.")
    err_msg = final_state.get("error_message")

    if err_msg and resp_type not in ["fallback", "error"]: resp_type = "error"
    if not user_content and err_msg : user_content = err_msg
    elif not user_content: user_content = "Processed." if resp_type != "error" else "Error."

    log_list = final_state.get("thinking_log", ["Log missing."])
    if not log_list: log_list = ["Log empty."]

    response = {
        "response_type": resp_type, "content": user_content, "plot_image_bytes": img_b64,
        "plot_config_json": final_state.get("plot_config_json"),
        "plot_insights": final_state.get("plot_insights"),
        "thinking_log_str": "\n".join(log_list), "error": err_msg
    }
    return response

if __name__ == '__main__':
    print("Running agent_workflow.py standalone test...")
    if llm is None: print("LLM failed to initialize. Standalone test aborted.")
    else:
        data = {'ID': [1,2,3,4,5], 'DOB': ['01/01/1990','05/15/1985','11/20/2000','03/10/1995','07/07/1980'], 'Value': [10,20,15,25,12]}
        dummy_df = pd.DataFrame(data)
        dummy_df.rename(columns={'DOB': 'Date of Birth'}, inplace=True) # Match expected DOB col name
        print("\nDummy DataFrame for testing:\n", dummy_df.head())
        queries = ["show age distribution", "bar chart of Value by ID", "hello there"]
        for q in queries:
            print(f"\n--- TEST: '{q}' ---")
            res = run_agent(q, dummy_df.copy())
            print(f"  Type: {res.get('response_type')}, Content: {res.get('content')}")
            if res.get('plot_config_json'): print(f"  Config: {res.get('plot_config_json')}")
            if res.get('plot_insights'): print(f"  Insights: {res.get('plot_insights')}")
            if res.get('error'): print(f"  Error: {res.get('error')}")
            # print(f"  Log:\n{res.get('thinking_log_str')}") # Uncomment for full log
            print("-" * 10)