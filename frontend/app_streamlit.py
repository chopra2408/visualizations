# --- START OF FILE app_streamlit.py ---
import streamlit as st
import requests
import traceback
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import plotly.io # To parse Plotly JSON

# --- Environment Variable Loading ---
load_dotenv() # Load .env file from the current directory or parent
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000") # Default if not set

if not BACKEND_BASE_URL:
    st.error("FATAL ERROR: BACKEND_BASE_URL environment variable not set. Please set it in your .env file or environment.")
    st.stop() # Halt execution if backend URL is missing

# Construct full endpoint URLs
UPLOAD_URL = f"{BACKEND_BASE_URL.rstrip('/')}/uploadfile/"
PROCESS_QUERY_URL = f"{BACKEND_BASE_URL.rstrip('/')}/process_query/"


# --- Initialize session state variables (idempotent) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "df_columns" not in st.session_state:
    st.session_state.df_columns = []
if "df_head" not in st.session_state:
    st.session_state.df_head = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = ""
if "plot_key_counter" not in st.session_state: # For generating unique keys for plots
    st.session_state.plot_key_counter = 0


# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Conversational Data Agent")
st.title("📊 Conversational Data Analysis Agent")

# --- Sidebar for File Upload and Data Info ---
with st.sidebar:
    st.header("1. Upload Your Data")
    # Unique key for file_uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader_widget"
    )

    if uploaded_file is not None:
        # Unique key for button
        if st.button("Upload and Start New Session", key="upload_button_widget"):
            with st.spinner("Uploading and processing file... This may take a moment."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    # Increased timeout for potentially large files or slow server
                    upload_response = requests.post(UPLOAD_URL, files=files, timeout=120)
                    upload_response.raise_for_status() # Raises HTTPError for bad responses

                    file_info = upload_response.json()
                    # Update session state
                    st.session_state.current_session_id = file_info["session_id"]
                    st.session_state.df_columns = file_info["columns"]
                    st.session_state.df_head = file_info["df_head"]
                    st.session_state.current_filename = file_info["filename"]
                    st.session_state.messages = [] # Clear chat history for new session
                    st.session_state.plot_key_counter = 0 # Reset plot key counter

                    st.success(f"File '{st.session_state.current_filename}' processed! Session ID: ...{st.session_state.current_session_id[-6:]}")
                    st.rerun() # Force rerun to update main UI elements based on new session

                except requests.exceptions.Timeout:
                    st.error(f"Upload failed: The request to the backend timed out. The server might be slow or the file too large.")
                except requests.exceptions.HTTPError as http_err:
                    error_detail = "No specific error detail from server."
                    try: # Try to parse detailed error from backend
                        error_detail = http_err.response.json().get("detail", str(http_err))
                    except json.JSONDecodeError: # If response is not JSON
                        error_detail = http_err.response.text if http_err.response.text else str(http_err)
                    st.error(f"Upload failed (HTTP {http_err.response.status_code if http_err.response else 'Unknown'}): {error_detail}")
                except requests.exceptions.RequestException as req_err:
                    st.error(f"Upload failed: Could not connect to the backend at {UPLOAD_URL}. Error: {req_err}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during file upload: {e}")
                    traceback.print_exc() # Log full traceback to console for debugging

    if st.session_state.current_session_id:
        st.sidebar.markdown("---")
        st.sidebar.success(f"Active Session: `...{st.session_state.current_session_id[-12:]}`")
        st.sidebar.info(f"Current File: **{st.session_state.current_filename}**")
        with st.sidebar.expander("Data Preview (Head & Columns)", expanded=False):
            # Unique keys for text_area widgets
            st.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True, key="sidebar_df_cols_preview")
            st.text_area("First 5 rows (Data Head):", value=st.session_state.df_head, height=150, disabled=True, key="sidebar_df_head_preview")

# --- Main Chat Interface ---
st.header("2. Chat with Your Data")

# Display chat messages from history
# Each message object in st.session_state.messages should have a unique 'timestamp'
for i, msg_obj in enumerate(st.session_state.messages):
    with st.chat_message(msg_obj["role"]):
        # Display pre-summary if it exists
        if msg_obj.get("pre_summary_content"):
            st.markdown(">" + msg_obj["pre_summary_content"].strip()) # Using strip for cleaner display
            st.markdown("---") # Separator

        # Display main content
        st.markdown(msg_obj.get("content", "").strip()) # Main textual content

        # Assistant-specific elements (plots, insights, logs)
        if msg_obj["role"] == "assistant":
            # Display Plotly chart if JSON is present
            if msg_obj.get("plotly_fig_json"):
                try:
                    plotly_fig = plotly.io.from_json(msg_obj["plotly_fig_json"])
                    # Unique key for plotly_chart in history
                    plot_key_hist = f"plot_hist_{i}_{msg_obj.get('timestamp', 'default_ts')}"
                    st.plotly_chart(plotly_fig, use_container_width=True, key=plot_key_hist)
                except Exception as e_plot_render:
                    st.error(f"Error rendering interactive plot from history: {e_plot_render}")
                    # print(f"DEBUG: Failed to render plot from history. Key: {plot_key_hist}. Error: {e_plot_render}")
                    # st.json(msg_obj.get("plotly_fig_json")) # Uncomment to show raw JSON for debugging

            # Expander for "Plot Insights"
            if msg_obj.get("plot_insights"):
                # Expander keys are auto-generated based on label, usually fine unless labels are dynamic and identical
                with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                    st.markdown(msg_obj["plot_insights"])

            # Expander for "Thinking Process"
            thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
            # Check if there's anything to show in the expander
            show_thinking_details = bool(msg_obj.get("plot_config_json") or msg_obj.get("thinking_log_str") or (msg_obj.get("error") and msg_obj.get("response_type") == "error"))

            if show_thinking_details:
                with st.expander(thinking_expander_title, expanded=False):
                    if msg_obj.get("plot_config_json"):
                        st.write("**Plot Configuration Used:**")
                        try:
                            config_dict = json.loads(msg_obj["plot_config_json"])
                            st.json(config_dict, expanded=False) # st.json handles its own keying
                        except json.JSONDecodeError:
                            st.text(msg_obj["plot_config_json"]) # Display as text if not valid JSON

                    if msg_obj.get("thinking_log_str"):
                        st.write("**Agent's Process Log:**")
                        # Unique key for text_area in history
                        log_key_hist = f"log_hist_{i}_{msg_obj.get('timestamp', 'default_ts')}"
                        st.text_area("Log Details:", value=msg_obj["thinking_log_str"], height=200, disabled=True, key=log_key_hist)

                    if msg_obj.get("response_type"):
                         st.caption(f"Agent Action Type: `{msg_obj['response_type']}`")
                    if msg_obj.get("error") and msg_obj.get("response_type") == "error": # Show specific agent error
                        st.error(f"Agent Error Detail: {msg_obj['error']}")


# Chat input widget
# Unique key for chat_input
if prompt := st.chat_input("Ask about your data or request a plot...", key="main_chat_input_widget"):
    if not st.session_state.current_session_id:
        st.warning("⚠️ Please upload a data file first using the sidebar.")
        st.stop() # Prevent further execution if no session

    current_message_timestamp = datetime.now().isoformat() # For unique message object and keys
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_message_timestamp})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare for assistant's response (streaming)
    with st.chat_message("assistant"):
        pre_summary_content_full = ""
        qna_content_full = ""

        # Placeholders for streaming text
        pre_summary_placeholder = st.empty()
        qna_placeholder = st.empty()
        # Container for final non-text elements like plots from the stream
        final_response_container = st.container()

        # This dict will store all parts of the assistant's response for appending to history
        assistant_response_for_history = {"role": "assistant", "timestamp": current_message_timestamp}

        payload = {"session_id": st.session_state.current_session_id, "query": prompt}

        try:
            # Stream response from backend
            # Increased timeout for potentially long agent runs
            with requests.post(PROCESS_QUERY_URL, json=payload, stream=True, timeout=360) as r:
                r.raise_for_status() # Check for HTTP errors before starting to iterate

                for line in r.iter_lines():
                    if line: # filter out keep-alive new lines
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            chunk_type = chunk_data.get("type")

                            if chunk_type == "thinking_process_update": # Handles pre-summary
                                pre_summary_content_full += chunk_data.get("chunk", "")
                                pre_summary_placeholder.markdown(">" + pre_summary_content_full.strip() + "▌")
                            elif chunk_type == "content": # Handles Q&A content
                                qna_content_full += chunk_data.get("chunk", "")
                                qna_placeholder.markdown(qna_content_full.strip() + "▌")
                            elif chunk_type == "final_agent_response":
                                response_data = chunk_data.get("data", {})
                                # Merge full agent response into history object (will overwrite streamed text if also present in final)
                                assistant_response_for_history.update(response_data)

                                # Clear streaming text placeholders as final response is comprehensive
                                pre_summary_placeholder.empty()
                                qna_placeholder.empty()

                                # Display elements from the final_agent_response in their own container
                                with final_response_container:
                                    main_content_final = response_data.get("content", "Processing complete.")
                                    st.markdown(main_content_final.strip())
                                    # Ensure main content is in history if it came from final response
                                    if "content" not in assistant_response_for_history or not assistant_response_for_history["content"]:
                                        assistant_response_for_history["content"] = main_content_final

                                    if response_data.get("plotly_fig_json"):
                                        try:
                                            plotly_fig_final = plotly.io.from_json(response_data["plotly_fig_json"])
                                            # Unique key for plotly_chart from stream
                                            st.session_state.plot_key_counter += 1
                                            plot_key_stream = f"plot_stream_{st.session_state.plot_key_counter}_{current_message_timestamp}"
                                            st.plotly_chart(plotly_fig_final, use_container_width=True, key=plot_key_stream)
                                        except Exception as e_plot_final:
                                            st.error(f"Error rendering interactive plot from final response: {e_plot_final}")

                                    if response_data.get("plot_insights"):
                                        with st.expander("🔍 View Plot Insights/Summary", expanded=True): # Expand by default for new plot
                                            st.markdown(response_data["plot_insights"])

                                    thinking_exp_title_final = "⚙️ View Agent's Thinking & Configuration (Final)"
                                    show_thinking_final = bool(response_data.get("plot_config_json") or response_data.get("thinking_log_str") or (response_data.get("error") and response_data.get("response_type") == "error"))
                                    if show_thinking_final:
                                        with st.expander(thinking_exp_title_final, expanded=False):
                                            if response_data.get("plot_config_json"):
                                                st.write("**Plot Configuration Used:**")
                                                try: st.json(json.loads(response_data["plot_config_json"]), expanded=False)
                                                except: st.text(response_data["plot_config_json"])
                                            if response_data.get("thinking_log_str"):
                                                st.write("**Agent's Process Log:**")
                                                log_key_final_stream = f"log_final_stream_{current_message_timestamp}"
                                                st.text_area("Log Details:", value=response_data["thinking_log_str"], height=200, disabled=True, key=log_key_final_stream)
                                            if response_data.get("response_type"): st.caption(f"Agent Action Type: `{response_data['response_type']}`")
                                            if response_data.get("error") and response_data.get("response_type") == "error": st.error(f"Agent Error Detail: {response_data['error']}")
                            elif chunk_type == "system": # E.g., "Q&A stream ended." (for backend debugging)
                                print(f"System Message from Stream: {chunk_data.get('message')}")
                            elif chunk_type == "error": # Error sent as a specific chunk type by backend
                                error_msg_chunk = chunk_data.get("chunk") or chunk_data.get("content", "Unknown error from stream.")
                                st.error(f"Backend Stream Error: {error_msg_chunk}")
                                assistant_response_for_history["content"] = f"Error from backend: {error_msg_chunk}"
                                assistant_response_for_history["response_type"] = "error"
                                break # Stop processing stream on explicit error chunk
                        except json.JSONDecodeError:
                            print(f"Stream: Failed to decode JSON line: {line.decode('utf-8', errors='ignore')}")
                        except Exception as e_chunk_process:
                            print(f"Stream: Error processing chunk: {e_chunk_process}")
                            traceback.print_exc() # Log full traceback for server-side debugging
                            st.warning(f"A minor error occurred while displaying part of the response: {e_chunk_process}")

                # Finalize streamed text placeholders after loop (if not cleared by final_agent_response)
                if pre_summary_content_full and pre_summary_placeholder.markdown: # Check if placeholder still exists
                    pre_summary_placeholder.markdown(">" + pre_summary_content_full.strip()) # Remove cursor
                    if "pre_summary_content" not in assistant_response_for_history: # Store if not already part of final_agent_response
                        assistant_response_for_history["pre_summary_content"] = pre_summary_content_full

                if qna_content_full and qna_placeholder.markdown: # Check if placeholder still exists
                    qna_placeholder.markdown(qna_content_full.strip()) # Remove cursor
                    if "content" not in assistant_response_for_history or not assistant_response_for_history["content"]: # Store if not part of final
                        assistant_response_for_history["content"] = qna_content_full
                        if "response_type" not in assistant_response_for_history: # If it was pure Q&A
                            assistant_response_for_history["response_type"] = "query_data"

                # Ensure there's some content if nothing else was set (e.g. if stream ended abruptly before final_agent_response)
                if "content" not in assistant_response_for_history or not assistant_response_for_history.get("content", "").strip():
                     assistant_response_for_history["content"] = "Request processed. (No specific textual output received)"
                     if "response_type" not in assistant_response_for_history:
                         assistant_response_for_history["response_type"] = "unknown" # If type wasn't set by stream

                st.session_state.messages.append(assistant_response_for_history)
                # No st.rerun() here, new messages are added to state, chat_message handles display

        except requests.exceptions.Timeout:
            err_msg = f"Request timed out connecting to the backend at {PROCESS_QUERY_URL}. The agent might be taking too long or the server is unresponsive."
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg; assistant_response_for_history["response_type"] = "error"; st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.HTTPError as http_err_main:
            error_detail_main = "No specific error detail from server."
            try: error_detail_main = http_err_main.response.json().get("detail",str(http_err_main)) if http_err_main.response else str(http_err_main)
            except: error_detail_main=str(http_err_main)
            err_msg = f"Query failed (HTTP {http_err_main.response.status_code if http_err_main.response else 'Unknown'}): {error_detail_main}"
            st.error(err_msg); assistant_response_for_history["content"] = err_msg; assistant_response_for_history["response_type"] = "error"; st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.RequestException as req_err_main:
            err_msg = f"Query failed: Could not connect to the backend at {PROCESS_QUERY_URL}. Error: {req_err_main}"
            st.error(err_msg); assistant_response_for_history["content"] = err_msg; assistant_response_for_history["response_type"] = "error"; st.session_state.messages.append(assistant_response_for_history)
        except Exception as e_main_query:
            err_msg = f"An unexpected error occurred in the Streamlit app while processing your query: {e_main_query}"
            st.error(err_msg); traceback.print_exc(); assistant_response_for_history["content"] = err_msg; assistant_response_for_history["response_type"] = "error"; st.session_state.messages.append(assistant_response_for_history)

# --- END OF FILE app_streamlit.py ---