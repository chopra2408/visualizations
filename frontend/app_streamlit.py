import streamlit as st
import requests
import base64
import traceback
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# --- Environment Variable Loading ---
# Load environment variables from .env file
load_dotenv()
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

if not BACKEND_BASE_URL:
    st.error("FATAL ERROR: BACKEND_BASE_URL environment variable not set. Please set it in your .env file or environment.")
    st.stop() # Halt execution if backend URL is missing

# Construct full endpoint URLs
UPLOAD_URL = f"{BACKEND_BASE_URL.rstrip('/')}/uploadfile/"
PROCESS_QUERY_URL = f"{BACKEND_BASE_URL.rstrip('/')}/process_query/"


# --- Initialize session state variables ---
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


st.set_page_config(layout="wide", page_title="Cloud Data Agent")
st.title("📊 Conversational Data Analysis Agent (Cloud Backend)")

if not BACKEND_BASE_URL: # Should have been caught above, but good for robustness
    st.error("Backend URL is not configured. Application cannot start.")
    st.stop()

# --- File Uploader ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader")

    if uploaded_file is not None:
        # Add a button to explicitly trigger upload/new session
        if st.button("Upload and Process File", key="upload_button"):
            with st.spinner("Uploading and processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    # --- Important: Add Timeout ---
                    upload_response = requests.post(UPLOAD_URL, files=files, timeout=60) # 60 seconds timeout
                    upload_response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
                    
                    file_info = upload_response.json()
                    st.session_state.current_session_id = file_info["session_id"]
                    st.session_state.df_columns = file_info["columns"]
                    st.session_state.df_head = file_info["df_head"]
                    st.session_state.current_filename = file_info["filename"]
                    st.session_state.messages = [] # Clear chat on new file upload
                    
                    st.success(f"File '{st.session_state.current_filename}' uploaded! New session started.")
                    # Force a re-run to update the main UI with new session info
                    st.rerun()

                except requests.exceptions.Timeout:
                    st.error(f"Upload failed: The request to {UPLOAD_URL} timed out. The server might be slow or the file too large.")
                except requests.exceptions.HTTPError as http_err:
                    error_detail = "No specific error detail from server."
                    try:
                        error_detail = http_err.response.json().get("detail", str(http_err))
                    except json.JSONDecodeError:
                        error_detail = http_err.response.text # Show raw text if not JSON
                    st.error(f"Upload failed (HTTP {http_err.response.status_code}): {error_detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: Could not connect to backend at {UPLOAD_URL}. Error: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during file upload: {e}")
                    traceback.print_exc()
    
    if st.session_state.current_session_id:
        st.sidebar.success(f"Active Session: ...{st.session_state.current_session_id[-12:]}")
        st.sidebar.info(f"File: {st.session_state.current_filename}")
        with st.sidebar.expander("Data Preview (Head & Columns)"):
            st.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True, key="sidebar_cols_display")
            st.text_area("First 5 rows:", value=st.session_state.df_head, height=150, disabled=True, key="sidebar_head_display")


# --- Chat Interface ---
st.header("2. Chat with Your Data")

# Display chat messages from history
for i, msg_obj in enumerate(st.session_state.messages):
    with st.chat_message(msg_obj["role"]):
        # Display pre-summary if it exists (from new streaming logic)
        if msg_obj.get("pre_summary_content"):
            st.markdown(">" + msg_obj["pre_summary_content"])
            st.markdown("---")

        # Display main content
        st.markdown(msg_obj.get("content", "")) # Main textual content

        if msg_obj["role"] == "assistant":
            if msg_obj.get("plot_image_bytes"):
                try:
                    img_bytes = base64.b64decode(msg_obj["plot_image_bytes"])
                    st.image(img_bytes, caption="Generated Plot", use_container_width=True)
                except Exception as e:
                    st.error(f"Error decoding plot image: {e}")
            
            # Expander for "Plot Insights"
            if msg_obj.get("plot_insights"):
                with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                    st.markdown(msg_obj["plot_insights"])
            
            # Expander for "Thinking Process"
            thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
            show_thinking_details = bool(msg_obj.get("plot_config_json") or msg_obj.get("thinking_log_str"))

            if show_thinking_details:
                with st.expander(thinking_expander_title, expanded=False):
                    if msg_obj.get("plot_config_json"):
                        st.write("**Plot Configuration Used:**")
                        try:
                            config_dict = json.loads(msg_obj["plot_config_json"])
                            st.json(config_dict, expanded=False)
                        except: # If it's not valid JSON, display as text
                            st.text(msg_obj["plot_config_json"])
                    
                    if msg_obj.get("thinking_log_str"):
                        st.write("**Agent's Process Log:**")
                        # Using timestamp or index for unique key
                        log_key = f"log_{msg_obj.get('timestamp', i)}"
                        st.text_area("Log:", value=msg_obj["thinking_log_str"], height=200, disabled=True, key=log_key)
                    
                    if msg_obj.get("response_type"):
                         st.caption(f"Agent Action Type: {msg_obj['response_type']}")
                    if msg_obj.get("error") and msg_obj.get("response_type") == "error":
                        st.error(f"Agent Error Detail: {msg_obj['error']}")


# Chat input
if prompt := st.chat_input("Ask about your data or request a plot...", key="chat_prompt"):
    if not st.session_state.current_session_id:
        st.warning("Please upload a data file first using the sidebar.")
        st.stop() # Prevent further execution if no session

    current_message_timestamp = datetime.now().isoformat() # For unique keys if needed
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_message_timestamp})
    
    with st.chat_message("user"): # Display user message immediately
        st.markdown(prompt)

    # Prepare for assistant's response
    with st.chat_message("assistant"):
        pre_summary_content_full = ""
        qna_content_full = ""
        
        pre_summary_placeholder = st.empty()
        qna_placeholder = st.empty()
        final_response_container = st.container() # For plot, insights from final_agent_response

        # This dict will store all parts of the assistant's response for history
        assistant_response_for_history = {"role": "assistant", "timestamp": current_message_timestamp}

        payload = {"session_id": st.session_state.current_session_id, "query": prompt}
        
        try:
            # --- Important: Add Timeout ---
            with requests.post(PROCESS_QUERY_URL, json=payload, stream=True, timeout=300) as r: # 300s (5 min) timeout for potentially long agent runs
                r.raise_for_status() # Check for HTTP errors (4xx or 5xx)
                
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            chunk_type = chunk_data.get("type")

                            if chunk_type == "pre_summary_chunk":
                                pre_summary_content_full += chunk_data.get("chunk", "")
                                pre_summary_placeholder.markdown(">" + pre_summary_content_full + "▌")
                            elif chunk_type == "content": # Q&A content
                                qna_content_full += chunk_data.get("chunk", "")
                                qna_placeholder.markdown(qna_content_full + "▌")
                            elif chunk_type == "final_agent_response":
                                response_data = chunk_data.get("data", {})
                                assistant_response_for_history.update(response_data) # Merge full agent response

                                # Clear streaming text placeholders as final response is here
                                pre_summary_placeholder.empty()
                                qna_placeholder.empty()
                                
                                with final_response_container:
                                    main_content = response_data.get("content", "Processing complete.")
                                    st.markdown(main_content)
                                    # If main content comes from here, ensure it's in history
                                    if "content" not in assistant_response_for_history or not assistant_response_for_history["content"]:
                                        assistant_response_for_history["content"] = main_content

                                    if response_data.get("plot_image_bytes"):
                                        img_bytes = base64.b64decode(response_data["plot_image_bytes"])
                                        st.image(img_bytes, caption="Generated Plot", use_container_width=True)
                                    
                                    if response_data.get("plot_insights"):
                                        with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                                            st.markdown(response_data["plot_insights"])
                                    
                                    thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
                                    show_thinking_details = bool(response_data.get("plot_config_json") or response_data.get("thinking_log_str"))
                                    if show_thinking_details:
                                        with st.expander(thinking_expander_title, expanded=False):
                                            # ... (display plot_config, thinking_log, response_type, error as before)
                                            if response_data.get("plot_config_json"):
                                                st.write("**Plot Configuration Used:**")
                                                try:
                                                    config_dict = json.loads(response_data["plot_config_json"])
                                                    st.json(config_dict, expanded=False)
                                                except: st.text(response_data["plot_config_json"])
                                            
                                            if response_data.get("thinking_log_str"):
                                                st.write("**Agent's Process Log:**")
                                                st.text_area("Log:", value=response_data["thinking_log_str"], height=200, disabled=True, key=f"log_final_{current_message_timestamp}")
                                            
                                            if response_data.get("response_type"):
                                                st.caption(f"Agent Action Type: {response_data['response_type']}")
                                            if response_data.get("error") and response_data.get("response_type") == "error":
                                                st.error(f"Agent Error Detail: {response_data['error']}")
                            
                            elif chunk_type == "system": # E.g., "Q&A stream ended."
                                print(f"System Message from Stream: {chunk_data.get('message')}")
                            elif chunk_type == "error": # Error sent within the stream by backend
                                error_msg = chunk_data.get("chunk") or chunk_data.get("content", "Unknown error from stream.")
                                st.error(f"Backend Stream Error: {error_msg}")
                                assistant_response_for_history["content"] = f"Error: {error_msg}"
                                assistant_response_for_history["response_type"] = "error"
                                break # Stop processing stream on error

                        except json.JSONDecodeError:
                            print(f"Stream: Failed to decode JSON line: {line.decode('utf-8', errors='ignore')}")
                        except Exception as e_chunk:
                            print(f"Stream: Error processing chunk: {e_chunk}")
                            traceback.print_exc()
                            st.warning(f"Error displaying part of the response: {e_chunk}")

            # Finalize streamed text placeholders after loop
            if pre_summary_content_full:
                pre_summary_placeholder.markdown(">" + pre_summary_content_full) # Remove cursor
                assistant_response_for_history["pre_summary_content"] = pre_summary_content_full
            
            if qna_content_full: # This means it was a Q&A-only response
                qna_placeholder.markdown(qna_content_full) # Remove cursor
                assistant_response_for_history["content"] = qna_content_full
                if "response_type" not in assistant_response_for_history : # if not set by error or final_agent_response
                        assistant_response_for_history["response_type"] = "query_data"
            
            # Ensure there's some content if nothing else was set
            if "content" not in assistant_response_for_history or not assistant_response_for_history.get("content"):
                 assistant_response_for_history["content"] = "Processed request." # Default message
            
            st.session_state.messages.append(assistant_response_for_history)
            # No st.rerun() here, new messages will display on next natural rerun (e.g. next chat input)

        except requests.exceptions.Timeout:
            err_msg = f"Request timed out connecting to {PROCESS_QUERY_URL}. The backend might be processing a very long query or is unresponsive."
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.HTTPError as http_err:
            error_detail = "No specific error detail from server."
            try:
                error_detail = http_err.response.json().get("detail", str(http_err))
            except json.JSONDecodeError:
                error_detail = http_err.response.text
            err_msg = f"Query failed (HTTP {http_err.response.status_code}): {error_detail}"
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.RequestException as e:
            err_msg = f"Query failed: Could not connect to backend at {PROCESS_QUERY_URL}. Error: {e}"
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except Exception as e:
            err_msg = f"An unexpected error occurred in Streamlit while processing query: {e}"
            st.error(err_msg)
            traceback.print_exc()
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)