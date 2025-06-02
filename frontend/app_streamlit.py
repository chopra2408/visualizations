import streamlit as st
import requests
import base64
from datetime import datetime
import traceback
import json 
import os

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [] # To store chat history objects
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "df_columns" not in st.session_state:
    st.session_state.df_columns = []
if "df_head" not in st.session_state:
    st.session_state.df_head = ""

st.set_page_config(layout="wide", page_title="Data Analysis Agent")
st.title("📊 Conversational Data Analysis Agent")

# --- File Uploader ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if st.session_state.current_session_id is None or st.button("Reload File and Start New Session"):
            with st.spinner("Uploading and processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    upload_response = requests.post("http://localhost:8000/uploadfile/", files=files)
                    upload_response.raise_for_status()
                    file_info = upload_response.json()
                    st.session_state.current_session_id = file_info["session_id"]
                    st.session_state.df_columns = file_info["columns"]
                    st.session_state.df_head = file_info["df_head"]
                    st.session_state.messages = [] # Clear chat on new file
                    st.success(f"File '{file_info['filename']}' uploaded! Session ID: {st.session_state.current_session_id}")
                    st.subheader("Data Preview:")
                    st.text_area("First 5 rows:", value=st.session_state.df_head, height=150, disabled=True)
                    st.write("Columns:", st.session_state.df_columns)
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    
    if st.session_state.current_session_id:
        st.sidebar.success(f"Active Session: {st.session_state.current_session_id[:8]}...")
        st.sidebar.subheader("Data Context")
        st.sidebar.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True)


# --- Chat Interface ---
st.header("2. Chat with Your Data")

for msg_obj in st.session_state.messages:
    with st.chat_message(msg_obj["role"]):
        # Display pre-summary if it exists
        if msg_obj.get("pre_summary_content"):
            st.markdown(">" + msg_obj["pre_summary_content"]) # Display as a quote or styled
            st.markdown("---") # Separator

        # Display main content
        st.markdown(msg_obj.get("content", ""))

        if msg_obj["role"] == "assistant":
            # Plot image from the 'data' part of final_agent_response
            if msg_obj.get("plot_image_bytes"):
                img_bytes = base64.b64decode(msg_obj["plot_image_bytes"])
                st.image(img_bytes, caption="Generated Plot", use_container_width =True)
            
            if msg_obj.get("plot_insights"):
                with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                    st.markdown(msg_obj["plot_insights"])
            
            thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
            show_thinking_details = False
            if msg_obj.get("plot_config_json") or msg_obj.get("thinking_log_str"):
                show_thinking_details = True

            if show_thinking_details:
                with st.expander(thinking_expander_title, expanded=False):
                    if msg_obj.get("plot_config_json"):
                        st.write("**Plot Configuration Used:**")
                        try:
                            config_dict = json.loads(msg_obj["plot_config_json"])
                            st.json(config_dict, expanded=False)
                        except:
                            st.text(msg_obj["plot_config_json"])
                    
                    if msg_obj.get("thinking_log_str"):
                        st.write("**Agent's Process Log:**")
                        st.text_area("Log:", value=msg_obj["thinking_log_str"], height=250, disabled=True, key=f"log_{msg_obj.get('timestamp', len(st.session_state.messages))}") # Ensure unique key
                    
                    if msg_obj.get("response_type"):
                         st.caption(f"Agent Action Type: {msg_obj['response_type']}")
                    if msg_obj.get("error") and msg_obj.get("response_type") == "error":
                        st.error(f"Agent Error: {msg_obj['error']}")


if prompt := st.chat_input("Ask about your data or request a plot... (e.g., 'show age distribution')"):
    if not st.session_state.current_session_id:
        st.warning("Please upload a data file first using the sidebar.")
    else:
        current_message_timestamp = datetime.now().isoformat() # For unique keys
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_message_timestamp})

        payload = {"session_id": st.session_state.current_session_id, "query": prompt}
        backend_url = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000/run_agent/")

        with st.chat_message("assistant"):
            pre_summary_content_full = ""
            qna_content_full = ""
            
            # Placeholders for different parts of the response
            pre_summary_placeholder = st.empty()
            qna_placeholder = st.empty() # For Q&A content
            final_response_placeholder = st.container() # For plot, insights etc. from final_agent_response

            assistant_response_data_for_history = {"role": "assistant", "timestamp": current_message_timestamp}

            try:
                with requests.post(backend_url, json=payload, stream=True, timeout=180) as r: # stream=True, increased timeout
                    r.raise_for_status()
                    
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
                                    # This is the full agent response for visualize/fallback
                                    response_data = chunk_data.get("data", {})
                                    assistant_response_data_for_history.update(response_data) # Merge into history object

                                    # Clear placeholders used for streaming text if they were used
                                    pre_summary_placeholder.empty()
                                    qna_placeholder.empty()
                                    
                                    # Now display the final agent response components
                                    with final_response_placeholder:
                                        main_content = response_data.get("content", "No textual response provided.")
                                        st.markdown(main_content)
                                        assistant_response_data_for_history["content"] = main_content


                                        if response_data.get("plot_image_bytes"):
                                            img_bytes = base64.b64decode(response_data["plot_image_bytes"])
                                            st.image(img_bytes, caption="Generated Plot", use_container_width =True)
                                        
                                        if response_data.get("plot_insights"):
                                            with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                                                st.markdown(response_data["plot_insights"])
                                        
                                        thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
                                        show_thinking_details = False
                                        if response_data.get("plot_config_json") or response_data.get("thinking_log_str"):
                                            show_thinking_details = True
                                        
                                        if show_thinking_details:
                                            with st.expander(thinking_expander_title, expanded=False):
                                                if response_data.get("plot_config_json"):
                                                    st.write("**Plot Configuration Used:**")
                                                    try:
                                                        config_dict = json.loads(response_data["plot_config_json"])
                                                        st.json(config_dict, expanded=False)
                                                    except: st.text(response_data["plot_config_json"])
                                                
                                                if response_data.get("thinking_log_str"):
                                                    st.write("**Agent's Process Log:**")
                                                    st.text_area("Log:", value=response_data["thinking_log_str"], height=250, disabled=True, key=f"log_final_{current_message_timestamp}")
                                                
                                                if response_data.get("response_type"):
                                                    st.caption(f"Agent Action Type: {response_data['response_type']}")
                                                if response_data.get("error") and response_data.get("response_type") == "error":
                                                    st.error(f"Agent Error: {response_data['error']}")


                                elif chunk_type == "system":
                                    print(f"Stream system message: {chunk_data.get('message')}")
                                elif chunk_type == "error":
                                    error_chunk_content = chunk_data.get("chunk") or chunk_data.get("content", "Unknown stream error")
                                    st.error(f"Stream Error: {error_chunk_content}")
                                    assistant_response_data_for_history["content"] = f"Error: {error_chunk_content}"
                                    assistant_response_data_for_history["response_type"] = "error"
                                    break # Stop processing further stream on error

                            except json.JSONDecodeError:
                                print(f"Stream: Failed to decode line: {line}")
                
                # Finalize placeholders after stream
                if pre_summary_content_full:
                    pre_summary_placeholder.markdown(">" + pre_summary_content_full)
                    assistant_response_data_for_history["pre_summary_content"] = pre_summary_content_full
                
                if qna_content_full: # This means it was a Q&A response
                    qna_placeholder.markdown(qna_content_full)
                    assistant_response_data_for_history["content"] = qna_content_full
                    if "response_type" not in assistant_response_data_for_history : # if not set by error or final_agent_response
                         assistant_response_data_for_history["response_type"] = "query_data"

                st.session_state.messages.append(assistant_response_data_for_history)

            except requests.exceptions.HTTPError as http_err:
                # ... (error handling as before, ensure it populates assistant_response_data_for_history) ...
                error_content = f"HTTP error from backend: {http_err}"
                try: 
                    error_detail = http_err.response.json().get("detail", str(http_err))
                    error_content = f"Backend Error: {error_detail}"
                except: pass
                st.error(error_content)
                assistant_response_data_for_history["content"] = error_content
                assistant_response_data_for_history["response_type"] = "error"
                st.session_state.messages.append(assistant_response_data_for_history)

            except requests.exceptions.RequestException as e:
                # ... (error handling as before) ...
                st.error(f"Could not connect to backend: {e}")
                assistant_response_data_for_history["content"] = f"Error connecting to backend: {e}"
                assistant_response_data_for_history["response_type"] = "error"
                st.session_state.messages.append(assistant_response_data_for_history)
            except Exception as e:
                # ... (error handling as before) ...
                st.error(f"An unexpected error occurred: {e}")
                traceback.print_exc()
                assistant_response_data_for_history["content"] = f"Unexpected error in Streamlit: {e}"
                assistant_response_data_for_history["response_type"] = "error"
                st.session_state.messages.append(assistant_response_data_for_history)