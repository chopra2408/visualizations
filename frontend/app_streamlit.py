import streamlit as st
import requests
import base64
import json 
import os

FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="📊 Statistical Analyzer AI")

st.title("📊 Statistical Analyzer AI")
st.markdown("""
Upload a CSV or XLSX file, then ask questions or request visualizations about your data!
The AI will try to understand your request and generate responses or plots.
""")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "file_info" not in st.session_state:
    st.session_state.file_info = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "processed_file_identifier" not in st.session_state:
    st.session_state.processed_file_identifier = None

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        current_file_identifier = (uploaded_file.name, uploaded_file.size, uploaded_file.type)

        if st.session_state.processed_file_identifier != current_file_identifier or not st.session_state.session_id:
            st.session_state.messages = [] # Reset chat on new file processing
            with st.spinner("Processing file..."):
                files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{FASTAPI_URL}/uploadfile/", files=files_payload)
                    response.raise_for_status()

                    file_upload_response = response.json()
                    backend_session_id = file_upload_response.get("session_id")

                    if backend_session_id:
                        st.session_state.session_id = backend_session_id
                        st.session_state.file_info = file_upload_response
                        st.session_state.processed_file_identifier = current_file_identifier
                        st.success(f"File '{st.session_state.file_info['filename']}' processed successfully!")
                        st.session_state.df_preview = st.session_state.file_info.get("df_head", "Preview not available.")
                    else:
                        st.error("Backend did not return a session ID. File processing may have failed or response format is incorrect.")
                        st.session_state.session_id = None
                        st.session_state.file_info = None
                        st.session_state.processed_file_identifier = None

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")
                    st.session_state.session_id = None
                    st.session_state.file_info = None
                    st.session_state.processed_file_identifier = None
                except Exception as e:
                    st.error(f"Error processing file upload: {e}")
                    st.session_state.session_id = None
                    st.session_state.file_info = None
                    st.session_state.processed_file_identifier = None

    if st.session_state.file_info and st.session_state.session_id:
        st.subheader("File Information:")
        st.write(f"**Name:** {st.session_state.file_info.get('filename', 'N/A')}")
        st.write(f"**Columns:** {', '.join(st.session_state.file_info.get('columns', []))}")
        if st.session_state.df_preview:
            st.subheader("Data Preview (First 5 Rows):")
            st.text_area("DataFrame Head", value=st.session_state.df_preview, height=200, disabled=True)

st.header("2. Chat with Your Data")

if not st.session_state.session_id:
    st.warning("Please upload a data file first.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_streamed_response"): # If it was a streamed response
                st.markdown(message["content"]) # Display the accumulated content
            else: # For non-streamed (e.g., visualization)
                st.markdown(message.get("content", ""))
                if message.get("plot_config_str"):
                    with st.expander("View Plot Configuration Used"):
                        st.code(message["plot_config_str"], language="json")
                if "image_b64" in message and message["image_b64"] is not None: # Expecting b64 string
                    try:
                        img_bytes = base64.b64decode(message["image_b64"])
                        st.image(img_bytes, caption="Generated Plot", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image from history: {e}")


    if prompt := st.chat_input("Ask about your data or request a visualization..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty() # For streaming display
            full_response_content = ""
            assistant_message_for_history = {"role": "assistant", "content": ""}

            try:
                payload = {"session_id": st.session_state.session_id, "query": prompt}
                # Use stream=True for requests
                with requests.post(f"{FASTAPI_URL}/process_query/", json=payload, stream=True) as r:
                    r.raise_for_status() # Check for HTTP errors early

                    # Determine if response is streaming (ndjson) or a single JSON object
                    content_type = r.headers.get("content-type", "")

                    if "application/x-ndjson" in content_type:
                        assistant_message_for_history["is_streamed_response"] = True
                        for line in r.iter_lines(): # Process ndjson line by line
                            if line:
                                try:
                                    chunk_data = json.loads(line.decode('utf-8'))
                                    if chunk_data.get("type") == "content":
                                        full_response_content += chunk_data.get("chunk", "")
                                        message_placeholder.markdown(full_response_content + "▌")
                                    elif chunk_data.get("type") == "system":
                                        print(f"System message from stream: {chunk_data.get('message')}") # Log or display if needed
                                    elif chunk_data.get("type") == "error": # Handle error from stream
                                        error_detail = chunk_data.get("data", {}).get("error", "Unknown stream error")
                                        full_response_content += f"\n⚠️ Error during stream: {error_detail}"
                                        message_placeholder.error(full_response_content)
                                        break # Stop processing stream on error
                                except json.JSONDecodeError:
                                    print(f"Warning: Could not decode JSON line from stream: {line}")
                        message_placeholder.markdown(full_response_content) # Final update without cursor
                        assistant_message_for_history["content"] = full_response_content

                    else: # Assume it's a single JSON response (e.g., for visualization)
                        assistant_message_for_history["is_streamed_response"] = False
                        agent_res_json = r.json() # Parse the whole JSON

                        response_type = agent_res_json.get("response_type")
                        content = agent_res_json.get("content", "")
                        plot_bytes_b64 = agent_res_json.get("plot_image_bytes") # Expecting base64 string
                        plot_config_json_str = agent_res_json.get("plot_config_json")
                        error_msg = agent_res_json.get("error")

                        text_parts_for_display = []
                        if error_msg: text_parts_for_display.append(f"⚠️ Error: {error_msg}")
                        if content: text_parts_for_display.append(content)
                        
                        full_response_content = "\n\n".join(filter(None, text_parts_for_display))
                        if not full_response_content and not plot_bytes_b64 and not plot_config_json_str:
                            full_response_content = "Received an empty non-streaming response."
                        
                        message_placeholder.markdown(full_response_content)
                        assistant_message_for_history["content"] = full_response_content

                        if plot_config_json_str and response_type == "visualize":
                            try:
                                pretty_config = json.dumps(json.loads(plot_config_json_str), indent=2)
                                with st.expander("View Plot Configuration Used"):
                                    st.code(pretty_config, language="json")
                                assistant_message_for_history["plot_config_str"] = pretty_config
                            except json.JSONDecodeError:
                                st.warning("Could not parse plot config for display.")
                                assistant_message_for_history["plot_config_str"] = plot_config_json_str


                        if plot_bytes_b64:
                            try:
                                img_bytes = base64.b64decode(plot_bytes_b64)
                                if img_bytes:
                                    st.image(img_bytes, caption="Generated Plot", use_container_width=True)
                                    assistant_message_for_history["image_b64"] = plot_bytes_b64 # Store b64 string
                                else: st.warning("Received empty plot image data.")
                            except Exception as e: st.error(f"Error displaying plot: {e}")
                
                st.session_state.messages.append(assistant_message_for_history)

            except requests.exceptions.RequestException as e:
                error_text = f"Error communicating with AI backend: {e}"
                message_placeholder.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "is_streamed_response": False})
            except Exception as e:
                error_text = f"An unexpected error occurred in frontend: {e}"
                import traceback
                st.error(f"{error_text}\n```\n{traceback.format_exc()}\n```")
                st.session_state.messages.append({"role": "assistant", "content": error_text, "is_streamed_response": False})