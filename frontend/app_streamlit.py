# frontend/app_streamlit.py

import streamlit as st
import requests
import traceback
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import plotly.io as plotly_io
import streamlit.components.v1 as components

# --- Environment Variable Loading ---
dotenv_path_streamlit = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path_streamlit):
    dotenv_path_streamlit = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path_streamlit)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

if not BACKEND_BASE_URL:
    st.error("FATAL ERROR: BACKEND_BASE_URL environment variable not set.")
    st.stop()

UPLOAD_URL = f"{BACKEND_BASE_URL.rstrip('/')}/uploadfile/"
PROCESS_QUERY_URL = f"{BACKEND_BASE_URL.rstrip('/')}/process_query/"
# SCREENCAST_UPLOAD_URL is not used by the JS component in this "download-only" version
# SCREENCAST_UPLOAD_URL = f"{BACKEND_BASE_URL.rstrip('/')}/upload_screencast/"


# --- Initialize session state variables ---
default_session_vars = {
    "messages": [], "current_session_id": None, "df_columns": [], "df_head": "",
    "current_filename": "", "plot_key_counter": 0,
}
for key, value in default_session_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Conversational Data Agent")
st.title("📊 Conversational Data Analysis Agent")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader_widget")

    if uploaded_file:
        if st.button("Upload and Start New Session", key="upload_button_widget"):
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    upload_response = requests.post(UPLOAD_URL, files=files, timeout=120)
                    upload_response.raise_for_status()
                    file_info = upload_response.json()
                    st.session_state.current_session_id = file_info["session_id"]
                    st.session_state.df_columns = file_info["columns"]
                    st.session_state.df_head = file_info["df_head"]
                    st.session_state.current_filename = file_info["filename"]
                    st.session_state.messages = []
                    st.session_state.plot_key_counter = 0
                    st.success(f"File '{st.session_state.current_filename}' processed! Session: ...{st.session_state.current_session_id[-6:]}")
                    st.rerun()
                except Exception as e:
                    st.error(f"File upload failed: {str(e)[:500]}") # Show first 500 chars of error
                    traceback.print_exc()


    if st.session_state.current_session_id:
        st.sidebar.markdown("---")
        st.sidebar.success(f"Active Session: `...{st.session_state.current_session_id[-12:]}`")
        st.sidebar.info(f"Current File: **{st.session_state.current_filename}**")
        with st.sidebar.expander("Data Preview", expanded=False):
            st.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True, key="sidebar_df_cols_preview")
            st.text_area("Data Head:", value=st.session_state.df_head, height=150, disabled=True, key="sidebar_df_head_preview")

        st.sidebar.markdown("---")
        st.sidebar.header("🎬 Record Screen (Client-Side Download)")

        # IMPORTANT: Replace with your actual published NPM package name
        PACKAGE_NAME = "simple-screen-recorder"  # <<<< YOUR PUBLISHED NPM PACKAGE NAME
        PACKAGE_VERSION = "latest" # Or your specific version

        if PACKAGE_NAME.startswith('@'):
            cdn_url = f"https://cdn.jsdelivr.net/npm/{PACKAGE_NAME}@{PACKAGE_VERSION}/+esm"
        else:
            cdn_url = f"https://cdn.jsdelivr.net/npm/{PACKAGE_NAME}@{PACKAGE_VERSION}/+esm"

        html_component = f"""
        <div id="recorderInterfaceRootSSR" style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <button id="startRecordButtonSSR" style="padding: 8px 12px; margin-right: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Start Recording</button>
            <button id="stopRecordButtonSSR" style="padding: 8px 12px; background-color: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer;" disabled>Stop</button>
            <p id="recorderStatusSSR" style="margin-top: 10px; font-size: 0.9em;">Status: Idle</p>
            <p id="actionStatusSSR" style="font-size: 0.9em;"></p>
            <video id="videoPreviewSSR" controls muted style="max-width: 100%; margin-top:10px; display:none;"></video>
        </div>

        <script type="module">
            const rootElSSR = document.getElementById('recorderInterfaceRootSSR');
            let ScreenRecorder, SimpleScreenRecorderStatus;

            try {{
                const module = await import('{cdn_url}');
                ScreenRecorder = module.ScreenRecorder;
                SimpleScreenRecorderStatus = module.SimpleScreenRecorderStatus;

                if (!ScreenRecorder || !SimpleScreenRecorderStatus) {{
                    throw new Error('ScreenRecorder or SimpleScreenRecorderStatus not found. Check package exports.');
                }}

                const startButton = document.getElementById('startRecordButtonSSR');
                const stopButton = document.getElementById('stopRecordButtonSSR');
                const recorderStatusEl = document.getElementById('recorderStatusSSR');
                const actionStatusEl = document.getElementById('actionStatusSSR');
                const videoPreviewEl = document.getElementById('videoPreviewSSR');

                const sessionId = "{st.session_state.current_session_id}"; // For context, not for upload URL here

                let currentRecorderBlob = null;
                let currentFilename = "screen-recording.webm";

                const recorder = new ScreenRecorder({{
                    onStatusChange: (status, details) => {{
                        recorderStatusEl.textContent = `Status: ${{status}}`;
                        if (details) {{
                            recorderStatusEl.textContent += ` (${{details instanceof Error ? details.message : String(details)}})`;
                        }}
                        startButton.disabled = status === SimpleScreenRecorderStatus.RECORDING || status === SimpleScreenRecorderStatus.REQUESTING_PERMISSION;
                        stopButton.disabled = status !== SimpleScreenRecorderStatus.RECORDING;
                        if (status !== SimpleScreenRecorderStatus.STOPPED) {{
                           actionStatusEl.textContent = "";
                        }}
                        recorderStatusEl.style.color = (status === SimpleScreenRecorderStatus.ERROR || status === SimpleScreenRecorderStatus.PERMISSION_DENIED) ? "red" : "inherit";
                    }},
                    onRecordingComplete: (blob, filename) => {{
                        currentRecorderBlob = blob;
                        currentFilename = filename;
                        const blobUrl = URL.createObjectURL(blob);
                        videoPreviewEl.src = blobUrl;
                        videoPreviewEl.style.display = 'block';

                        recorderStatusEl.textContent = "Status: Stopped. Preparing download...";
                        actionStatusEl.textContent = "";

                        try {{
                            if (ScreenRecorder && currentRecorderBlob && currentFilename) {{
                                ScreenRecorder.downloadBlob(currentRecorderBlob, currentFilename);
                                actionStatusEl.textContent = `Download of '${{currentFilename}}' initiated. Check browser downloads.`;
                                actionStatusEl.style.color = "green";
                                console.log(`Client-side download initiated for ${{currentFilename}}`);
                            }} else {{
                                throw new Error("Recorder utilities or blob not available for download.");
                            }}
                        }} catch (downloadError) {{
                            console.error("Client-side download error:", downloadError);
                            actionStatusEl.textContent = `Error initiating download: ${{downloadError.message}}`;
                            actionStatusEl.style.color = "red";
                        }}
                        recorderStatusEl.textContent = "Status: Idle. Download processed.";
                    }}
                }});

                startButton.onclick = async () => {{
                    if (!sessionId) {{
                        recorderStatusEl.textContent = "Status: Error - Streamlit Session ID missing.";
                        recorderStatusEl.style.color = "red";
                        startButton.disabled = true; // Disable if no session
                        return;
                    }}
                    videoPreviewEl.style.display = 'none';
                    if (videoPreviewEl.src && videoPreviewEl.src.startsWith('blob:')) {{
                        URL.revokeObjectURL(videoPreviewEl.src);
                    }}
                    videoPreviewEl.src = '';
                    currentRecorderBlob = null;
                    actionStatusEl.textContent = '';

                    try {{ await recorder.startRecording(); }}
                    catch (err) {{ console.error("Error from recorder.startRecording() call:", err); }}
                }};

                stopButton.onclick = () => {{ recorder.stopRecording(); }};

                if (!sessionId) {{ // Initial check
                     recorderStatusEl.textContent = "Status: Error - Streamlit session ID invalid or missing.";
                     recorderStatusEl.style.color = "red";
                     startButton.disabled = true;
                }}

            }} catch (err) {{
                console.error("Failed to load or initialize ScreenRecorder from CDN: ", err);
                if (rootElSSR) {{
                    rootElSSR.innerHTML = `<p style='color:red;'><b>Error loading screen recorder:</b><br/>${{err.message}}.<br/>Package: '<b>${PACKAGE_NAME}</b>@<b>${PACKAGE_VERSION}</b>'</p><p>CDN URL: {cdn_url}</p><p>Check browser console (F12).</p>`;
                }}
            }}
        </script>
        """
        components.html(html_component, height=400, scrolling=False)

    else:
        st.sidebar.info("Upload a data file to enable screen recording and other features.")

# --- Main Chat Interface ---
# ... (The rest of your Streamlit app_streamlit.py code for chat, etc. - THIS REMAINS THE SAME) ...
st.header("2. Chat with Your Data")

for i, msg_obj in enumerate(st.session_state.messages):
    with st.chat_message(msg_obj["role"]):
        if msg_obj.get("pre_summary_content"):
            st.markdown(">" + msg_obj["pre_summary_content"].strip())
            st.markdown("---")

        st.markdown(msg_obj.get("content", "").strip())

        if msg_obj["role"] == "assistant":
            if msg_obj.get("plotly_fig_json"):
                try:
                    plotly_fig = plotly_io.from_json(msg_obj["plotly_fig_json"])
                    plot_key_hist = f"plot_hist_{i}_{msg_obj.get('timestamp', datetime.now().timestamp())}"
                    st.plotly_chart(plotly_fig, use_container_width=True, key=plot_key_hist)
                except Exception as e_plot_render:
                    st.error(f"Error rendering interactive plot from history: {e_plot_render}")

            if msg_obj.get("plot_insights"):
                with st.expander("🔍 View Plot Insights/Summary", expanded=False):
                    st.markdown(msg_obj["plot_insights"])

            thinking_expander_title = "⚙️ View Agent's Thinking & Configuration"
            show_thinking_details = bool(
                msg_obj.get("plot_config_json")
                or msg_obj.get("thinking_log_str")
                or (msg_obj.get("error") and msg_obj.get("response_type") == "error")
            )

            if show_thinking_details:
                with st.expander(thinking_expander_title, expanded=False):
                    if msg_obj.get("plot_config_json"):
                        st.write("**Plot Configuration Used:**")
                        try:
                            config_dict = json.loads(msg_obj["plot_config_json"])
                            st.json(config_dict, expanded=False)
                        except json.JSONDecodeError:
                            st.text(msg_obj["plot_config_json"])

                    if msg_obj.get("thinking_log_str"):
                        st.write("**Agent's Process Log:**")
                        log_key_hist_text = f"log_hist_text_{i}_{msg_obj.get('timestamp', datetime.now().timestamp())}"
                        st.text_area(
                            "Log Details:",
                            value=msg_obj["thinking_log_str"],
                            height=200,
                            disabled=True,
                            key=log_key_hist_text,
                        )

                    if msg_obj.get("response_type"):
                        st.caption(f"Agent Action Type: `{msg_obj['response_type']}`")
                    if msg_obj.get("error") and msg_obj.get("response_type") == "error":
                        st.error(f"Agent Error Detail: {msg_obj['error']}")

if prompt := st.chat_input(
    "Ask about your data or request a plot...",
    key="main_chat_input_widget",
    disabled=not st.session_state.current_session_id,
):
    current_message_timestamp = datetime.now().isoformat()
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": current_message_timestamp}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pre_summary_content_full = ""
        qna_content_full = ""
        pre_summary_placeholder = st.empty()
        qna_placeholder = st.empty()
        final_response_container = st.container()
        assistant_response_for_history = {
            "role": "assistant",
            "timestamp": current_message_timestamp,
        }
        payload = {"session_id": st.session_state.current_session_id, "query": prompt}

        try:
            with requests.post(
                PROCESS_QUERY_URL, json=payload, stream=True, timeout=360
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode("utf-8"))
                            chunk_type = chunk_data.get("type")

                            if chunk_type == "thinking_process_update":
                                pre_summary_content_full += chunk_data.get("chunk", "")
                                pre_summary_placeholder.markdown(
                                    ">" + pre_summary_content_full.strip() + "▌"
                                )
                            elif chunk_type == "content":
                                qna_content_full += chunk_data.get("chunk", "")
                                qna_placeholder.markdown(
                                    qna_content_full.strip() + "▌"
                                )
                            elif chunk_type == "final_agent_response":
                                response_data = chunk_data.get("data", {})
                                assistant_response_for_history.update(response_data)

                                pre_summary_placeholder.empty()
                                qna_placeholder.empty()

                                with final_response_container:
                                    main_content_final = response_data.get(
                                        "content", "Processing complete."
                                    )
                                    st.markdown(main_content_final.strip())
                                    if "content" not in assistant_response_for_history or not assistant_response_for_history["content"]:
                                        assistant_response_for_history["content"] = main_content_final

                                    if response_data.get("plotly_fig_json"):
                                        try:
                                            plotly_fig_final = plotly_io.from_json(
                                                response_data["plotly_fig_json"]
                                            )
                                            st.session_state.plot_key_counter += 1
                                            plot_key_stream = f"plot_stream_{st.session_state.plot_key_counter}_{current_message_timestamp}"
                                            st.plotly_chart(
                                                plotly_fig_final,
                                                use_container_width=True,
                                                key=plot_key_stream,
                                            )
                                        except Exception as e_plot_final:
                                            st.error(
                                                f"Error rendering interactive plot from final response: {e_plot_final}"
                                            )

                                    if response_data.get("plot_insights"):
                                        with st.expander(
                                            "🔍 View Plot Insights/Summary", expanded=True
                                        ):
                                            st.markdown(response_data["plot_insights"])

                                    thinking_exp_title_final = "⚙️ View Agent's Thinking & Configuration (Final)"
                                    show_thinking_final = bool(
                                        response_data.get("plot_config_json")
                                        or response_data.get("thinking_log_str")
                                        or (
                                            response_data.get("error")
                                            and response_data.get("response_type")
                                            == "error"
                                        )
                                    )
                                    if show_thinking_final:
                                        with st.expander(
                                            thinking_exp_title_final, expanded=False
                                        ):
                                            if response_data.get("plot_config_json"):
                                                st.write("**Plot Configuration Used:**")
                                                try:
                                                    st.json(
                                                        json.loads(
                                                            response_data[
                                                                "plot_config_json"
                                                            ]
                                                        ),
                                                        expanded=False,
                                                    )
                                                except:
                                                    st.text(
                                                        response_data[
                                                            "plot_config_json"
                                                        ]
                                                    )
                                            if response_data.get("thinking_log_str"):
                                                st.write("**Agent's Process Log:**")
                                                log_key_final_stream_text = f"log_final_stream_text_{current_message_timestamp}"
                                                st.text_area(
                                                    "Log Details:",
                                                    value=response_data[
                                                        "thinking_log_str"
                                                    ],
                                                    height=200,
                                                    disabled=True,
                                                    key=log_key_final_stream_text,
                                                )
                                            if response_data.get("response_type"):
                                                st.caption(
                                                    f"Agent Action Type: `{response_data['response_type']}`"
                                                )
                                            if response_data.get(
                                                "error"
                                            ) and response_data.get(
                                                "response_type"
                                            ) == "error":
                                                st.error(
                                                    f"Agent Error Detail: {response_data['error']}"
                                                )
                            elif chunk_type == "system":
                                print(
                                    f"System Message from Stream: {chunk_data.get('message')}"
                                )
                            elif chunk_type == "error":
                                error_msg_chunk = chunk_data.get(
                                    "chunk"
                                ) or chunk_data.get(
                                    "content", "Unknown error from stream."
                                )
                                st.error(f"Backend Stream Error: {error_msg_chunk}")
                                assistant_response_for_history[
                                    "content"
                                ] = f"Error from backend: {error_msg_chunk}"
                                assistant_response_for_history[
                                    "response_type"
                                ] = "error"
                                break
                        except json.JSONDecodeError:
                            print(
                                f"Stream: Failed to decode JSON line: {line.decode('utf-8', errors='ignore')}"
                            )
                        except Exception as e_chunk_process:
                            print(f"Stream: Error processing chunk: {e_chunk_process}")
                            traceback.print_exc()
                            st.warning(
                                f"A minor error occurred while displaying part of the response: {e_chunk_process}"
                            )

                if pre_summary_content_full and hasattr(pre_summary_placeholder, 'empty') and not pre_summary_placeholder._is_empty:
                    pre_summary_placeholder.markdown(
                        ">" + pre_summary_content_full.strip()
                    )
                    if "pre_summary_content" not in assistant_response_for_history:
                        assistant_response_for_history["pre_summary_content"] = pre_summary_content_full

                if qna_content_full and hasattr(qna_placeholder, 'empty') and not qna_placeholder._is_empty:
                    qna_placeholder.markdown(qna_content_full.strip())
                    if "content" not in assistant_response_for_history or not assistant_response_for_history["content"]:
                        assistant_response_for_history["content"] = qna_content_full
                        if "response_type" not in assistant_response_for_history:
                             assistant_response_for_history["response_type"] = "query_data"


                if "content" not in assistant_response_for_history or not assistant_response_for_history.get("content", "").strip():
                    assistant_response_for_history[
                        "content"
                    ] = "Request processed. (No specific textual output was generated for this query)"
                    if "response_type" not in assistant_response_for_history:
                        assistant_response_for_history["response_type"] = "unknown"

                st.session_state.messages.append(assistant_response_for_history)

        except requests.exceptions.Timeout:
            err_msg = f"Request timed out connecting to the backend at {PROCESS_QUERY_URL}."
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.HTTPError as http_err_main:
            error_detail_main = "No specific error detail from server."
            try:
                error_detail_main = (
                    http_err_main.response.json().get("detail", str(http_err_main))
                    if http_err_main.response
                    else str(http_err_main)
                )
            except: # Fallback if response is not JSON
                error_detail_main = http_err_main.response.text if http_err_main.response and http_err_main.response.text else str(http_err_main)
            err_msg = f"Query failed (HTTP {http_err_main.response.status_code if http_err_main.response else 'Unknown'}): {error_detail_main}"
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except requests.exceptions.RequestException as req_err_main:
            err_msg = f"Query failed: Could not connect to the backend at {PROCESS_QUERY_URL}. Error: {req_err_main}"
            st.error(err_msg)
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)
        except Exception as e_main_query:
            err_msg = f"An unexpected error occurred in the Streamlit app while processing your query: {e_main_query}"
            st.error(err_msg)
            traceback.print_exc()
            assistant_response_for_history["content"] = err_msg
            assistant_response_for_history["response_type"] = "error"
            st.session_state.messages.append(assistant_response_for_history)