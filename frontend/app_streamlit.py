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

# --- Initialize session state variables ---
default_session_vars = {
    "messages": [], "current_session_id": None, "df_columns": [], "df_head": "",
    "current_filename": "", "plot_key_counter": 0,
    "show_recorder_ui_modal": False, 
    "show_next_steps_screencast_modal": False, 
    "screencast_blob_url_for_preview": None, 
    "screencast_filename_from_js": "advanced-screen-recording.webm",
    "audio_enabled_for_js_recorder": True, 
    "recorder_component_value": None,
    "current_max_recording_time": None # Added for timed recordings
}
for key, value in default_session_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Conversational Data Agent")


# --- Main App Area ---
header_cols = st.columns([0.8, 0.2]) 
with header_cols[0]:
    st.title("📊 Conversational Data Analysis Agent")

with header_cols[1]:
    if st.session_state.current_session_id:
        # --- MODIFIED: Replaced single button with a popover for time limits ---
        with st.popover("🎬 Record Screencast", use_container_width=True):
            st.markdown("**Select recording duration:**")
            
            def set_recording_params(max_time=None):
                st.session_state.current_max_recording_time = max_time
                st.session_state.show_recorder_ui_modal = True
                st.session_state.show_next_steps_screencast_modal = False 
                st.session_state.screencast_blob_url_for_preview = None 
                # st.rerun() # Rerun will be triggered by Streamlit implicitly after button click

            if st.button("No Limit", key="record_no_limit_popover", use_container_width=True, on_click=set_recording_params, args=(None,)):
                pass # on_click handles the logic
            if st.button("15 Seconds", key="record_15s_popover", use_container_width=True, on_click=set_recording_params, args=(15,)):
                pass
            if st.button("30 Seconds", key="record_30s_popover", use_container_width=True, on_click=set_recording_params, args=(30,)):
                pass
            if st.button("1 Minute", key="record_1m_popover", use_container_width=True, on_click=set_recording_params, args=(60,)):
                pass
    else:
        st.caption("Upload data to enable screencast.")


# --- MODAL 1: Record a Screencast UI (using your JS component) ---
if st.session_state.show_recorder_ui_modal and st.session_state.current_session_id:
    with st.container():
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True)

        m_header, m_close = st.columns([0.9, 0.1])
        with m_header:
            st.subheader("⏺️ Record a screencast")
        with m_close:
            if st.button("✖", key="close_recorder_ui_modal", help="Close Recorder UI"):
                st.session_state.show_recorder_ui_modal = False
                st.rerun()
        
        st.write("This will record your screen using the browser. Ensure you grant necessary permissions.")
        st.session_state.audio_enabled_for_js_recorder = st.checkbox(
            "🎤 Also record audio",
            value=st.session_state.audio_enabled_for_js_recorder,
            key="audio_for_js_recorder_checkbox"
        )
        
        # --- Get current max recording time for JS ---
        max_rec_time_py = st.session_state.get("current_max_recording_time")
        if max_rec_time_py:
            st.caption(f"Recording will be limited to {max_rec_time_py} seconds. Download is direct from browser.")
        else:
            st.caption("Recording has no time limit. Download is direct from browser.")


        PACKAGE_NAME = "advanced-screen-recorder" 
        PACKAGE_VERSION = "0.1.0" 

        if PACKAGE_NAME.startswith('@'):
            cdn_url = f"https://cdn.jsdelivr.net/npm/{PACKAGE_NAME}@{PACKAGE_VERSION}/dist/index.js" 
        else:
            cdn_url = f"https://cdn.jsdelivr.net/npm/{PACKAGE_NAME}@{PACKAGE_VERSION}/+esm"

        component_key = f"adv_ssr_component_{st.session_state.current_session_id}_{max_rec_time_py or 'nolimit'}"

        html_component = f"""
        <div id="recorderInterfaceRootSSR_modal" style="font-family: sans-serif; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: white;">
            <p style="font-size:0.9em; margin-bottom:10px;">Click "Start Recording" below. Use "Stop" when finished.</p>
            <button id="startRecordButtonSSR_modal" style="padding: 8px 12px; margin-right: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Start Recording</button>
            <button id="stopRecordButtonSSR_modal" style="padding: 8px 12px; background-color: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer;" disabled>Stop Recording</button>
            <p id="recorderStatusSSR_modal" style="margin-top: 10px; font-size: 0.9em;">Status: Idle</p>
            <p id="actionStatusSSR_modal" style="font-size: 0.9em;"></p>
        </div>

        <script type="module">
            const rootElSSR_modal = document.getElementById('recorderInterfaceRootSSR_modal');
            let AdvancedScreenRecorder_modal, ASPlayerStatus_modal; 

            function debounce(func, wait) {{ /* ... debounce function ... */
                let timeout;
                return function executedFunction(...args) {{
                    const later = () => {{
                        clearTimeout(timeout);
                        func(...args);
                    }};
                    clearTimeout(timeout);
                    timeout = setTimeout(later, wait);
                }};
            }}

            try {{
                const module = await import('{cdn_url}');
                AdvancedScreenRecorder_modal = module.AdvancedScreenRecorder;
                ASPlayerStatus_modal = module.ASPlayerStatus; 

                if (!AdvancedScreenRecorder_modal || !ASPlayerStatus_modal) {{
                    throw new Error('AdvancedScreenRecorder or ASPlayerStatus not found in {PACKAGE_NAME}. Check exports.');
                }}

                const startButton_modal = document.getElementById('startRecordButtonSSR_modal');
                const stopButton_modal = document.getElementById('stopRecordButtonSSR_modal');
                const recorderStatusEl_modal = document.getElementById('recorderStatusSSR_modal');
                const actionStatusEl_modal = document.getElementById('actionStatusSSR_modal');
                
                const includeAudio = {json.dumps(st.session_state.audio_enabled_for_js_recorder)};
                const maxRecTimeFromPython = {json.dumps(max_rec_time_py)}; // Injected max recording time

                const sendValueToStreamlitDebounced = debounce((value) => {{
                    Streamlit.setComponentValue(value);
                }}, 300);

                // --- Construct recorder options, including maxRecordingTimeSeconds ---
                const recorderOptions = {{
                    mediaStreamConstraints: {{ 
                        audio: includeAudio 
                    }},
                    onStatusChange: (status, details) => {{
                        if (!recorderStatusEl_modal) return;
                        let statusText = `Status: ${{status}}`;
                        if (details) {{
                            statusText += ` (${{details instanceof Error ? details.message : String(details)}})`;
                        }}
                        // Add time limit info to status if applicable and not already part of a detailed message
                        if (typeof maxRecTimeFromPython === 'number' && maxRecTimeFromPython > 0 && status === ASPlayerStatus_modal.IDLE && !details) {{
                            statusText += ` (Limit: ${{maxRecTimeFromPython}}s)`;
                        }}
                        recorderStatusEl_modal.textContent = statusText;

                        startButton_modal.disabled = status === ASPlayerStatus_modal.RECORDING || status === ASPlayerStatus_modal.REQUESTING_PERMISSION;
                        stopButton_modal.disabled = status !== ASPlayerStatus_modal.RECORDING;
                        if (status !== ASPlayerStatus_modal.STOPPED) {{
                           actionStatusEl_modal.textContent = "";
                        }}
                         recorderStatusEl_modal.style.color = (status === ASPlayerStatus_modal.ERROR || status === ASPlayerStatus_modal.PERMISSION_DENIED) ? "red" : "inherit";
                    }},
                    onRecordingComplete: (blob, filename) => {{
                        AdvancedScreenRecorder_modal.downloadBlob(blob, filename); 
                        actionStatusEl_modal.textContent = `Download of '${{filename}}' initiated. Check browser downloads.`;
                        actionStatusEl_modal.style.color = "green";
                        console.log(`Client-side download initiated for ${{filename}}`);
                        
                        const blobUrl = URL.createObjectURL(blob);
                        sendValueToStreamlitDebounced({{
                            type: "recordingComplete",
                            filename: filename,
                            blobUrl: blobUrl,
                        }});
                        recorderStatusEl_modal.textContent = "Status: Download processed. You can close this recorder.";
                    }}
                }};

                // Add maxRecordingTimeSeconds to options if it's a valid number
                if (typeof maxRecTimeFromPython === 'number' && maxRecTimeFromPython > 0) {{
                    recorderOptions.maxRecordingTimeSeconds = maxRecTimeFromPython;
                    if (recorderStatusEl_modal.textContent.includes("Idle")) {{ // Update initial idle message
                         recorderStatusEl_modal.textContent = `Status: Idle (Limit: ${{maxRecTimeFromPython}}s)`;
                    }}
                }}
                
                const recorder_modal = new AdvancedScreenRecorder_modal(recorderOptions);

                startButton_modal.onclick = async () => {{
                    actionStatusEl_modal.textContent = '';
                    try {{ 
                        await recorder_modal.startRecording(); 
                    }}
                    catch (err) {{ 
                        console.error("Error from recorder.startRecording() call:", err); 
                        recorderStatusEl_modal.textContent = `Status: Error - ${{err.message}}`;
                        recorderStatusEl_modal.style.color = "red";
                    }}
                }};

                stopButton_modal.onclick = () => {{ recorder_modal.stopRecording(); }};

            }} catch (err) {{
                console.error("Failed to load or initialize AdvancedScreenRecorder from CDN (modal): ", err);
                if (rootElSSR_modal) {{
                    rootElSSR_modal.innerHTML = `<p style='color:red;'><b>Error loading screen recorder:</b><br/>${{err.message}}.<br/>Package: '<b>${PACKAGE_NAME}</b>@<b>${PACKAGE_VERSION}</b>'</p><p>CDN URL: {cdn_url}</p><p>Check browser console (F12) and ensure the package is published correctly and the CDN URL is valid.</p>`;
                }}
            }}
        </script>
        """
        component_return_value = components.html(html_component, height=250, scrolling=False) # Added dynamic key

        if component_return_value is not None: 
            if isinstance(component_return_value, dict):
                if st.session_state.get("recorder_component_value_processed_id") != id(component_return_value):
                    st.session_state.recorder_component_value_processed_id = id(component_return_value)

                    if component_return_value.get("type") == "recordingComplete":
                        st.session_state.screencast_blob_url_for_preview = component_return_value.get("blobUrl")
                        st.session_state.screencast_filename_from_js = component_return_value.get("filename", "advanced-screen-recording.webm")
                        st.session_state.show_recorder_ui_modal = False 
                        st.session_state.show_next_steps_screencast_modal = True 
                        st.rerun()
            else:
                print(f"Unexpected data type from HTML component: {type(component_return_value)}, value: {component_return_value}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- MODAL 2: Next Steps After Screencast ---
if st.session_state.show_next_steps_screencast_modal:
    with st.container():
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True)
        m2_header, m2_close = st.columns([0.9, 0.1])
        with m2_header:
            st.subheader("🎉 Recording Downloaded! Next steps...")
        with m2_close:
            if st.button("✖", key="close_next_steps_screencast", help="Close Next Steps"):
                st.session_state.show_next_steps_screencast_modal = False
                st.session_state.screencast_blob_url_for_preview = None
                st.rerun()

        st.markdown("---")
        st.markdown("#### Step 1: Preview your video (if browser allows)")
        if st.session_state.screencast_blob_url_for_preview:
            try:
                st.video(st.session_state.screencast_blob_url_for_preview, format="video/webm") 
                st.caption(f"Playing: `{st.session_state.screencast_filename_from_js}`")
            except Exception as e_video_preview:
                st.warning(f"Could not display video preview directly in Streamlit: {e_video_preview}")
                st.info("You can preview the downloaded file in your browser or a media player.")
        else:
            st.info("No video preview available. The recording was downloaded by your browser.")

        st.markdown("---")
        st.markdown(f"#### Step 2: Locate the downloaded file")
        st.write(f"Your recording (`{st.session_state.screencast_filename_from_js}`) has been downloaded by your browser. Please check your browser's default download location.")
        st.caption("The video is likely in WebM format.")

        st.markdown("---")
        st.markdown("#### Step 3: Share your video")
        st.caption("You can now share this video on various platforms or via email.")

        st.markdown("</div>", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader_widget")

    if uploaded_file:
        with st.form("upload_form", clear_on_submit=True):
            submitted = st.form_submit_button("Upload and Start New Session")
            if submitted:
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
                        st.session_state.show_recorder_ui_modal = False
                        st.session_state.show_next_steps_screencast_modal = False
                        st.session_state.screencast_blob_url_for_preview = None
                        st.session_state.current_max_recording_time = None # Reset on new session
                        st.success(f"File '{st.session_state.current_filename}' processed! Session: ...{st.session_state.current_session_id[-6:]}")
                        st.rerun() 
                    except Exception as e:
                        st.error(f"File upload failed: {str(e)[:500]}")
                        traceback.print_exc()

    if st.session_state.current_session_id:
        st.markdown("---")
        st.success(f"Active Session: `{st.session_state.current_session_id}`") 
        st.info(f"Current File: **{st.session_state.current_filename}**")
        with st.expander("Data Preview", expanded=False):
            st.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True, key="sidebar_df_cols_preview")
            st.text_area("Data Head:", value=st.session_state.df_head, height=150, disabled=True, key="sidebar_df_head_preview")
    else:
        st.info("Upload a data file to begin.")

# --- Main Chat Interface (remains unchanged, ensure this part is identical to your previous working version) ---
st.header("2. Chat with Your Data") # ... (rest of your chat interface code) ...
# (The existing chat interface code is assumed to be here, as it was long and not directly modified by this request)
# Please re-insert your full chat interface code from the "st.header("2. Chat with Your Data")" line onwards.
# For brevity in this response, I am omitting the full chat UI, but ensure it's in your final file.
# The following is a placeholder for where your chat UI code should be:

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
                    if assistant_response_for_history.get("response_type") != "error": 
                        assistant_response_for_history[
                            "content"
                        ] = "Request processed. (No specific textual output was generated for this query)"
                        if "response_type" not in assistant_response_for_history:
                            assistant_response_for_history["response_type"] = "unknown_visualize_or_action"


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
            except: 
                error_detail_main = http_err_main.response.text if http_err_main.response and http_err_main.response.text else str(http_err_main)
            
            err_msg = f"Query failed (HTTP {http_err_main.response.status_code if http_err_main.response else 'Unknown Status'}): {error_detail_main}"
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

# --- Global CSS for Modal Styling (Basic) ---
st.markdown(
    """
    <style>
    .modal-box {
        padding: 25px;
        background-color: #2E2E38; 
        border-radius: 10px;
        border: 1px solid #4a4a58;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .modal-box h3, .modal-box h4, .modal-box p, .modal-box small, .modal-box label, 
    .modal-box div[data-testid="stCaptionContainer"], .modal-box li, .modal-box strong {
        color: #FFFFFF !important;
    }
    .modal-box .stCheckbox > label > div {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True
)