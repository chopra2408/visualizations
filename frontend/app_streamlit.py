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

# --- URLs (Backend interaction for screencast is now REMOVED) ---
UPLOAD_DATA_URL = f"{BACKEND_BASE_URL.rstrip('/')}/uploadfile/"
PROCESS_QUERY_URL = f"{BACKEND_BASE_URL.rstrip('/')}/process_query/"
# UPLOAD_CLIENT_CONVERTED_SCREENCAST_URL is REMOVED

# --- Initialize session state variables ---
default_session_vars = {
    "messages": [], "current_session_id": None, "df_columns": [], "df_head": "",
    "current_data_filename": "", "plot_key_counter": 0,
    "show_recorder_ui_modal": False,
    "show_next_steps_screencast_modal": False,
    "screencast_local_blob_url_for_preview": None,
    "screencast_final_filename_from_js": "screen-recording.webm",
    "audio_enabled_for_js_recorder": True,
    "recorder_component_value": None,
    "current_max_recording_time": None,
    "selected_target_format_for_js": "webm", # "webm", "mp4", "mkv" - what JS should try to make
    "popover_selected_format_radio": "webm", # Tracks the radio: 'webm', 'mp4', 'mkv'
    # Server upload related states are REMOVED as we are not uploading screencasts
    # "server_upload_status": "idle",
    # "server_uploaded_file_info": None,
    # "last_upload_error_message": None,
    "recorder_component_value_processed_id": None,
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
        with st.popover("🎬 Record Screencast", use_container_width=True):
            st.markdown("**1. Select Desired Output Format:**")
            st.session_state.popover_selected_format_radio = st.radio(
                "Recording Format (Browser will attempt to generate):",
                options=["webm", "mp4", "mkv"],
                format_func=lambda x: {
                    "webm": "WebM (Recommended, Direct Download)",
                    "mp4": "MP4 (Browser Generation, Direct Download)",
                    "mkv": "MKV (Browser Generation, Direct Download)"
                }.get(x, x.upper()),
                key="popover_format_radio_main_v3",
                horizontal=True,
                index=["webm", "mp4", "mkv"].index(st.session_state.get("popover_selected_format_radio", "webm"))
            )

            st.markdown("**2. Select Recording Duration:**")

            def set_and_open_recorder_modal(max_time=None):
                # The radio choice directly becomes the target format for JS
                st.session_state.selected_target_format_for_js = st.session_state.popover_selected_format_radio

                st.session_state.current_max_recording_time = max_time
                st.session_state.show_recorder_ui_modal = True
                st.session_state.show_next_steps_screencast_modal = False
                st.session_state.screencast_local_blob_url_for_preview = None
                # No server upload states to reset

            time_limit_buttons_col1, time_limit_buttons_col2 = st.columns(2)
            current_radio_format_key_suffix = st.session_state.popover_selected_format_radio

            with time_limit_buttons_col1:
                st.button("No Limit", key=f"record_nolimit_{current_radio_format_key_suffix}_v3", use_container_width=True, on_click=set_and_open_recorder_modal, args=(None,))
                st.button("30 Seconds", key=f"record_30s_{current_radio_format_key_suffix}_v3", use_container_width=True, on_click=set_and_open_recorder_modal, args=(30,))
            with time_limit_buttons_col2:
                st.button("15 Seconds", key=f"record_15s_{current_radio_format_key_suffix}_v3", use_container_width=True, on_click=set_and_open_recorder_modal, args=(15,))
                st.button("1 Minute", key=f"record_1m_{current_radio_format_key_suffix}_v3", use_container_width=True, on_click=set_and_open_recorder_modal, args=(60,))
    else:
        st.caption("Upload data to enable screencast.")


# --- MODAL 1: Record a Screencast UI ---
if st.session_state.show_recorder_ui_modal and st.session_state.current_session_id:
    with st.container():
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True)
        m_header, m_close = st.columns([0.9, 0.1])
        with m_header:
            st.subheader("⏺️ Record a screencast")
        with m_close:
            if st.button("✖", key="close_recorder_ui_modal_comp_v3", help="Close Recorder UI"):
                st.session_state.show_recorder_ui_modal = False
                st.rerun()

        st.write("Grant screen sharing permissions. A 3s countdown appears after selection.")
        st.session_state.audio_enabled_for_js_recorder = st.checkbox(
            "🎤 Also record audio",
            value=st.session_state.audio_enabled_for_js_recorder,
            key="audio_for_js_recorder_checkbox_comp_modal_v3"
        )

        max_rec_time_py = st.session_state.get("current_max_recording_time")
        target_format_for_js = st.session_state.get("selected_target_format_for_js", "webm")

        format_display_name_map = {
            "webm": "WebM (Recommended)",
            "mp4": "MP4 (Browser will attempt generation)",
            "mkv": "MKV (Browser will attempt generation)"
        }
        format_display_name = format_display_name_map.get(target_format_for_js, "Unknown Format")

        caption_text = f"Target Format: **{format_display_name.upper()}**. Your browser will attempt to generate this format."
        if max_rec_time_py:
            caption_text += f" Time Limit: **{max_rec_time_py}s**."
        else:
            caption_text += " No time limit."
        caption_text += " The video will be downloaded directly to your computer after recording."
        st.caption(caption_text)

        js_component_params = {
            "audio": st.session_state.audio_enabled_for_js_recorder,
            "timeLimit": max_rec_time_py,
            "delay": 3,
            "format": target_format_for_js, # "webm", "mp4", "mkv" - what JS recorder should try to make
            # "processingStrategy" and "uploadUrl" are REMOVED as all are direct downloads
        }

        component_instance_key = f"custom_ssr_instance_key_{st.session_state.current_session_id}_{target_format_for_js}_{max_rec_time_py or 'nolimit'}_v3"

        html_component = f"""
        <div id="recorderInterfaceRootSSR_modal_v3" style="font-family: sans-serif; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: white;">
            <button id="startRecordButtonSSR_modal_v3" style="padding: 8px 12px; margin-right: 10px; color: white; border: none; border-radius: 4px; cursor: pointer; background-color: #4CAF50;">Start Recording</button>
            <button id="stopRecordButtonSSR_modal_v3" style="padding: 8px 12px; margin-right: 10px; color: white; border: none; border-radius: 4px; cursor: pointer; background-color: #f44336;" disabled>Stop Recording</button>
            <p id="countdownSSR_modal_v3" style="font-size: 1.5em; color: #FFA500; margin-top:10px; min-height:1.2em;"></p>
            <p id="recorderStatusSSR_modal_v3" style="margin-top: 5px; font-size: 0.9em;">Status: Idle</p>
            <p id="actionStatusSSR_modal_v3" style="font-size: 0.9em; min-height:1.2em;"></p>
            {'''
            <!-- Upload button is REMOVED -->
            '''}
        </div>

        <script type="module">
            // --- Embedded ScreenRecorder Class (from previous correct version) ---
            class ScreenRecorder {{
                constructor() {{ /* ... same definition as before ... */ this.mediaStream = null; this.mediaRecorder = null; this.recordedBlobs = []; this.timerInterval = null; this.countdownInterval = null; this.onRecordingStart = null; this.onRecordingStop = null; this.onCountdown = null; this.onError = null; this.onStatusUpdate = null; }}
                _getMimeType(requestedFormat = 'webm') {{ /* ... same definition as before ... */ const format = requestedFormat.toLowerCase(); let preferredType = null; const types = {{ webm: ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8,opus', 'video/webm;codecs=vp8', 'video/webm'], mp4: ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=h264', 'video/mp4'], mkv: ['video/x-matroska;codecs=avc1', 'video/x-matroska;codecs=vp9', 'video/x-matroska'] }}; const checkTypes = types[format] || types.webm; for (const type of checkTypes) {{ if (MediaRecorder.isTypeSupported(type)) {{ preferredType = type; break; }} }} if (!preferredType) {{ if (format === 'mp4' && MediaRecorder.isTypeSupported('video/mp4')) preferredType = 'video/mp4'; else if (MediaRecorder.isTypeSupported('video/webm')) preferredType = 'video/webm'; else {{ console.error("No suitable MIME type for MediaRecorder."); return {{ mimeType: null, fileExtension: 'unknown' }}; }} }} let fileExtension = format; if (preferredType.includes('mp4')) fileExtension = 'mp4'; else if (preferredType.includes('x-matroska')) fileExtension = 'mkv'; else if (preferredType.includes('webm')) fileExtension = 'webm'; console.log(`[Recorder] Requested: ${{format}}, Effective MIME: ${{preferredType}}, Extension: ${{fileExtension}}`); return {{ mimeType: preferredType, fileExtension: fileExtension }}; }}
                async startRecording(options = {{}}) {{ /* ... same definition as before ... */ const {{ timeLimit = null, delay = 3, audio = true, format = 'webm' }} = options; if (this.mediaRecorder && this.mediaRecorder.state === "recording") {{ const err = new Error("Recording in progress."); if (this.onError) this.onError(err); return Promise.reject(err); }} this.recordedBlobs = []; if(this.onStatusUpdate) this.onStatusUpdate("Requesting screen permission..."); try {{ this.mediaStream = await navigator.mediaDevices.getDisplayMedia({{ video: true, audio: audio }}); this.mediaStream.getVideoTracks()[0].onended = () => {{ if(this.onStatusUpdate) this.onStatusUpdate("Screen sharing stopped by user."); this.stopRecording(false); }}; if (this.onCountdown) this.onCountdown(delay); let countdown = delay; return new Promise((resolve, reject) => {{ this.countdownInterval = setInterval(() => {{ countdown--; if (this.onCountdown) this.onCountdown(countdown); if (countdown <= 0) {{ clearInterval(this.countdownInterval); this.countdownInterval = null; try {{ this._initiateRecording(timeLimit, format); resolve(); }} catch (initError) {{ if (this.onError) this.onError(initError); this._cleanup(); reject(initError); }} }} }}, 1000); }}); }} catch (err) {{ if(this.onStatusUpdate) this.onStatusUpdate(`Permission denied/error: ${{err.message}}`); if (this.onError) this.onError(err); this._cleanup(); return Promise.reject(err); }} }}
                _initiateRecording(timeLimit, requestedFormat) {{ /* ... same definition as before ... */ const {{ mimeType, fileExtension }} = this._getMimeType(requestedFormat); if (!mimeType) {{ const err = new Error("MIME type selection failed for MediaRecorder."); if (this.onError) this.onError(err); this._cleanup(); throw err; }} try {{ this.mediaRecorder = new MediaRecorder(this.mediaStream, {{ mimeType }}); }} catch (e) {{ const err = new Error(`Failed to create MediaRecorder: ${{e.message}} (MIME: ${{mimeType}})`); if (this.onError) this.onError(err); this._cleanup(); throw err; }} this.mediaRecorder.onstop = () => {{ if (this.timerInterval) {{ clearTimeout(this.timerInterval); this.timerInterval = null; }} const blob = new Blob(this.recordedBlobs, {{ type: this.mediaRecorder.mimeType }}); const finalFileName = `screen-recording-${{new Date().toISOString().replace(/[:.]/g, '-')}}.${{fileExtension}}`; console.log(`[Recorder] Stopped. Blob type: ${{blob.type}}, size: ${{blob.size}}, filename: ${{finalFileName}}`); if (this.onRecordingStop) this.onRecordingStop(blob, finalFileName, false); this._cleanup(); }}; this.mediaRecorder.ondataavailable = (event) => {{ if (event.data && event.data.size > 0) this.recordedBlobs.push(event.data); }}; this.mediaRecorder.start(); if(this.onStatusUpdate) this.onStatusUpdate(`Recording started (Attempting: ${{fileExtension}}, Actual MIME: ${{this.mediaRecorder.mimeType}}).`); if (this.onRecordingStart) this.onRecordingStart(); if (timeLimit && timeLimit > 0) {{ this.timerInterval = setTimeout(() => {{ if (this.mediaRecorder && this.mediaRecorder.state === "recording") this.stopRecording(); }}, timeLimit * 1000); }} }}
                stopRecording(stopMediaRecorderInstance = true) {{ /* ... same definition as before ... */ if (this.countdownInterval) {{ clearInterval(this.countdownInterval); this.countdownInterval = null; if (this.onRecordingStop) this.onRecordingStop(null, null, true); this._cleanup(); return; }} if (this.mediaRecorder && this.mediaRecorder.state === "recording" && stopMediaRecorderInstance) this.mediaRecorder.stop(); else if (this.mediaRecorder && this.mediaRecorder.state !== "inactive" && !stopMediaRecorderInstance) {{ if (this.mediaRecorder.state === "recording") this.mediaRecorder.stop(); else this._cleanup(); }} else this._cleanup(); if (this.timerInterval) {{ clearTimeout(this.timerInterval); this.timerInterval = null; }} }}
                _cleanup() {{ /* ... same definition as before ... */ if (this.mediaStream) {{ this.mediaStream.getTracks().forEach(track => track.stop()); this.mediaStream = null; }} this.mediaRecorder = null; this.recordedBlobs = []; if (this.timerInterval) clearTimeout(this.timerInterval); this.timerInterval = null; if (this.countdownInterval) clearInterval(this.countdownInterval); this.countdownInterval = null; if(this.onStatusUpdate) this.onStatusUpdate("Idle. Resources cleaned."); }}
                static download(blob, filename) {{ /* ... same definition as before ... */ const url = URL.createObjectURL(blob); const a = document.createElement("a"); document.body.appendChild(a); a.style.display = "none"; a.href = url; a.download = filename; a.click(); window.URL.revokeObjectURL(url); document.body.removeChild(a); console.log(`[Recorder] Download triggered for ${{filename}}`); }}
            }} // --- End of ScreenRecorder Class ---

            const params = {json.dumps(js_component_params)};
            const startBtn = document.getElementById('startRecordButtonSSR_modal_v3');
            const stopBtn = document.getElementById('stopRecordButtonSSR_modal_v3');
            const statusEl = document.getElementById('recorderStatusSSR_modal_v3');
            const countdownEl = document.getElementById('countdownSSR_modal_v3');
            const actionEl = document.getElementById('actionStatusSSR_modal_v3');
            // Upload button is removed from JS logic as well

            function debounce(func, wait) {{ /* ... same as before ... */ let timeout; return function executedFunction(...args) {{ const later = () => {{ clearTimeout(timeout); func(...args); }}; clearTimeout(timeout); timeout = setTimeout(later, wait); }}; }}
            const sendValueToStreamlitDebounced = debounce((value) => {{ if (window.Streamlit) window.Streamlit.setComponentValue(value); else console.warn("Streamlit obj not found"); }}, 300);

            const recorder = new ScreenRecorder();
            recorder.onStatusUpdate = (message) => {{ if (statusEl) statusEl.textContent = `Status: ${{message}}`; }};
            recorder.onCountdown = (count) => {{ if (countdownEl) countdownEl.textContent = count > 0 ? `Recording in ${{count}}...` : (count === 0 ? 'REC 🔴' : ''); }};
            recorder.onRecordingStart = () => {{
                if (statusEl) statusEl.textContent = "Status: Recording..."; if (countdownEl) countdownEl.textContent = "REC 🔴";
                if (startBtn) startBtn.disabled = true; if (stopBtn) stopBtn.disabled = false;
                if (actionEl) actionEl.textContent = '';
            }};
            recorder.onError = (error) => {{
                if (statusEl) {{ statusEl.textContent = `Error: ${{error.message}}`; statusEl.style.color="red"; }}
                if (countdownEl) countdownEl.textContent = ''; if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true; if (actionEl) actionEl.textContent = `Failed: ${{error.message}}`;
            }};

            recorder.onRecordingStop = async (blob, filename, cancelled) => {{
                if (countdownEl) countdownEl.textContent = ''; if (startBtn) startBtn.disabled = false; if (stopBtn) stopBtn.disabled = true;
                if (cancelled) {{ if (statusEl) statusEl.textContent = "Status: Recording cancelled."; if (actionEl) actionEl.textContent = "Operation cancelled."; return; }}
                if (!blob || !filename) {{ if (statusEl) statusEl.textContent = "Status: Stopped, no data captured."; if (actionEl) actionEl.textContent = "No video data recorded."; return; }}

                // ALL recordings are now direct download
                ScreenRecorder.download(blob, filename);
                let formatGenerated = params.format.toUpperCase();
                if (filename.includes('.')) {{ // Try to get actual extension from generated filename
                    formatGenerated = filename.split('.').pop().toUpperCase();
                }}
                if (actionEl) {{ actionEl.textContent = `${{formatGenerated}} Download of '${{filename}}' initiated.`; actionEl.style.color = "green"; }}
                
                sendValueToStreamlitDebounced({{
                    type: "directDownloadComplete", // Same event for all formats now
                    filename: filename,
                    blobUrl: URL.createObjectURL(blob), // For preview
                    generatedFormat: params.format // The originally requested format
                }});
                if (statusEl) statusEl.textContent = `Status: ${{formatGenerated}} downloaded. Close modal or record again.`;
            }};

            startBtn.onclick = async () => {{ /* ... same as before, ensure UI reset ... */ if(statusEl) statusEl.style.color="inherit"; if(actionEl) actionEl.textContent = ''; if(countdownEl) countdownEl.textContent = ''; recorder.startRecording({{ timeLimit: params.timeLimit, delay: params.delay, audio: params.audio, format: params.format }}).catch(err => {{}}); }};
            stopBtn.onclick = () => {{ recorder.stopRecording(); }};
            if (statusEl && params.timeLimit) statusEl.textContent = `Status: Idle (Limit: ${{params.timeLimit}}s)`; else if (statusEl) statusEl.textContent = `Status: Idle (No limit)`;
        </script>
        """
        try:
             # Height can be reduced as upload button is gone
             st.session_state.recorder_component_value = components.html(html_component, height=250, scrolling=False, key=component_instance_key)
        except TypeError as e_html_key:
            if "unexpected keyword argument 'key'" in str(e_html_key):
                #   st.warning("Hint: Streamlit version <1.11.0. `key` for components.html ignored.")
                st.session_state.recorder_component_value = components.html(html_component, height=250, scrolling=False)
            else: raise e_html_key

        if st.session_state.recorder_component_value is not None and \
        st.session_state.recorder_component_value_processed_id != id(st.session_state.recorder_component_value):
            st.session_state.recorder_component_value_processed_id = id(st.session_state.recorder_component_value)
            component_data = st.session_state.recorder_component_value
            if isinstance(component_data, dict):
                event_type = component_data.get("type")
                st.session_state.screencast_final_filename_from_js = component_data.get("originalFilename") or component_data.get("filename", "recording.bin")

                if event_type == "directDownloadComplete": # WebM
                    st.session_state.screencast_local_blob_url_for_preview = component_data.get("blobUrl")
                    st.session_state.server_upload_status = "not_applicable_direct_download" # Mark as not applicable
                    st.session_state.show_recorder_ui_modal = False
                    st.session_state.show_next_steps_screencast_modal = True
                    st.rerun()
                elif event_type == "serverUploadComplete": # MP4/MKV uploaded successfully
                    st.session_state.server_upload_status = "completed"
                    st.session_state.server_uploaded_file_info = {
                        "uploaded_filename_on_server": component_data.get("uploadedFilenameOnServer"),
                        "download_url_relative_path": component_data.get("downloadUrlRelativePath")
                    }
                    st.session_state.show_recorder_ui_modal = False
                    st.session_state.show_next_steps_screencast_modal = True
                    st.rerun()
                elif event_type == "serverUploadError": # MP4/MKV upload failed
                    st.session_state.server_upload_status = "error"
                    st.session_state.last_upload_error_message = component_data.get("error")
                    st.session_state.show_recorder_ui_modal = False # Still close recorder modal
                    st.session_state.show_next_steps_screencast_modal = True # Show next steps to report error
                    st.rerun()
            # No serverUploadComplete or serverUploadError events expected anymore
        st.markdown("</div>", unsafe_allow_html=True)

# --- MODAL 2: Next Steps After Screencast ---
if st.session_state.show_next_steps_screencast_modal:
    with st.container():
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True)
        m2_header, m2_close = st.columns([0.9, 0.1])
        with m2_header:
            st.subheader("🎉 Recording Downloaded!")
        with m2_close:
            if st.button("✖", key="close_next_steps_modal_comp_v3", help="Close"):
                st.session_state.show_next_steps_screencast_modal = False
                st.session_state.screencast_local_blob_url_for_preview = None
                st.rerun()
        st.markdown("---")

        final_filename = st.session_state.screencast_final_filename_from_js
        # Get the format that was actually generated/downloaded (passed from JS)
        generated_format = st.session_state.get("last_generated_format_for_modal2", "file").upper()

        st.success(f"Your {generated_format} recording (`{final_filename}`) was generated by your browser and downloaded directly.")

        # Determine mime type for preview based on filename extension (best effort)
        preview_mime_type = "application/octet-stream"
        if final_filename.lower().endswith(".webm"): preview_mime_type = "video/webm"
        elif final_filename.lower().endswith(".mp4"): preview_mime_type = "video/mp4"
        elif final_filename.lower().endswith(".mkv"): preview_mime_type = "video/x-matroska" # Browser support for MKV preview varies

        if st.session_state.screencast_local_blob_url_for_preview:
            st.markdown(f"#### Preview your {generated_format} video (if browser supports this format for preview)")
            try:
                st.video(st.session_state.screencast_local_blob_url_for_preview, format=preview_mime_type)
            except Exception as e_video_preview:
                st.warning(f"Could not display video preview: {e_video_preview}. The format might not be playable in this browser view, or the Blob URL was revoked.")
        else:
            st.info("No local preview available or preview already closed.")
        st.markdown(f"Please check your browser's default download location for `{final_filename}`.")
        st.markdown("---")
        st.markdown("#### Next Steps: Share your video from your downloads folder.")
        st.markdown("</div>", unsafe_allow_html=True)


# --- Sidebar (Existing, Unchanged) ---
# [ ... keep your existing sidebar code ... ]
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file_sidebar = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader_main_sidebar_comp_v3")
    if uploaded_file_sidebar:
        with st.form("upload_form_main_sidebar_comp_v3", clear_on_submit=True):
            submitted_sidebar = st.form_submit_button("Upload and Start New Session")
            if submitted_sidebar:
                with st.spinner("Uploading..."):
                    files_sidebar = {"file": (uploaded_file_sidebar.name, uploaded_file_sidebar.getvalue(), uploaded_file_sidebar.type)}
                    try:
                        upload_response_sidebar = requests.post(UPLOAD_DATA_URL, files=files_sidebar, timeout=120)
                        upload_response_sidebar.raise_for_status()
                        file_info_sidebar = upload_response_sidebar.json()
                        for key_default, val_default in default_session_vars.items():
                            if key_default not in ["messages"]: st.session_state[key_default] = val_default
                        st.session_state.messages = []
                        st.session_state.current_session_id = file_info_sidebar["session_id"]
                        st.session_state.df_columns = file_info_sidebar["columns"]
                        st.session_state.df_head = file_info_sidebar["df_head"]
                        st.session_state.current_data_filename = file_info_sidebar["filename"]
                        st.success(f"File '{st.session_state.current_data_filename}' processed! Session: ...{st.session_state.current_session_id[-6:]}")
                        st.rerun()
                    except Exception as e_sidebar_upload: st.error(f"File upload failed: {str(e_sidebar_upload)[:500]}"); traceback.print_exc()
    if st.session_state.current_session_id:
        st.markdown("---"); st.success(f"Active Session: `{st.session_state.current_session_id}`"); st.info(f"File: **{st.session_state.current_data_filename}**")
        with st.expander("Data Preview", expanded=False):
            st.text_area("Columns:", value=", ".join(st.session_state.df_columns), height=100, disabled=True, key="sidebar_df_cols_preview_comp_v3")
            st.text_area("Data Head:", value=st.session_state.df_head, height=150, disabled=True, key="sidebar_df_head_preview_comp_v3")
    else: st.info("Upload a data file to begin.")


st.header("2. Chat with Your Data")
for i, msg_obj in enumerate(st.session_state.messages):
    with st.chat_message(msg_obj["role"]):
        if msg_obj.get("pre_summary_content"): st.markdown(">" + msg_obj["pre_summary_content"].strip()); st.markdown("---")
        st.markdown(msg_obj.get("content", "").strip())
        if msg_obj["role"] == "assistant":
            if msg_obj.get("plotly_fig_json"):
                try: st.plotly_chart(plotly_io.from_json(msg_obj["plotly_fig_json"]),use_container_width=True,key=f"plot_hist_comp_v3_{i}_{msg_obj.get('timestamp', datetime.now().timestamp())}")
                except Exception as e: st.error(f"Plot render error: {e}")
            if msg_obj.get("plot_insights"):
                with st.expander("🔍 View Plot Insights/Summary", expanded=False): st.markdown(msg_obj["plot_insights"])
            # Corrected plot_config_json handling from previous issue
            if msg_obj.get("plot_config_json"):
                st.write("**Plot Config:**")
                plot_config = msg_obj["plot_config_json"]
                try:
                    if isinstance(plot_config, str): config_to_display = json.loads(plot_config)
                    elif isinstance(plot_config, (dict, list)): config_to_display = plot_config
                    else: st.text(f"Raw plot config (type: {type(plot_config)}):\n{str(plot_config)}"); config_to_display = None
                    if config_to_display is not None: st.json(config_to_display, expanded=False)
                except json.JSONDecodeError: st.text(f"Invalid JSON in plot_config_json:\n{plot_config}")
                except Exception as e_cfg: st.error(f"Err display plot_config: {e_cfg}"); st.text(f"Raw: {plot_config}")

            show_thinking = bool(msg_obj.get("thinking_log_str") or (msg_obj.get("error") and msg_obj.get("response_type") == "error"))
            if show_thinking: # Only show thinking expander if there's thinking log or an error to show
                with st.expander("⚙️ View Agent's Thinking & Configuration", expanded=False):
                    if msg_obj.get("thinking_log_str"): st.write("**Agent Log:**"); st.text_area("Log:",value=msg_obj["thinking_log_str"],height=200,disabled=True,key=f"log_hist_text_comp_v3_{i}_{msg_obj.get('timestamp', datetime.now().timestamp())}")
                    if msg_obj.get("response_type"): st.caption(f"Action Type: `{msg_obj['response_type']}`")
                    if msg_obj.get("error") and msg_obj.get("response_type") == "error": st.error(f"Agent Error: {msg_obj['error']}")

if prompt := st.chat_input("Ask about your data...",key="main_chat_input_widget_comp_v3",disabled=not st.session_state.current_session_id):
    ts = datetime.now().isoformat(); st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": ts})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        pre_summary_full, qna_full, final_resp_cont = "", "", st.container()
        pre_sum_ph, qna_ph = st.empty(), st.empty()
        hist_entry = {"role": "assistant", "timestamp": ts}
        payload = {"session_id": st.session_state.current_session_id, "query": prompt}
        try:
            with requests.post(PROCESS_QUERY_URL, json=payload, stream=True, timeout=360) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if chunk.get("type") == "thinking_process_update": pre_summary_full += chunk.get("chunk",""); pre_sum_ph.markdown(">" + pre_summary_full.strip() + "▌")
                            elif chunk.get("type") == "content": qna_full += chunk.get("chunk",""); qna_ph.markdown(qna_full.strip() + "▌")
                            elif chunk.get("type") == "final_agent_response":
                                data = chunk.get("data",{}); hist_entry.update(data); pre_sum_ph.empty(); qna_ph.empty()
                                with final_resp_cont:
                                    main_content = data.get("content", "Processed."); st.markdown(main_content.strip())
                                    if not hist_entry.get("content"): hist_entry["content"] = main_content
                                    if data.get("plotly_fig_json"):
                                        st.session_state.plot_key_counter+=1; st.plotly_chart(plotly_io.from_json(data["plotly_fig_json"]),use_container_width=True,key=f"plot_stream_comp_v3_{st.session_state.plot_key_counter}_{ts}")
                                    if data.get("plot_insights"): 
                                        with st.expander("🔍 Plot Insights", expanded=True): st.markdown(data["plot_insights"])
                                    # Corrected plot_config_json handling
                                    if data.get("plot_config_json"):
                                        st.write("**Plot Config (Final):**")
                                        plot_config_final = data["plot_config_json"]
                                        try:
                                            if isinstance(plot_config_final, str): config_to_display_final = json.loads(plot_config_final)
                                            elif isinstance(plot_config_final, (dict, list)): config_to_display_final = plot_config_final
                                            else: st.text(f"Raw final plot_config (type: {type(plot_config_final)}):\n{str(plot_config_final)}"); config_to_display_final = None
                                            if config_to_display_final is not None: st.json(config_to_display_final, expanded=False)
                                        except json.JSONDecodeError: st.text(f"Invalid JSON in final plot_config_json:\n{plot_config_final}")
                                        except Exception as e_cfg_final: st.error(f"Err display final plot_config: {e_cfg_final}"); st.text(f"Raw final: {plot_config_final}")

                                    show_final_thinking = bool(data.get("thinking_log_str") or (data.get("error") and data.get("response_type")=="error"))
                                    if show_final_thinking: # Only show if log or error
                                        with st.expander("⚙️ Agent Thinking (Final)", expanded=False):
                                            if data.get("thinking_log_str"): st.write("**Agent Log:**"); st.text_area("Log:",value=data["thinking_log_str"],height=200,disabled=True,key=f"log_final_stream_text_comp_v3_{ts}")
                                            if data.get("response_type"): st.caption(f"Action: `{data['response_type']}`")
                                            if data.get("error") and data.get("response_type")=="error": st.error(f"Agent Error: {data['error']}")
                            elif chunk.get("type") == "error": hist_entry.update({"content":f"Stream Error: {chunk.get('chunk',chunk.get('content','Unknown'))}", "response_type":"error"}); st.error(f"Stream Error: {chunk.get('chunk',chunk.get('content','Unknown'))}"); break
                        except Exception as e_chunk: print(f"Chunk proc error: {e_chunk}"); st.warning(f"Display error: {e_chunk}")
                if not qna_ph._is_empty: qna_ph.empty();
                if not pre_sum_ph._is_empty: pre_sum_ph.empty();
                if pre_summary_full and "pre_summary_content" not in hist_entry: hist_entry["pre_summary_content"] = pre_summary_full
                if qna_full and not hist_entry.get("content"): hist_entry["content"]=qna_full; hist_entry.setdefault("response_type","query_data")
                if not hist_entry.get("content","").strip() and hist_entry.get("response_type")!="error": hist_entry["content"]="Request processed."; hist_entry.setdefault("response_type","unknown_action")
                st.session_state.messages.append(hist_entry)
        except requests.exceptions.RequestException as e_req: err_msg=f"Connection Error: {e_req}"; st.error(err_msg); st.session_state.messages.append({"role":"assistant","content":err_msg,"response_type":"error","timestamp":ts})
        except Exception as e_main: err_msg=f"Unexpected Error: {e_main}"; st.error(err_msg); traceback.print_exc(); st.session_state.messages.append({"role":"assistant","content":err_msg,"response_type":"error","timestamp":ts})


# --- Global CSS for Modal Styling (Unchanged) ---
st.markdown(
    """
    <style>
    .modal-box { padding: 25px; background-color: #2E2E38; border-radius: 10px; border: 1px solid #4a4a58; box-shadow: 0 5px 15px rgba(0,0,0,0.3); margin-top: 15px; margin-bottom: 15px; }
    .modal-box h3, .modal-box h4, .modal-box p, .modal-box small, .modal-box label, .modal-box div[data-testid="stCaptionContainer"], .modal-box li, .modal-box strong, .modal-box .stCheckbox > label > div { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True
)