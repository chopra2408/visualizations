import streamlit as st
import sys
import os
st.set_page_config(layout="wide", page_title="Conversational Data Agent")

frontend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(frontend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CRITICAL_ERROR_OCCURRED = False
CRITICAL_ERROR_MESSAGE = ""
UTILS_AVAILABLE = False

try:
    from backend.models import PlotConfig
    from backend.utils import generate_plot_from_config
    UTILS_AVAILABLE = True
except ImportError as e:
    CRITICAL_ERROR_OCCURRED = True
    import json # Added import json for dummy PlotConfig
    CRITICAL_ERROR_MESSAGE = (
        "FATAL ERROR: Could not import `PlotConfig` from `backend.models` or "
        "`generate_plot_from_config` from `backend.utils`.\n"
        "Ensure `models.py` and `utils.py` are in the 'backend' directory, "
        "'backend' has an `__init__.py` (can be empty),\n"
        "and 'backend' is at the same level as 'frontend'.\n"
        f"Details: {e}"
    )
    class PlotConfig: # Dummy
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump_json(self): return json.dumps(self.__dict__)
        @classmethod
        def model_validate_json(cls, json_str): return cls(**json.loads(json_str))
    def generate_plot_from_config(df, config): # Dummy
        print("WARNING: Using dummy generate_plot_from_config.")
        return None
import requests
import traceback
import json # This was already here, good.
from datetime import datetime
from dotenv import load_dotenv
import plotly.io as plotly_io
import streamlit.components.v1 as components
import pandas as pd
import io

# --- Environment Variable Loading ---
dotenv_path_streamlit = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path_streamlit):
    dotenv_path_streamlit = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path_streamlit)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

if not BACKEND_BASE_URL and not CRITICAL_ERROR_OCCURRED:
    CRITICAL_ERROR_OCCURRED = True
    CRITICAL_ERROR_MESSAGE = "FATAL ERROR: BACKEND_BASE_URL environment variable not set."

# --- NOW, AFTER st.set_page_config(), we can use other Streamlit commands ---
if CRITICAL_ERROR_OCCURRED:
    st.error(CRITICAL_ERROR_MESSAGE)
    st.stop()

# --- URLs ---
UPLOAD_DATA_URL = f"{BACKEND_BASE_URL.rstrip('/')}/uploadfile/"
PROCESS_QUERY_URL = f"{BACKEND_BASE_URL.rstrip('/')}/process_query/"

# --- Initialize session state variables ---
default_session_vars = {
    "messages": [], "current_session_id": None, "df_columns": [], "df_head": "",
    "current_data_filename": "", "plot_key_counter": 0, "user_just_submitted": False,
    # Screencast related states
    "show_recorder_ui_modal": False,
    "show_next_steps_screencast_modal": False,
    "screencast_local_blob_url_for_preview": None,
    "screencast_final_filename_from_js": "screen-recording.webm",
    "audio_enabled_for_js_recorder": True,
    "recorder_component_value": None, # For data returned by JS component
    "current_max_recording_time": None,
    "selected_target_format_for_js": "webm",
    "popover_selected_format_radio": "webm", # Tracks radio button in popover
    "recorder_component_value_processed_id": None, # To avoid re-processing component value
    # Plot interaction states
    "current_view_mode": "faceted",
    "enlarged_facet_config_json": None,
    "enlarged_facet_column_name": None,
    "enlarged_facet_value_selected": None,
    "uploaded_dataframe": None
}
for key, value in default_session_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Main App UI ---
# Title and Screencast Button Area
main_header_cols = st.columns([0.85, 0.15])
with main_header_cols[0]:
    st.title("📊 Conversational Data Analysis Agent")

with main_header_cols[1]:
    if st.session_state.current_session_id:
        with st.popover("🎬 Record Screencast", use_container_width=True):
            st.markdown("**1. Select Output Format:**")
            st.session_state.popover_selected_format_radio = st.radio(
                "Format:", options=["webm", "mp4", "mkv"],
                format_func=lambda x: {"webm": "WebM", "mp4": "MP4", "mkv": "MKV"}.get(x, x.upper()),
                key="popover_format_radio_v8", horizontal=True,
                index=["webm", "mp4", "mkv"].index(st.session_state.get("popover_selected_format_radio", "webm"))
            )
            st.markdown("**2. Select Duration:**")
            def _set_and_open_recorder_modal(max_time=None):
                st.session_state.selected_target_format_for_js = st.session_state.popover_selected_format_radio
                st.session_state.current_max_recording_time = max_time
                st.session_state.show_recorder_ui_modal = True
                st.session_state.show_next_steps_screencast_modal = False # Reset this
                st.session_state.screencast_local_blob_url_for_preview = None # Reset preview
            
            rec_cols = st.columns(2)
            rec_cols[0].button("No Limit", key="rec_nolimit_v8", on_click=_set_and_open_recorder_modal, args=(None,), use_container_width=True)
            rec_cols[0].button("30 Sec", key="rec_30s_v8", on_click=_set_and_open_recorder_modal, args=(30,), use_container_width=True)
            rec_cols[1].button("15 Sec", key="rec_15s_v8", on_click=_set_and_open_recorder_modal, args=(15,), use_container_width=True)
            rec_cols[1].button("1 Min", key="rec_1m_v8", on_click=_set_and_open_recorder_modal, args=(60,), use_container_width=True)
    elif not st.session_state.current_session_id:
        st.caption("Upload data to enable screencast.")


# --- MODAL 1: Record a Screencast UI ---
if st.session_state.show_recorder_ui_modal and st.session_state.current_session_id:
    with st.container():
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True) # Your custom modal class

        modal1_cols = st.columns([0.9, 0.1])
        with modal1_cols[0]:
            st.subheader("⏺️ Record a Screencast")
        with modal1_cols[1]:
            if st.button("✖", key="close_recorder_modal_btn_v9", help="Close Recorder"):
                st.session_state.show_recorder_ui_modal = False
                # Potentially send a 'cancel' event to JS if recorder is active
                st.rerun()
        
        st.write("Please grant screen sharing permissions when prompted. A 3-second countdown will begin after you select a screen/window to share.")
        
        st.session_state.audio_enabled_for_js_recorder = st.checkbox(
            "🎤 Include audio from microphone",
            value=st.session_state.audio_enabled_for_js_recorder,
            key="audio_checkbox_v9"
        )
        
        # Display chosen format and time limit for user confirmation
        format_display = st.session_state.selected_target_format_for_js.upper()
        time_limit_s = st.session_state.current_max_recording_time
        time_limit_display = f"{time_limit_s}s" if time_limit_s else "No limit"
        st.caption(f"Target Format: **{format_display}** | Recording Time Limit: **{time_limit_display}**")
        st.caption("The recording will be downloaded directly to your computer by your browser after you stop it or the time limit is reached.")

        # Prepare parameters for the JavaScript component
        js_component_params = {
            "audio": st.session_state.audio_enabled_for_js_recorder,
            "timeLimit": time_limit_s,  # in seconds, or null for no limit
            "delay": 3,  # countdown delay in seconds
            "format": st.session_state.selected_target_format_for_js, # "webm", "mp4", "mkv"
        }

        # Create a unique key for the component instance to ensure it re-renders if params change
        # This key is not currently used in components.html() but kept for reference or future use
        component_instance_key = (
            f"screen_recorder_comp_{st.session_state.current_session_id}_"
            f"{js_component_params['format']}_{js_component_params['timeLimit'] or 'nolimit'}_v9"
        )
        
        html_recorder_component = f"""
        <div id="recorderInterfaceRoot" style="font-family: sans-serif; padding: 15px; border: 1px solid #444; border-radius: 8px; background-color: #383840; color: white;">
            <button id="startRecordButton" style="padding: 10px 15px; margin-right: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">Start Recording</button>
            <button id="stopRecordButton" style="padding: 10px 15px; background-color: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer;" disabled>Stop Recording</button>
            <p id="countdownDisplay" style="font-size: 1.8em; color: #FFA500; margin-top:12px; min-height:1.5em; font-weight: bold;"></p>
            <p id="recorderStatus" style="margin-top: 8px; font-size: 0.95em;">Status: Idle</p>
            <p id="actionInfo" style="font-size: 0.9em; min-height:1.3em; color: #90EE90;"></p>
        </div>

        <script type="module">
            // IMPORTANT: PASTE YOUR FULL ScreenRecorder CLASS DEFINITION HERE
            // This is a highly condensed placeholder.
            class ScreenRecorder {{
                constructor() {{
                    this.mediaStream = null; this.mediaRecorder = null; this.recordedBlobs = [];
                    this.timerInterval = null; this.countdownInterval = null;
                    this.onRecordingStart = null; this.onRecordingStop = null;
                    this.onCountdown = null; this.onError = null; this.onStatusUpdate = null;
                    console.log("[Recorder] Instance created.");
                }}

                _getMimeType(requestedFormat = 'webm') {{
                    const format = requestedFormat.toLowerCase();
                    let preferredType = null;
                    const types = {{
                        webm: ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'],
                        mp4: ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=h264', 'video/mp4'], // Browser support for MP4 recording varies
                        mkv: ['video/x-matroska;codecs=avc1', 'video/x-matroska'] // Browser support for MKV recording varies
                    }};
                    const checkTypes = types[format] || types.webm; // Default to webm
                    for (const type of checkTypes) {{
                        if (MediaRecorder.isTypeSupported(type)) {{
                            preferredType = type;
                            break;
                        }}
                    }}
                    if (!preferredType) {{ // Broader fallback if specific codecs not supported
                        if (format === 'mp4' && MediaRecorder.isTypeSupported('video/mp4')) preferredType = 'video/mp4';
                        else if (MediaRecorder.isTypeSupported('video/webm')) preferredType = 'video/webm';
                        else {{ console.error("No suitable MIME type found for MediaRecorder."); return {{ mimeType: null, fileExtension: 'bin' }}; }}
                    }}
                    let fileExtension = format; // Default to requested
                    if (preferredType.includes('mp4')) fileExtension = 'mp4';
                    else if (preferredType.includes('x-matroska')) fileExtension = 'mkv';
                    else if (preferredType.includes('webm')) fileExtension = 'webm';
                    
                    console.log(`[Recorder] Requested format: ${{format}}, Effective MIME: ${{preferredType}}, File extension: ${{fileExtension}}`);
                    return {{ mimeType: preferredType, fileExtension: fileExtension }};
                }}

                async startRecording(options = {{}}) {{
                    const {{ timeLimit = null, delay = 3, audio = true, format = 'webm' }} = options;
                    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {{
                        const err = new Error("Recording already in progress.");
                        if (this.onError) this.onError(err); return Promise.reject(err);
                    }}
                    this.recordedBlobs = [];
                    if(this.onStatusUpdate) this.onStatusUpdate("Requesting screen sharing permission...");

                    try {{
                        this.mediaStream = await navigator.mediaDevices.getDisplayMedia({{ video: true, audio: audio }});
                        this.mediaStream.getVideoTracks()[0].onended = () => {{ // User stopped sharing via browser UI
                            if(this.onStatusUpdate) this.onStatusUpdate("Screen sharing stopped by user.");
                            this.stopRecording(false); // Don't try to stop mediaRecorder again if it's already stopping
                        }};
                        
                        if (this.onCountdown) this.onCountdown(delay);
                        let countdown = delay;
                        return new Promise((resolve, reject) => {{
                            this.countdownInterval = setInterval(() => {{
                                countdown--;
                                if (this.onCountdown) this.onCountdown(countdown);
                                if (countdown <= 0) {{
                                    clearInterval(this.countdownInterval); this.countdownInterval = null;
                                    try {{
                                        this._initiateRecording(timeLimit, format); // Pass format here
                                        resolve();
                                    }} catch (initError) {{
                                        if (this.onError) this.onError(initError);
                                        this._cleanup(); reject(initError);
                                    }}
                                }}
                            }}, 1000);
                        }});
                    }} catch (err) {{
                        if(this.onStatusUpdate) this.onStatusUpdate(`Permission denied or error: ${{err.message}}`);
                        if (this.onError) this.onError(err);
                        this._cleanup(); return Promise.reject(err);
                    }}
                }}

                _initiateRecording(timeLimit, requestedFormat) {{
                    const {{ mimeType, fileExtension }} = this._getMimeType(requestedFormat);
                    if (!mimeType) {{
                        const err = new Error("MIME type selection failed. Cannot start MediaRecorder.");
                        if (this.onError) this.onError(err); this._cleanup(); throw err;
                    }}
                    try {{
                        this.mediaRecorder = new MediaRecorder(this.mediaStream, {{ mimeType }});
                    }} catch (e) {{
                        const err = new Error(`Failed to create MediaRecorder: ${{e.message}} (MIME: ${{mimeType}})`);
                        if (this.onError) this.onError(err); this._cleanup(); throw err;
                    }}

                    this.mediaRecorder.onstop = () => {{
                        if (this.timerInterval) {{ clearTimeout(this.timerInterval); this.timerInterval = null; }}
                        const superBuffer = new Blob(this.recordedBlobs, {{ type: this.mediaRecorder.mimeType }});
                        const finalFileName = `screen-recording-${{new Date().toISOString().replace(/[:.]/g, '-')}}.${{fileExtension}}`;
                        console.log(`[Recorder] Recording stopped. Blob type: ${{superBuffer.type}}, size: ${{superBuffer.size}}, Filename: ${{finalFileName}}`);
                        if (this.onRecordingStop) this.onRecordingStop(superBuffer, finalFileName, false); // false means not cancelled
                        this._cleanup();
                    }};
                    this.mediaRecorder.ondataavailable = (event) => {{
                        if (event.data && event.data.size > 0) {{ this.recordedBlobs.push(event.data); }}
                    }};
                    this.mediaRecorder.start();
                    if(this.onStatusUpdate) this.onStatusUpdate(`Recording started (Format: ${{fileExtension.toUpperCase()}}, MIME: ${{this.mediaRecorder.mimeType}}).`);
                    if (this.onRecordingStart) this.onRecordingStart();

                    if (timeLimit && timeLimit > 0) {{
                        this.timerInterval = setTimeout(() => {{
                            if (this.mediaRecorder && this.mediaRecorder.state === "recording") {{
                                this.stopRecording();
                            }}
                        }}, timeLimit * 1000);
                    }}
                }}

                stopRecording(stopMediaRecorderInstance = true) {{
                    if (this.countdownInterval) {{ // If countdown is active, cancel it
                        clearInterval(this.countdownInterval); this.countdownInterval = null;
                        if (this.onRecordingStop) this.onRecordingStop(null, null, true); // true means cancelled
                        this._cleanup(); return;
                    }}
                    if (this.mediaRecorder && this.mediaRecorder.state === "recording" && stopMediaRecorderInstance) {{
                        this.mediaRecorder.stop(); // This will trigger onstop event
                    }} else if (this.mediaRecorder && this.mediaRecorder.state !== "inactive" && !stopMediaRecorderInstance) {{
                        // If user stopped sharing, mediaRecorder might be 'paused' or still 'recording' but stream ended
                        if (this.mediaRecorder.state === "recording") this.mediaRecorder.stop();
                        else this._cleanup(); // If already paused/inactive, just cleanup
                    }} else {{
                        this._cleanup(); // If no mediaRecorder or not recording
                    }}
                    if (this.timerInterval) {{ clearTimeout(this.timerInterval); this.timerInterval = null; }}
                }}

                _cleanup() {{
                    if (this.mediaStream) {{
                        this.mediaStream.getTracks().forEach(track => track.stop());
                        this.mediaStream = null;
                    }}
                    this.mediaRecorder = null; // Already handled by onstop, but good for explicit cancel
                    this.recordedBlobs = [];
                    if (this.timerInterval) clearTimeout(this.timerInterval); this.timerInterval = null;
                    if (this.countdownInterval) clearInterval(this.countdownInterval); this.countdownInterval = null;
                    if(this.onStatusUpdate) this.onStatusUpdate("Idle. Resources cleaned up.");
                    console.log("[Recorder] Resources cleaned up.");
                }}

                static download(blob, filename) {{
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    document.body.appendChild(a);
                    a.style.display = "none";
                    a.href = url;
                    a.download = filename;
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    console.log(`[Recorder] Download triggered for: ${{filename}}`);
                }}
            }}
            // --- End of ScreenRecorder Class ---

            const params = {json.dumps(js_component_params)};
            const startBtn = document.getElementById('startRecordButton');
            const stopBtn = document.getElementById('stopRecordButton');
            const statusEl = document.getElementById('recorderStatus');
            const countdownEl = document.getElementById('countdownDisplay');
            const actionEl = document.getElementById('actionInfo');

            // Debounce function to avoid overwhelming Streamlit with rapid updates
            function debounce(func, wait) {{
                let timeout;
                return function executedFunction(...args) {{
                    const later = () => {{ clearTimeout(timeout); func(...args); }};
                    clearTimeout(timeout); timeout = setTimeout(later, wait);
                }};
            }}
            const sendValueToStreamlit = debounce((value) => {{
                if (window.Streamlit) {{
                    window.Streamlit.setComponentValue(value);
                }} else {{
                    console.warn("Streamlit object not found. Cannot send component value.");
                }}
            }}, 300); // Send updates at most every 300ms


            const recorder = new ScreenRecorder();

            recorder.onStatusUpdate = (message) => {{ if (statusEl) statusEl.textContent = `Status: ${{message}}`; }};
            recorder.onCountdown = (count) => {{
                if (countdownEl) {{
                    if (count > 0) countdownEl.textContent = `Recording in ${{count}}...`;
                    else if (count === 0) countdownEl.textContent = 'REC 🔴';
                    else countdownEl.textContent = ''; // Clear after REC
                }}
            }};
            recorder.onRecordingStart = () => {{
                if (statusEl) statusEl.textContent = "Status: Recording...";
                if (countdownEl) countdownEl.textContent = "REC 🔴";
                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                if (actionEl) actionEl.textContent = '';
            }};
            recorder.onError = (error) => {{
                if (statusEl) {{ statusEl.textContent = `Error: ${{error.message}}`; statusEl.style.color="red"; }}
                if (countdownEl) countdownEl.textContent = '';
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                if (actionEl) actionEl.textContent = `Failed: ${{error.message}}`;
            }};
            recorder.onRecordingStop = async (blob, filename, cancelled) => {{
                if (countdownEl) countdownEl.textContent = '';
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;

                if (cancelled) {{
                    if (statusEl) statusEl.textContent = "Status: Recording cancelled.";
                    if (actionEl) actionEl.textContent = "Operation cancelled by user or system.";
                    sendValueToStreamlit({{ type: "recordingCancelled" }});
                    return;
                }}
                if (!blob || !filename) {{
                    if (statusEl) statusEl.textContent = "Status: Stopped, but no data was captured.";
                    if (actionEl) actionEl.textContent = "No video data recorded.";
                    sendValueToStreamlit({{ type: "noDataCaptured" }});
                    return;
                }}

                // All recordings are now direct download
                ScreenRecorder.download(blob, filename);
                let formatGenerated = params.format.toUpperCase();
                if (filename.includes('.')) {{ // Try to get actual extension from generated filename
                    formatGenerated = filename.split('.').pop().toUpperCase();
                }}
                if (actionEl) {{
                    actionEl.textContent = `${{formatGenerated}} recording '${{filename}}' download initiated.`;
                    actionEl.style.color = "lightgreen"; // Use a success color
                }}
                
                // Send completion info back to Streamlit
                sendValueToStreamlit({{
                    type: "directDownloadComplete",
                    filename: filename,
                    blobUrl: URL.createObjectURL(blob), // For potential preview in Streamlit Modal 2
                    generatedFormat: params.format // The originally requested format
                }});
                if (statusEl) statusEl.textContent = `Status: ${{formatGenerated}} downloaded. You can close this or record again.`;
            }};

            startBtn.onclick = async () => {{
                if(statusEl) statusEl.style.color="white"; // Reset error color
                if(actionEl) actionEl.textContent = '';
                if(countdownEl) countdownEl.textContent = '';
                recorder.startRecording({{ 
                    timeLimit: params.timeLimit, 
                    delay: params.delay, 
                    audio: params.audio,
                    format: params.format
                }}).catch(err => {{
                    // onError callback already handles UI for this
                    console.error("Start recording promise rejected:", err);
                }});
            }};
            stopBtn.onclick = () => {{
                recorder.stopRecording();
            }};

            // Initial status message
            if (statusEl) {{
                let initialMsg = "Status: Idle.";
                if (params.timeLimit) initialMsg += ` (Time Limit: ${{params.timeLimit}}s)`;
                else initialMsg += " (No time limit)";
                initialMsg += ` (Target: ${{params.format.toUpperCase()}})`;
                statusEl.textContent = initialMsg;
            }}
        </script>
        """
        
        try:
            # The height might need adjustment based on your recorder UI's content
            # No explicit key is used here; Streamlit implicitly keys based on arguments.
            # If html_recorder_component (which includes js_component_params) changes,
            # it's a new component instance.
            component_value = components.html(html_recorder_component, height=280)

            # Process the value returned by the JavaScript component
            # This block executes if component_value is truthy (not None) AND
            # its object ID is different from the last processed one.
            if component_value and (st.session_state.recorder_component_value_processed_id != id(component_value)):
                st.session_state.recorder_component_value_processed_id = id(component_value)
                st.session_state.recorder_component_value = component_value # Store the raw value
                
                # Enhanced logging to understand what component_value is, especially if it's not a dict
                print(f"Streamlit received from JS: {component_value}, type: {type(component_value)}") 

                # CRITICAL FIX: Check if component_value is a dictionary before calling .get()
                if isinstance(component_value, dict):
                    event_type = component_value.get("type") 
                    if event_type == "directDownloadComplete":
                        st.session_state.screencast_final_filename_from_js = component_value.get("filename", "recording.file")
                        st.session_state.screencast_local_blob_url_for_preview = component_value.get("blobUrl")
                        # Transition to Modal 2
                        st.session_state.show_recorder_ui_modal = False
                        st.session_state.show_next_steps_screencast_modal = True
                        st.rerun()
                    elif event_type == "recordingCancelled" or event_type == "noDataCaptured":
                        # Modal 1 stays open, user sees status message from JS, can try again or close.
                        pass 
                    # Add handling for other event_types if your JS sends them (e.g., specific errors)
                else:
                    # This branch is hit if component_value is truthy, its id is new, but it's NOT a dictionary.
                    # This is where a DeltaGenerator would land if the error message is literal,
                    # or any other non-dict type sent from JS (though JS is coded to send objects/dicts).
                    pass
                    # No st.rerun() here, as the expected state transition relies on dict processing.
                    # The UI will update due to Streamlit's natural flow, showing the warning.

        except Exception as e_comp:
            st.error(f"Error with recorder component: {e_comp}")
            traceback.print_exc()

        st.markdown("</div>", unsafe_allow_html=True)

# --- MODAL 2: Next Steps After Screencast ---
if st.session_state.show_next_steps_screencast_modal:
    with st.container(): # Use st.dialog for newer Streamlit versions if preferred
        st.markdown("<div class='modal-box'>", unsafe_allow_html=True)
        modal2_cols = st.columns([0.9, 0.1])
        with modal2_cols[0]: st.subheader("🎉 Recording Downloaded!")
        with modal2_cols[1]:
            if st.button("✖", key="close_next_steps_modal_btn_v8", help="Close"):
                st.session_state.show_next_steps_screencast_modal = False
                st.session_state.screencast_local_blob_url_for_preview = None # Clear preview URL
                st.rerun()
        st.markdown("---")
        final_filename = st.session_state.screencast_final_filename_from_js
        st.success(f"Your recording (`{final_filename}`) should have started downloading directly to your computer.")
        
        if st.session_state.screencast_local_blob_url_for_preview:
            st.markdown("#### Preview (if browser supports this format):")
            try:
                # Determine mime type for preview
                mime_type = "application/octet-stream"
                if final_filename.lower().endswith(".webm"): mime_type = "video/webm"
                elif final_filename.lower().endswith(".mp4"): mime_type = "video/mp4"
                elif final_filename.lower().endswith(".mkv"): mime_type = "video/x-matroska"
                st.video(st.session_state.screencast_local_blob_url_for_preview, format=mime_type)
            except Exception as e_vid_preview:
                st.warning(f"Could not display video preview: {e_vid_preview}")
        else:
            st.info("No local preview available for this recording.")
        st.markdown(f"Please check your browser's default download location for `{final_filename}`.")
        st.markdown("</div>", unsafe_allow_html=True)


# --- Sidebar for Upload ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="uploader_main_v8_sidebar")
    if uploaded_file:
        # Using a button instead of form submit for simpler state management here
        if st.button("Upload and Start New Session", key="upload_button_sidebar_v8", use_container_width=True):
            with st.spinner("Processing file..."):
                files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(UPLOAD_DATA_URL, files=files_payload, timeout=120)
                    response.raise_for_status()
                    file_info = response.json()

                    # Full reset of session state for a new session
                    for k_default in default_session_vars.keys():
                        st.session_state[k_default] = default_session_vars[k_default]
                    st.session_state.messages = [] # Ensure messages are explicitly cleared

                    st.session_state.current_session_id = file_info["session_id"]
                    st.session_state.current_data_filename = file_info["filename"]
                    st.session_state.df_columns = file_info["columns"]
                    st.session_state.df_head = file_info["df_head"]

                    uploaded_file.seek(0)
                    df_bytes = uploaded_file.getvalue()
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.uploaded_dataframe = pd.read_csv(io.BytesIO(df_bytes))
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        st.session_state.uploaded_dataframe = pd.read_excel(io.BytesIO(df_bytes))
                    
                    st.success(f"File '{file_info['filename']}' processed! Session: ...{file_info['session_id'][-6:]}")
                    st.rerun() 
                except Exception as e_upload:
                    st.error(f"File upload failed: {e_upload}"); traceback.print_exc()

    if st.session_state.current_session_id:
        st.markdown("---")
        st.success(f"Active Session: `{st.session_state.current_session_id.split('-')[0]}`") # Show only part of ID
        st.info(f"File: **{st.session_state.current_data_filename}**")
        with st.expander("Data Preview", expanded=False):
            st.text_area("Columns:", ", ".join(st.session_state.df_columns), height=80, disabled=True, key="cols_preview_sidebar_v8")
            st.text_area("Data Head:", st.session_state.df_head, height=150, disabled=True, key="head_preview_sidebar_v8")
    else:
        st.info("Upload a data file to begin analysis.")

# --- Main Chat and Plot Display Area ---
st.header("2. Chat with Your Data")

def display_plot_with_interaction(fig_json_str, plot_config_json_str, plot_insights_str, message_key_idx):
    if not (fig_json_str and plot_config_json_str and UTILS_AVAILABLE):
        if fig_json_str and not UTILS_AVAILABLE: st.warning("Plotting utilities missing.")
        return

    try:
        plot_obj = plotly_io.from_json(fig_json_str)
        current_plot_config = PlotConfig.model_validate_json(plot_config_json_str)

        st.plotly_chart(plot_obj, use_container_width=True, key=f"plot_main_{message_key_idx}")

        if plot_insights_str:
            with st.expander("🔍 Plot Insights", expanded=False): st.markdown(plot_insights_str)
        with st.expander("⚙️ Plot Configuration", expanded=False): st.json(json.loads(plot_config_json_str))

        if current_plot_config.facet_column and \
           st.session_state.uploaded_dataframe is not None and \
           current_plot_config.facet_column in st.session_state.uploaded_dataframe.columns:
            
            facet_col_name_local = current_plot_config.facet_column
            unique_facet_values = sorted(list(st.session_state.uploaded_dataframe[facet_col_name_local].astype(str).unique()))
            
            if unique_facet_values:
                options_for_select = ["<Select facet to enlarge>"] + unique_facet_values
                selectbox_key = f"facet_select_{message_key_idx}"
                
                current_selection_index = 0
                if st.session_state.current_view_mode == "enlarged" and \
                   st.session_state.enlarged_facet_column_name == facet_col_name_local and \
                   st.session_state.enlarged_facet_config_json == plot_config_json_str and \
                   st.session_state.enlarged_facet_value_selected in options_for_select:
                    try: current_selection_index = options_for_select.index(st.session_state.enlarged_facet_value_selected)
                    except ValueError: pass
                
                def selectbox_on_change_callback(): # Renamed for clarity
                    selected_value = st.session_state[selectbox_key]
                    if selected_value != "<Select facet to enlarge>":
                        st.session_state.current_view_mode = "enlarged"
                        st.session_state.enlarged_facet_value_selected = selected_value
                        st.session_state.enlarged_facet_column_name = facet_col_name_local
                        st.session_state.enlarged_facet_config_json = plot_config_json_str
                    elif st.session_state.current_view_mode == "enlarged" and \
                         st.session_state.enlarged_facet_config_json == plot_config_json_str:
                        st.session_state.current_view_mode = "faceted"
                        st.session_state.enlarged_facet_value_selected = None

                st.selectbox(f"🔍 Enlarge {facet_col_name_local}:", options_for_select, index=current_selection_index, 
                             key=selectbox_key, on_change=selectbox_on_change_callback)
    except Exception as e_disp_plot:
        st.error(f"Error in display_plot_with_interaction for msg {message_key_idx}: {e_disp_plot}"); traceback.print_exc()


# Main display logic based on view_mode
if st.session_state.current_view_mode == "enlarged" and \
   st.session_state.enlarged_facet_value_selected and \
   st.session_state.enlarged_facet_config_json and \
   st.session_state.uploaded_dataframe is not None and \
   UTILS_AVAILABLE:

    st.subheader(f"Enlarged View: {st.session_state.enlarged_facet_column_name} = {st.session_state.enlarged_facet_value_selected}")

    if st.button("⬅️ Back to Full Chat & Faceted View", key="back_button_enlarged_v8"):
        st.session_state.current_view_mode = "faceted"
        st.session_state.enlarged_facet_value_selected = None # Reset this to make selectbox default
        st.rerun()

    try:
        full_df = st.session_state.uploaded_dataframe
        base_config_obj = PlotConfig.model_validate_json(st.session_state.enlarged_facet_config_json)
        facet_col = st.session_state.enlarged_facet_column_name
        selected_val = st.session_state.enlarged_facet_value_selected

        if facet_col not in full_df.columns: st.error(f"Facet column '{facet_col}' no longer in DataFrame."); st.stop()
        
        filtered_df = full_df[full_df[facet_col].astype(str) == str(selected_val)]

        if not filtered_df.empty:
            new_color_by = base_config_obj.color_by_column
            if base_config_obj.color_by_column == facet_col: new_color_by = None

            enlarged_plot_config = PlotConfig(
                plot_type=base_config_obj.plot_type, x_column=base_config_obj.x_column,
                y_column=base_config_obj.y_column, color_by_column=new_color_by,
                title=f"{selected_val}: {base_config_obj.y_column or 'Data'} vs {base_config_obj.x_column or 'Index'}",
                xlabel=base_config_obj.xlabel or base_config_obj.x_column,
                ylabel=base_config_obj.ylabel or base_config_obj.y_column,
                facet_column=None, facet_row=None,
                bins=base_config_obj.bins, bar_style=base_config_obj.bar_style,
                top_n_categories=None, aggregate_by_color_col_method=None, limit_categories_auto=False
            )
            enlarged_fig_json_str = generate_plot_from_config(filtered_df, enlarged_plot_config)

            if enlarged_fig_json_str:
                plot_key = f"enlarged_plot_main_{hash(str(selected_val) + enlarged_plot_config.plot_type + (enlarged_plot_config.x_column or '') + (enlarged_plot_config.y_column or ''))}"
                st.plotly_chart(plotly_io.from_json(enlarged_fig_json_str), use_container_width=True, key=plot_key)
                with st.expander("⚙️ Enlarged Plot Configuration", expanded=False):
                    st.json(json.loads(enlarged_plot_config.model_dump_json()))
            else: st.error(f"Could not generate enlarged plot for '{selected_val}'.")
        else: st.warning(f"No data for '{facet_col} = {selected_val}'.")
    except Exception as e_render_enlarged:
        st.error(f"Error rendering enlarged plot: {e_render_enlarged}"); traceback.print_exc()
else: # Display chat history (faceted view or non-plot messages)
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("pre_summary_content"): st.markdown(f">{msg['pre_summary_content'].strip()}"); st.markdown("---")
            st.markdown(msg.get("content", "").strip())

            if msg["role"] == "assistant" and msg.get("plotly_fig_json") and msg.get("plot_config_json") and UTILS_AVAILABLE:
                display_plot_with_interaction(
                    msg.get("plotly_fig_json"), msg.get("plot_config_json"),
                    msg.get("plot_insights"), f"hist_msg_{i}"
                )
            elif msg.get("plotly_fig_json") and not UTILS_AVAILABLE:
                try: st.plotly_chart(plotly_io.from_json(msg["plotly_fig_json"]),use_container_width=True,key=f"fallback_plot_hist_render_{i}")
                except: st.warning("Fallback: Could not display plot from history.")
            
            if msg["role"] == "assistant" and (msg.get("thinking_log_str") or msg.get("error")):
                with st.expander("⚙️ Agent Details", expanded=False):
                    if msg.get("thinking_log_str"): st.text_area("Log:", msg["thinking_log_str"], height=150, disabled=True, key=f"log_text_hist_render_{i}")
                    if msg.get("response_type"): st.caption(f"Action: `{msg['response_type']}`")
                    if msg.get("error"): st.error(f"Error: {msg['error']}")

# --- Chat Input ---
if prompt := st.chat_input("Ask about your data...", key="chat_input_main_v8", disabled=not st.session_state.current_session_id):
    if st.session_state.current_view_mode == "enlarged": # If user types new query, reset view
        st.session_state.current_view_mode = "faceted"
        st.session_state.enlarged_facet_value_selected = None 
        st.session_state.enlarged_facet_config_json = None
        st.session_state.enlarged_facet_column_name = None

    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
    st.session_state.user_just_submitted = True
    st.rerun()

if st.session_state.user_just_submitted:
    st.session_state.user_just_submitted = False
    last_user_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "user" and not m.get("assistant_processed")), None)

    if last_user_msg:
        last_user_msg["assistant_processed"] = True
        current_prompt, current_ts = last_user_msg["content"], last_user_msg["timestamp"]

        with st.chat_message("assistant"):
            thinking_ph, content_ph, final_output_cont = st.empty(), st.empty(), st.container()
            assistant_entry = {"role": "assistant", "timestamp": current_ts}
            payload = {"session_id": st.session_state.current_session_id, "query": current_prompt}
            streamed_thinking, streamed_content = "", ""
            try:
                with requests.post(PROCESS_QUERY_URL, json=payload, stream=True, timeout=360) as r:
                    r.raise_for_status()
                    for line_bytes in r.iter_lines():
                        if not line_bytes: continue
                        line, chunk = line_bytes.decode("utf-8"), None
                        try: chunk = json.loads(line)
                        except json.JSONDecodeError: print(f"JSON Decode Err: {line}"); continue
                        
                        if chunk.get("type") == "thinking_process_update":
                            streamed_thinking += chunk.get("chunk", ""); thinking_ph.markdown(">" + streamed_thinking.strip() + "▌")
                        elif chunk.get("type") == "content":
                            streamed_content += chunk.get("chunk", ""); content_ph.markdown(streamed_content.strip() + "▌")
                        elif chunk.get("type") == "final_agent_response":
                            thinking_ph.empty(); content_ph.empty()
                            agent_data = chunk.get("data", {})
                            assistant_entry.update(agent_data) # Populate with all data from agent
                            
                            if not assistant_entry.get("content") and streamed_content: assistant_entry["content"] = streamed_content
                            elif not assistant_entry.get("content"): assistant_entry["content"] = agent_data.get("content", "Processed.")
                            if not assistant_entry.get("pre_summary_content") and streamed_thinking: assistant_entry["pre_summary_content"] = streamed_thinking
                            
                            with final_output_cont:
                                st.markdown(assistant_entry["content"].strip()) # Display main text first
                                if agent_data.get("plotly_fig_json") and agent_data.get("plot_config_json") and UTILS_AVAILABLE:
                                    st.session_state.current_view_mode = "faceted" # New plot from agent, show in faceted
                                    st.session_state.enlarged_facet_value_selected = None
                                    
                                    plot_idx_for_key = len(st.session_state.messages) # Index before appending current
                                    display_plot_with_interaction(
                                        agent_data["plotly_fig_json"], agent_data["plot_config_json"],
                                        agent_data.get("plot_insights"), f"new_plot_stream_{plot_idx_for_key}"
                                    )
                                elif agent_data.get("plotly_fig_json") and not UTILS_AVAILABLE: # Fallback
                                    try: st.plotly_chart(plotly_io.from_json(agent_data["plotly_fig_json"]),use_container_width=True,key=f"fallback_plot_stream_{len(st.session_state.messages)}")
                                    except: st.warning("Stream: Could not display plot (fallback).")

                                if agent_data.get("thinking_log_str") or agent_data.get("error"):
                                    with st.expander("⚙️ Agent Details (Final)", expanded=False): # No key
                                        if agent_data.get("thinking_log_str"): st.text_area("Log:", agent_data["thinking_log_str"], height=150, disabled=True, key=f"final_log_text_stream_{len(st.session_state.messages)}")
                                        if agent_data.get("response_type"): st.caption(f"Action: `{agent_data['response_type']}`")
                                        if agent_data.get("error"): st.error(f"Error: {agent_data['error']}")
                            break 
                        elif chunk.get("type") == "error":
                            err_msg = chunk.get('chunk', 'Unknown stream error')
                            assistant_entry.update({"content":f"Stream Error: {err_msg}", "response_type":"error", "error":err_msg})
                            final_output_cont.error(f"Stream Error: {err_msg}"); break
                
                if not assistant_entry.get("content"): assistant_entry["content"] = "Processing complete."
                st.session_state.messages.append(assistant_entry)
                st.rerun()

            except requests.exceptions.RequestException as e:
                err_msg = f"Conn error: {e}"; st.error(err_msg)
                st.session_state.messages.append({"role":"assistant", "content":err_msg, "error":str(e), "response_type":"error", "timestamp":current_ts})
                st.rerun()
            except Exception as e:
                err_msg = f"Error processing: {e}"; st.error(err_msg); traceback.print_exc()
                st.session_state.messages.append({"role":"assistant", "content":err_msg, "error":str(e), "response_type":"error", "timestamp":current_ts})
                st.rerun()

# --- Global CSS for Modal Styling ---
st.markdown("""<style>
.modal-box { padding: 20px; background-color: #2E2E38; border-radius: 8px; border: 1px solid #4a4a58; margin-top:10px; margin-bottom:10px;}
.modal-box h3, .modal-box p, .modal_box label { color: #FFFFFF !important; }
/* You might need more specific selectors for Streamlit's internal elements if default theming is overridden */
</style>""", unsafe_allow_html=True)