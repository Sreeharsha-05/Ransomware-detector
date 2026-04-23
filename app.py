"""
SRDC-GPT Ransomware Detection System — Streamlit Web Application
"""

import os
import io
import csv
import json
from datetime import datetime

import streamlit as st
import torch
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SRDC-GPT Ransomware Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR          = os.path.join(os.path.dirname(__file__), 'models')
FAMILY_MODEL_PATH  = os.path.join(MODEL_DIR, 'family_model.pth')
ZERODAY_MODEL_PATH = os.path.join(MODEL_DIR, 'zeroday_model.pth')

# ── Sample input ──────────────────────────────────────────────────────────────
SAMPLE_INPUT = """\
API:CreateRemoteThread
API:VirtualAllocEx
API:WriteProcessMemory
API:OpenProcess
REG:OPENED:HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run
REG:WRITTEN:HKEY_CURRENT_USER\\Software\\Policies\\Explorer
FILES:READ:C:\\Users\\victim\\Documents\\important.docx
FILES:CREATED:C:\\Users\\victim\\Documents\\important.docx.locked
FILES:DELETED:C:\\Users\\victim\\Documents\\important.docx
DIR:ENUMERATED:C:\\Users\\victim\\Documents
DIR:ENUMERATED:C:\\Users\\victim\\Desktop
STR:ransom_note.txt
STR:.locked
STR:YOUR FILES HAVE BEEN ENCRYPTED"""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 700; color: #e74c3c; margin-bottom: 0;}
.sub-title  {font-size: 1.05rem; color: #666; margin-top: 0.2rem; margin-bottom: 1.5rem;}
.result-box {
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    margin-top: 1rem;
}
.result-malicious {background: #fde8e8; border-left: 5px solid #e74c3c;}
.result-benign    {background: #e8f8e8; border-left: 5px solid #27ae60;}
.result-family    {background: #e8f0fe; border-left: 5px solid #3498db;}
.confidence-label {font-size: 0.9rem; color: #555;}
.section-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tokenizer_cached():
    from preprocessing import load_tokenizer
    return load_tokenizer()


@st.cache_resource(show_spinner=False)
def load_family_model_cached():
    from model_loader import load_family_model
    return load_family_model(FAMILY_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_zeroday_model_cached():
    from model_loader import load_zeroday_model
    return load_zeroday_model(ZERODAY_MODEL_PATH)


def models_available() -> tuple[bool, bool]:
    return os.path.exists(FAMILY_MODEL_PATH), os.path.exists(ZERODAY_MODEL_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## Model Setup")

    fam_ok, zd_ok = models_available()

    st.sidebar.markdown("**Family model:**  " + ("✅ Loaded" if fam_ok else "❌ Missing"))
    st.sidebar.markdown("**Zero-day model:** " + ("✅ Loaded" if zd_ok else "❌ Missing"))

    if not fam_ok or not zd_ok:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Download from Google Drive")
        st.sidebar.markdown(
            "Enter your Google Drive **file IDs** for the exported model checkpoints. "
            "See `export_models_colab.py` in the repo for how to create them."
        )
        fam_id = st.sidebar.text_input(
            "family_model.pth — Drive file ID", key="fam_gdrive_id"
        )
        zd_id = st.sidebar.text_input(
            "zeroday_model.pth — Drive file ID", key="zd_gdrive_id"
        )
        if st.sidebar.button("Download Models"):
            from model_loader import download_from_gdrive
            if fam_id and not fam_ok:
                with st.sidebar.status("Downloading family model…"):
                    ok = download_from_gdrive(fam_id, FAMILY_MODEL_PATH)
                st.sidebar.success("Downloaded!" if ok else "Failed — check file ID.")
            if zd_id and not zd_ok:
                with st.sidebar.status("Downloading zero-day model…"):
                    ok = download_from_gdrive(zd_id, ZERODAY_MODEL_PATH)
                st.sidebar.success("Downloaded!" if ok else "Failed — check file ID.")
            if fam_id or zd_id:
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Upload Model Files")
        uploaded_fam = st.sidebar.file_uploader(
            "Upload family_model.pth", type=["pth"], key="upload_fam"
        )
        uploaded_zd = st.sidebar.file_uploader(
            "Upload zeroday_model.pth", type=["pth"], key="upload_zd"
        )
        if uploaded_fam:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(FAMILY_MODEL_PATH, 'wb') as f:
                f.write(uploaded_fam.read())
            st.sidebar.success("family_model.pth saved.")
            st.rerun()
        if uploaded_zd:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(ZERODAY_MODEL_PATH, 'wb') as f:
                f.write(uploaded_zd.read())
            st.sidebar.success("zeroday_model.pth saved.")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Device")
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    st.sidebar.info(f"Running on: **{device_name}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built on [SRDC-GPT](https://github.com) | "
        "GPT-2 fine-tuned on LATAP corpus"
    )


# ── Probability bar chart ─────────────────────────────────────────────────────
def prob_bar_chart(all_probs: dict, title: str, highlight_key: str = None):
    labels = list(all_probs.keys())
    values = [all_probs[k] * 100 for k in labels]

    colors = []
    for k in labels:
        if k == highlight_key:
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 105], title="Probability (%)"),
        height=max(300, len(labels) * 35 + 60),
        margin=dict(l=130, r=60, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Debug token display ────────────────────────────────────────────────────────
def render_debug(result: dict, tokenizer):
    with st.expander("Debug — Processed Tokens & Section Masks"):
        st.markdown("**Processed section texts (what the model receives):**")
        sep_colors = {
            '[API]': '#FF6B6B', '[DROP]': '#4ECDC4', '[REG]': '#45B7D1',
            '[FILES]': '#96CEB4', '[EXT]': '#FFEAA7', '[DIR]': '#DDA0DD',
            '[STR]': '#98FB98',
        }
        for text in result.get('section_texts', []):
            sep = text.split(' ')[0] if text else ''
            color = sep_colors.get(sep, '#ccc')
            st.markdown(
                f'<span class="section-chip" style="background:{color}20;'
                f'border:1px solid {color};">{sep}</span> '
                f'<code style="font-size:0.8rem">{text[:200]}</code>',
                unsafe_allow_html=True,
            )

        tokens = result.get('tokens', [])
        if tokens:
            st.markdown(f"**Token count (non-pad):** {sum(1 for t in tokens if t != tokenizer.pad_token_id)}")
            decoded = tokenizer.decode(tokens, skip_special_tokens=False)
            st.text_area("Full decoded sequence", decoded, height=120, key="decoded_seq")


# ── Prediction history ────────────────────────────────────────────────────────
def add_to_history(entry: dict):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(entry)


def render_history():
    history = st.session_state.get('history', [])
    if not history:
        st.info("No predictions yet — run a detection to see history here.")
        return

    st.markdown(f"**{len(history)} prediction(s)**")

    # Download buttons
    col_txt, col_csv = st.columns([1, 1])
    with col_txt:
        txt_content = "\n\n".join(
            f"[{h['time']}] {h['type']} → {h['label']} ({h['confidence']*100:.1f}%)"
            for h in history
        )
        st.download_button(
            "Download as TXT",
            data=txt_content,
            file_name="srdc_gpt_predictions.txt",
            mime="text/plain",
        )
    with col_csv:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=['time', 'type', 'label', 'confidence', 'is_malicious', 'input_preview'],
            extrasaction='ignore',
        )
        writer.writeheader()
        writer.writerows(history)
        st.download_button(
            "Download as CSV",
            data=buf.getvalue(),
            file_name="srdc_gpt_predictions.csv",
            mime="text/csv",
        )

    st.markdown("---")
    for h in reversed(history):
        icon = "🔴" if h.get('is_malicious', True) else "🟢"
        st.markdown(
            f"{icon} `{h['time']}` &nbsp; **{h['type']}** → "
            f"**{h['label']}** &nbsp; `{h['confidence']*100:.1f}%`"
        )


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Title
    st.markdown('<p class="main-title">SRDC-GPT Ransomware Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Dynamic behavioral analysis powered by GPT-2 fine-tuned '
        'on the LATAP corpus &mdash; ransomware family classification and zero-day detection</p>',
        unsafe_allow_html=True,
    )

    fam_ok, zd_ok = models_available()
    if not fam_ok and not zd_ok:
        st.warning(
            "No model files found in `models/`. "
            "Use the sidebar to upload or download `family_model.pth` and `zeroday_model.pth`. "
            "See `export_models_colab.py` in the repo for how to export from your Colab session."
        )

    # Main tabs
    tab_detect, tab_history, tab_about = st.tabs(["Detection", "Prediction History", "About"])

    # ── Detection tab ───────────────────────────────────────────────────────
    with tab_detect:
        col_input, col_results = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### Behavioral Sequence Input")
            st.markdown(
                "Paste dynamic behavioral events below — one event per line. "
                "Supported prefixes: `API:`, `REG:`, `FILES:`, `DIR:`, `DROP:`, `STR:`, `EXT:`"
            )

            if st.button("Use Sample Input", use_container_width=True):
                st.session_state.input_text = SAMPLE_INPUT

            user_input = st.text_area(
                label="Behavioral sequence",
                value=st.session_state.get('input_text', ''),
                height=320,
                placeholder=SAMPLE_INPUT,
                label_visibility="collapsed",
                key="input_text",
            )

            col_btn1, col_btn2 = st.columns(2)
            run_family  = col_btn1.button("Predict Family",        type="primary", use_container_width=True)
            run_zeroday = col_btn2.button("Detect Zero-Day Threat", type="primary", use_container_width=True)

        with col_results:
            st.markdown("### Results")

            if run_family:
                if not user_input.strip():
                    st.warning("Please enter a behavioral sequence.")
                elif not fam_ok:
                    st.error("family_model.pth not found. Upload it via the sidebar.")
                else:
                    from inference import predict_family
                    with st.spinner("Loading tokenizer and model…"):
                        tokenizer = load_tokenizer_cached()
                        model     = load_family_model_cached()
                    with st.spinner("Running family classification…"):
                        result = predict_family(model, tokenizer, user_input)

                    label      = result['label']
                    confidence = result['confidence']
                    is_goodware = (label == 'Goodware')

                    if is_goodware:
                        box_class  = "result-benign"
                        title_text = "Classified as Goodware"
                        title_color = "#27ae60"
                        emoji = "✅"
                    else:
                        box_class  = "result-family"
                        title_text = "Ransomware Family"
                        title_color = "#1a5276"
                        emoji = "🔴"

                    st.markdown(
                        f'<div class="result-box {box_class}">'
                        f'<h3 style="margin:0;color:{title_color}">{emoji} {title_text}</h3>'
                        f'<h2 style="margin:4px 0;color:{title_color}">{label}</h2>'
                        f'<p class="confidence-label">Confidence: <b>{confidence*100:.1f}%</b></p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    fig = prob_bar_chart(
                        result['all_probs'],
                        "Family Probability Distribution",
                        highlight_key=label,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    render_debug(result, tokenizer)

                    add_to_history({
                        'time'         : datetime.now().strftime("%H:%M:%S"),
                        'type'         : 'Family',
                        'label'        : label,
                        'confidence'   : confidence,
                        'is_malicious' : not is_goodware,
                        'input_preview': user_input[:80].replace('\n', ' '),
                    })

            elif run_zeroday:
                if not user_input.strip():
                    st.warning("Please enter a behavioral sequence.")
                elif not zd_ok:
                    st.error("zeroday_model.pth not found. Upload it via the sidebar.")
                else:
                    from inference import predict_zeroday
                    with st.spinner("Loading tokenizer and model…"):
                        tokenizer = load_tokenizer_cached()
                        model     = load_zeroday_model_cached()
                    with st.spinner("Running zero-day detection…"):
                        result = predict_zeroday(model, tokenizer, user_input)

                    label        = result['label']
                    is_malicious = result['is_malicious']
                    probability  = result['probability']

                    box_class  = "result-malicious" if is_malicious else "result-benign"
                    icon       = "MALICIOUS" if is_malicious else "BENIGN"
                    icon_color = "#e74c3c" if is_malicious else "#27ae60"
                    emoji      = "🚨" if is_malicious else "✅"

                    st.markdown(
                        f'<div class="result-box {box_class}">'
                        f'<h3 style="margin:0;color:{icon_color}">{emoji} {icon}</h3>'
                        f'<h2 style="margin:4px 0;color:{icon_color}">{label}</h2>'
                        f'<p class="confidence-label">'
                        f'P(ransomware): <b>{probability*100:.1f}%</b></p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    fig = prob_bar_chart(
                        result['all_probs'],
                        "Detection Probability",
                        highlight_key=label,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    render_debug(result, tokenizer)

                    add_to_history({
                        'time'         : datetime.now().strftime("%H:%M:%S"),
                        'type'         : 'Zero-Day',
                        'label'        : label,
                        'confidence'   : probability if is_malicious else 1 - probability,
                        'is_malicious' : is_malicious,
                        'input_preview': user_input[:80].replace('\n', ' '),
                    })

            else:
                st.markdown(
                    "<div style='color:#888;padding:2rem;text-align:center'>"
                    "Enter a behavioral sequence and click a detection button."
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── History tab ─────────────────────────────────────────────────────────
    with tab_history:
        st.markdown("### Prediction History")
        render_history()

    # ── About tab ───────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("### Architecture")
        st.markdown("""
**SRDC-GPT** is a ransomware detection system based on GPT-2 fine-tuned on the
LATAP (Language-Augmented Threat Action Phrases) corpus of behavioral sequences.

#### Input
Dynamic behavioral events extracted from sandboxed execution:
- **API**: Windows API calls (e.g. `API:CreateRemoteThread`)
- **REG**: Registry operations (e.g. `REG:OPENED:HKCU\\...`)
- **FILES**: File operations (e.g. `FILES:READ:C:\\...`)
- **DIR**: Directory enumeration (e.g. `DIR:ENUMERATED:C:\\...`)
- **DROP**: Dropped files, **STR**: Strings, **EXT**: File extensions

#### Preprocessing Pipeline
1. Parse events into 7 feature channels
2. Semantic conversion (CamelCase → phrases, VERB_MAP, FEATURE_TYPE_MAP)
3. Per-channel tokenization with budget of 72 tokens each
4. Concatenate all 7 channels → final 512-token sequence
5. Section masks (7 × 512) track which tokens belong to which channel

#### Models
| Model | Architecture | Task |
|---|---|---|
| Family Classifier | GPT-2 + SectionMeanPool + Linear(5376→11) | 11-class ransomware family |
| Zero-Day Detector | GPT-2 + SectionMeanPool + MLP(5376→512→2) | Binary: ransomware vs benign |

#### Performance
| Model | Metric | Score | Paper Target |
|---|---|---|---|
| Family (V5, 12-class) | Val Accuracy | 0.8131 | — |
| Family (V5, 12-class) | Balanced Accuracy | 0.5177 | 0.5483 |
| Zero-Day | Accuracy | ~0.92 | 0.96 |
| Zero-Day | Recall | ~0.95 | 0.97 |
| Zero-Day | F1-macro | ~0.94 | 0.96 |
        """)


if __name__ == '__main__':
    main()
