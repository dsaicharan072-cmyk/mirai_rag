import os
import requests
import streamlit as st

# Application Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-policy {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 3px solid #6366F1;
        padding: 8px 12px;
        margin-top: 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check backend health
def check_backend_status():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if res.status_code == 200:
            return True, res.json()
        return False, None
    except Exception:
        return False, None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your MirAI Student Policy Advisor. You can ask me any question regarding attendance criteria, medical/duty leave procedures, clubs, exams, or campus conduct.",
            "sources": []
        }
    ]

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("MirAI Administration")
    st.caption("Official Policy Knowledge System (2026)")

    # System Status Indicator
    is_online, health_data = check_backend_status()
    if is_online:
        st.success("🟢 Backend Connected")
        if health_data:
            chunks = health_data.get("indexed_chunks", 0)
            st.info(f"📊 Vector Database: **{chunks}** policy chunks indexed")
    else:
        st.error("🔴 Backend Offline")
        st.warning("Ensure the FastAPI server is running at `http://localhost:8000`.")

    st.divider()

    # Document Ingestion Section
    st.subheader("📄 Handbook Ingestion")
    uploaded_file = st.file_uploader("Upload New Handbook (PDF)", type=["pdf"])
    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Processing PDF chunks and updating ChromaDB..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"Ingested {data['filename']} successfully! ({data['total_chunks']} chunks created)")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The PDF may be too large to process synchronously.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend server. Please verify FastAPI is running.")
                except Exception as e:
                    st.error(f"Error during upload: {str(e)}")

    st.divider()

    # Quick Audit Test Queries
    st.subheader("🧪 Certification Test Suite")
    st.write("Click a query to populate the prompt:")

    test_queries = [
        ("1. Attendance Marks", "I have 72% attendance. How many attendance marks will I get?"),
        ("2. Medical Leave (Ratnam)", "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"),
        ("3. Club Creation Process", "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"),
        ("4. Smoking Policy / Fine", "How much is the fine for smoking a cigarette on campus?")
    ]

    for label, query_text in test_queries:
        if st.button(label, use_container_width=True):
            st.session_state.quick_query = query_text

    st.divider()
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your MirAI Student Policy Advisor. How may I assist you today?",
                "sources": []
            }
        ]
        st.rerun()

# Main Chat Interface
st.markdown('<div class="badge-policy">Official MSOT Student Handbook 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">MirAI Policy Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous RAG advisor strictly grounded in official university regulations.</div>', unsafe_allow_html=True)

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Verified Handbook Sources"):
                for src in msg["sources"]:
                    page_label = f"Page {src['page']}" if src.get("page") else "Reference Excerpt"
                    st.markdown(f"**{page_label}:**")
                    st.markdown(f"_{src['content'][:250]}..._")

# Handle Quick Query Pre-fill
initial_input = ""
if "quick_query" in st.session_state:
    initial_input = st.session_state.pop("quick_query")

# Chat Input Box
user_prompt = st.chat_input("Ask a question about MirAI student policies...") or initial_input

if user_prompt:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Call Backend REST API
    with st.chat_message("assistant"):
        with st.spinner("Consulting official handbook policies..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": user_prompt},
                    timeout=90
                )

                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("answer", "No response generated.")
                    sources_data = data.get("sources", [])

                    st.markdown(answer_text)
                    if sources_data:
                        with st.expander("📚 Verified Handbook Sources"):
                            for src in sources_data:
                                page_label = f"Page {src['page']}" if src.get("page") else "Reference Excerpt"
                                st.markdown(f"**{page_label}:**")
                                st.markdown(f"_{src['content'][:250]}..._")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data
                    })
                elif response.status_code == 422:
                    error_msg = "Invalid query format. Please enter a valid question."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                else:
                    error_msg = f"Server returned status {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out while waiting for policy synthesis. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except requests.exceptions.ConnectionError:
                error_msg = "🔌 Connection Error: Unable to reach the FastAPI backend at `http://localhost:8000`. Please make sure the backend server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
