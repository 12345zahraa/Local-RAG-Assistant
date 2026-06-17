import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Helpers
# -----------------------------
def check_server_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        return response.status_code in (200, 401, 403)
    except requests.RequestException:
        return False


def index_document(uploaded_file) -> requests.Response | Exception:
    try:
        response = requests.post(
            f"{API_BASE_URL}/index-document",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            },
            timeout=180,
        )
        return response
    except requests.RequestException as exc:
        return exc


def ask_question(question: str):
    try:
        response = requests.get(
            f"{API_BASE_URL}/ask",
            params={"question": question},
            timeout=180,
        )
        return response
    except requests.RequestException as exc:
        return exc


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I’m your local document assistant. Ask me anything about the indexed content.",
        }
    ]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📚 Document Manager")
    st.caption("Upload a PDF to add it to the local knowledge base.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Only PDF files are supported.",
    )

    if st.button("Index Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a PDF file before indexing.")
        else:
            with st.spinner("Indexing document..."):
                result = index_document(uploaded_file)

            if isinstance(result, requests.Response) and result.status_code == 200:
                payload = result.json()
                st.success(
                    f"Document indexed successfully. {payload.get('chunk_count', 0)} chunk(s) added."
                )
            elif isinstance(result, requests.Response):
                try:
                    error_detail = result.json().get("detail", result.text)
                except ValueError:
                    error_detail = result.text
                st.error(f"Indexing failed: {error_detail}")
            else:
                st.error(
                    "Could not reach the FastAPI server. Please make sure the backend is running at "
                    f"{API_BASE_URL}."
                )

    st.divider()
    server_available = check_server_health()
    if server_available:
        st.success("Backend server is reachable.")
    else:
        st.warning("Backend server is not responding. Check that the API is running.")


# -----------------------------
# Main content
# -----------------------------
st.title("🧠 Local RAG Chat")
st.caption("Ask questions using your indexed documents.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(prompt)

        if isinstance(response, requests.Response) and response.status_code == 200:
            payload = response.json()
            answer = payload.get("answer", "")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif isinstance(response, requests.Response):
            try:
                error_detail = response.json().get("detail", response.text)
            except ValueError:
                error_detail = response.text
            message = f"Sorry, I couldn't process that request. {error_detail}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
        else:
            message = (
                "The server is not responding right now. Please verify that the FastAPI API is running "
                f"at {API_BASE_URL}."
            )
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
