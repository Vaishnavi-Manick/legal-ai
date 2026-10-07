import os
import sys
import time
import streamlit as st

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import LegalSearchEngine

# Page configuration
st.set_page_config(
    page_title="Legal AI Case Retrieval",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache search engine instance to load model only once
@st.cache_resource(show_spinner="Loading AILA dataset & pretrained Legal Transformer model...")
def load_search_engine():
    """Loads and caches LegalSearchEngine instance."""
    return LegalSearchEngine(candidate_pool_size=50)

def main():
    # Header Section
    st.title("⚖️ Legal AI Case Retrieval")
    st.caption("Transformer-based Legal Case Retrieval using AILA 2019")
    st.markdown("---")

    # Sidebar - System Information
    st.sidebar.header("System Information")
    st.sidebar.markdown("""
    **Dataset:** AILA 2019  
    **Total Documents:** 2,914 cases  
    **Stage 1 Retrieval:** TF-IDF  
    **Stage 2 Re-ranking:** Transformer Cross-Encoder  
    **Candidate Cases:** Top 50  
    **Final Results:** Top 5  
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("Model loaded once via Streamlit caching.")

    # Initialize Engine safely with user-friendly error handling
    try:
        engine = load_search_engine()
    except Exception as e:
        st.error(f"Failed to initialize Legal Search Engine: {str(e)}")
        st.stop()

    # User Input Section
    query_text = st.text_area(
        label="Enter your legal query",
        placeholder="Example: right to privacy under Article 21",
        height=100
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_clicked = st.button("🔍 Search Cases", type="primary", use_container_width=True)

    # Search Execution & Display Results
    if search_clicked:
        if not query_text or not query_text.strip():
            st.warning("Please enter a legal query before searching.")
        else:
            with st.spinner("Searching 2,914 case documents using TF-IDF & Legal Transformer..."):
                try:
                    start_time = time.time()
                    results = engine.search(query_text.strip(), top_k=5)
                    elapsed_time = time.time() - start_time
                except Exception as e:
                    st.error(f"An error occurred during search: {str(e)}")
                    results = []

            if not results:
                st.info("No relevant cases found for your query.")
            else:
                st.subheader("Top 5 Relevant Cases")
                st.caption(f"Retrieved in {elapsed_time:.2f} seconds")

                rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

                for item in results:
                    rank = item["rank"]
                    case_id = item["case_id"]
                    score = item["relevance_score"]
                    snippet = item["snippet"]
                    doc_text = item["document_text"]
                    case_name = item.get("case_name")
                    citation = item.get("citation")

                    emoji = rank_emojis[rank - 1] if rank <= len(rank_emojis) else f"#{rank}"
                    expander_title = f"{emoji} Rank {rank} — Case ID: {case_id} (Relevance Score: {score:.4f})"

                    with st.expander(expander_title, expanded=(rank == 1)):
                        if case_name:
                            st.markdown(f"**Case Name:** {case_name}")
                        if citation:
                            st.markdown(f"**Citation:** {citation}")

                        st.markdown("**Relevant Snippet:**")
                        st.info(f'"{snippet}"')

                        with st.popover("📄 View Full Case Document"):
                            st.text_area(
                                label=f"Full Text ({case_id})",
                                value=doc_text,
                                height=400,
                                disabled=True
                            )

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.85rem;'>"
        "Research Prototype | AILA 2019 | Transformer-based Retrieval"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
