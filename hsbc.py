# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:50:55 2024

@author: M Fakhri Pratama
"""

import streamlit as st
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
import time
# ================================
# 1. Initialize Pinecone
# ================================
pc = Pinecone(api_key="pcsk_7XaUW7_QCeooT2sWPnZHVjuep4jgbGJYQ8XYYqm7hZuyU6HisoAcqeU19ftPpH2dWbV53J")

# Check if the index exists, and create it if not
index_name = "hsbc-creditcard-terms"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ================================
# 2. Initialize Azure OpenAI Client
# ================================
openai_client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_version="2024-10-21",
    api_key="051854ff976243268f1bb5958c7c644b"
)

# ================================
# 3. Helper Functions
# ================================
@st.cache_data(show_spinner=False)
def generate_query_embedding_cached(query):
    """Generate embeddings for queries (cached)."""
    response = openai_client.embeddings.create(model="text-embedding-ada-002", input=query)
    return response.data[0].embedding

def gpt_process_query(chat_history, instruction="Refine this query for better semantic search accuracy."):
    """
    Refine the user query using GPT, considering the context of previous chat history.
    """
    try:
        # Combine chat history and instruction
        messages = [{"role": "system", "content": instruction}] + chat_history

        # Call GPT to refine the query
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with your specific model
            messages=messages,
            temperature=0.2,
            max_tokens=100
        )
        refined_query = response.choices[0].message.content.strip()
        return refined_query
    except Exception as e:
        st.warning(f"⚠ Failed to refine query: {e}")
        return chat_history[-1]["content"]  # Default to the last user query

def search_pinecone(query_text, top_k=5):
    """Perform semantic search in Pinecone."""
    query_embedding = generate_query_embedding_cached(query_text)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results.matches

def extract_entities_from_text(text):
    """Extract entities using GPT with a fallback to keyword extraction."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts entities from a text."},
                {"role": "user", "content": f"Extract entities (like cards, rebates, or students) from this text:\n{text}"}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content.split("\n")
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return []

def process_pinecone_results(pinecone_results):
    """Process Pinecone results to extract entities and relationships."""
    entities = set()
    relationships = []

    for result in pinecone_results:
        text = result.metadata.get("text", "")
        category = result.metadata.get("category", "")

        # Extract entities
        extracted_entities = extract_entities_from_text(text)
        entities.update(extracted_entities)

        # Identify relationships
        if category == "relationships":
            relationships.append(text)
    
    return list(entities), relationships

def generate_final_response(query, entities, relationships, semantic_results):
    """Generate final response by combining all extracted data."""
    context = "\n--- Top Semantic Results ---\n"
    for result in semantic_results:
        context += f"- {result.metadata.get('text', '')}\n"

    context += "\n--- Extracted Entities ---\n"
    for entity in entities:
        context += f"- {entity}\n"

    context += "\n--- Extracted Relationships ---\n"
    for rel in relationships:
        context += f"- {rel}\n"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant answering credit card-related questions for HSBC accurately. Make sure recommend at least one credit card from the bank"},
            {"role": "user", "content": f"Question: {query}\nContext: {context}\n\nAnswer the question using the context."}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return response.choices[0].message.content

def full_query_workflow(query, top_k=10, refine_query=True):
    """
    Full workflow: refine query -> embedding -> query Pinecone -> extract entities/relationships -> GPT response.
    """
    start_time = time.time()
    progress_bar = st.progress(0, text="Starting process...")

    # Step 1: Query Refinement with Chat History
    chat_history = st.session_state.messages.copy()
    if refine_query:
        progress_bar.progress(15, text="Refining query with GPT...")
        refined_query = gpt_process_query(chat_history + [{"role": "user", "content": query}])
    else:
        refined_query = query
    progress_bar.progress(25, text="Generating query embedding...")

    # Step 2: Generate Query Embedding
    query_embedding = generate_query_embedding_cached(refined_query)
    progress_bar.progress(50, text="Querying Pinecone database...")

    # Step 3: Semantic Search in Pinecone
    pinecone_results = search_pinecone(refined_query, top_k=top_k)
    progress_bar.progress(65, text="Extracting entities and relationships...")

    # Step 4: Extract Entities and Relationships
    entities, relationships = process_pinecone_results(pinecone_results)
    progress_bar.progress(80, text="Generating GPT response...")

    # Step 5: Generate Final Response Using GPT
    final_response = generate_final_response(refined_query, entities, relationships, pinecone_results)
    progress_bar.progress(100, text="Done!")

    # Clean up progress bar
    progress_bar.empty()
    st.write(f"✅ Processed in **{time.time() - start_time:.2f} seconds**.")

    return final_response, pinecone_results, entities, relationships


# ================================
# 4. Streamlit UI
# ================================
st.markdown("""
<style>
/* Custom Background Colors */
.sidebar .sidebar-content { background-color: #A50027; color: white; }
.reportview-container { background-color: white; }

/* Custom Image Styles */
.image-container { height: 200px; position: relative; }
.image-container img {
    position: absolute; top: 50%; left: 50%;
    height: auto; width: 100%; max-height: 100%;
    transform: translate(-50%, -50%);
    object-fit: contain;
}
.small-subheader {
    font-size: 18px; font-weight: bold; margin-bottom: 20px;
    border-bottom: 2px solid #A50027; padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)
 
#Sidebar Header
st.sidebar.image("logo.png", width=150)
st.sidebar.title("CardWise")
st.sidebar.write("Your AI-powered credit card assistant.")

# Sidebar Navigation with List-style Buttons
st.sidebar.markdown("""
    <style>
        /* General styling for navigation links */
        .nav-link {
            padding: 12px 15px; /* Consistent padding */
            border-radius: 5px;
            font-size: 16px;
            font-weight: normal;
            cursor: pointer;
            color: white;
            text-decoration: none;
            display: block;
            text-align: center; /* Center align text */
            margin-bottom: 5px;
        }
        
        /* Hover effect */
        .nav-link:hover {
            background-color: #8C001A; /* Darker red on hover */
        }
        
        /* Styling for selected link */
        .nav-link.selected {
            background-color: #A50027; /* HSBC Red for selected link */
            color: white; /* Ensure text color is white */
            font-weight: normal; /* Keep text weight normal */
            border: 2px solid transparent; /* Smooth appearance */
            box-shadow: inset 0px 0px 0px 2px #A50027; /* Optional shadow for emphasis */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation with Buttons
if "current_page" not in st.session_state:
    st.session_state.current_page = "Credit Card Assistant"

pages = [ "Credit Card Assistant", "Terms and Disclaimer"]

# Create buttons for each page
for page_name in pages:
    if st.sidebar.button(page_name, key=page_name):
        st.session_state.current_page = page_name

# Display the content of the selected page
page = st.session_state.current_page
# Navigation
#selected_page = st.sidebar.radio("Navigation", pages)

if page == "Credit Card Assistant":
    # Page Title and Description
    st.title("HSBC Credit Card Semantic Search App: Knowledge Graph Powered")
    st.caption("🔍 Search for terms and conditions using advanced knowledge graph semantic search powered by OpenAI and Pinecone!")

    # Sidebar Option for Results Limit
    top_k = st.sidebar.slider("Number of results:", 1, 10, value=5)

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query Input Section
    if query := st.chat_input("Enter your search query:"):
        # Add user query to session history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display the user's query in chat
        with st.chat_message("user"):
            st.markdown(query)

        # Process Query
        with st.spinner("Processing your query..."):
            try:
                # Run the full query workflow
                response, matches, entities, relationships = full_query_workflow(query, top_k=top_k)

                # Add the assistant's response to session history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Display Tabs for Results
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Final Response", "Extracted Entities", "Extracted Relationships", "Relevant Documents"
                ])

                # Tab 1: Final GPT Response
                with tab1:
                    st.markdown("### **Final Response**")
                    st.markdown(f"""<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; color: black;">
                        {response}
                    </div>""", unsafe_allow_html=True)

                # Tab 2: Extracted Entities
                with tab2:
                    st.markdown("### **Extracted Entities**")
                    # Clean up entities: Remove duplicates, empty strings, and leading/trailing whitespace
                    unique_entities = sorted(set([entity.strip() for entity in entities if entity.strip()]))

                    # Check if there are valid entities
                    if unique_entities:
                    # Display entities as a clean bullet list
                        for idx, entity in enumerate(unique_entities, start=1):
                            st.write(f"- **{idx}.** {entity}")
                    else:
                        st.info("No entities found in the relevant documents.")

                # Tab 3: Extracted Relationships
                with tab3:
                    st.markdown("### **Extracted Relationships**")
                    if relationships:
                        for rel in relationships:
                            st.write(f"- {rel}")
                    else:
                        st.info("No relationships found in the relevant documents.")

                # Tab 4: Relevant Documents
                with tab4:
                    st.markdown("### **Relevant Documents**")
                    if matches:
                        for match in matches:
                            with st.expander(f"📄 {match.metadata.get('filename', 'Unknown')} (Score: {match.score:.4f})"):
                                st.write(match.metadata.get("text", "No text available"))
                    else:
                        st.info("No relevant documents found.")

            except Exception as e:
                # Error Handling
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

elif page == "Terms and Disclaimer":
    st.title("Terms and Disclaimer")
    st.markdown("""
    **Terms of Use:**  
    - These terms and conditions govern your use of this application.  
    - By using this app, you accept these terms in full.

    **Disclaimer:**  
    - The information provided in this application is for general informational purposes only.  
    - This app is not intended to provide legal advice. Please consult a professional for specific advice.
    """)
