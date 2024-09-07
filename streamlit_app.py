import streamlit as st
import requests
import json
from io import StringIO
from docx import Document
import tiktoken  # Librería para contar tokens

# Function to read a .docx file and extract text
def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Function to count tokens in a document (adjust according to model's tokenizer)
def count_tokens(text, model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")  # Adjust encoding to model's tokenizer
    return len(encoding.encode(text))

# Function to split text into chunks that fit within token limits
def split_text_into_chunks(text, max_tokens_per_chunk, model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    # Split tokens into chunks that fit within the max token limit
    chunks = [tokens[i:i + max_tokens_per_chunk] for i in range(0, len(tokens), max_tokens_per_chunk)]
    
    # Decode tokens back into text
    text_chunks = [encoding.decode(chunk) for chunk in chunks]
    return text_chunks

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer!"
)

# Read API key from Streamlit secrets.
api_key = st.secrets["together"]["api_key"]

if not api_key:
    st.info("Please add your Together API key in secrets.", icon="🗝️")
else:
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, or .docx)", type=("txt", "md", "docx")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    # Add a submit button
    if st.button("Submit") and uploaded_file and question:
        # Process the uploaded file based on its type.
        if uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md"):
            document = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".docx"):
            document = read_docx(uploaded_file)

        # Concatenate the document and question
        input_text = f"Here's a document: {document} \n\n---\n\n {question}"

        # Split the input text into chunks that fit within the token limit
        max_allowed_tokens_per_chunk = 8193 - 3000  # Leave space for the new tokens (3000 max_tokens)
        text_chunks = split_text_into_chunks(input_text, max_allowed_tokens_per_chunk)

        # Loop through each chunk and send it to the API
        for i, chunk in enumerate(text_chunks):
            st.write(f"Processing chunk {i+1}/{len(text_chunks)}")

            messages = [
                {
                    "role": "user",
                    "content": chunk,
                }
            ]

            # Prepare the payload for the Together API.
            payload = {
                "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                "messages": messages,
                "max_tokens": 3000,  # Increased max_tokens to 3000
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Make a POST request to Together API.
            try:
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    stream=True,
                )

                # Stream the response to the app.
                if response.status_code == 200:
                    st.write(f"Response from the model for chunk {i+1}:")
                    for line in response.iter_lines():
                        if line:
                            decoded_line = json.loads(line.decode("utf-8"))
                            if "choices" in decoded_line:
                                st.write(decoded_line["choices"][0]["message"]["content"])
                else:
                    st.error(f"Error {response.status_code}: Unable to get a response from the Together API.")
                    st.error(response.text)  # Show detailed error message from the API
            except Exception as e:
                st.error(f"An error occurred: {e}")
