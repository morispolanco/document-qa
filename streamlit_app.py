import streamlit as st
import requests
import json

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
)

# Read API key from Streamlit secrets.
api_key = st.secrets["together"]["api_key"]

if not api_key:
    st.info("Please add your Together API key in secrets.", icon="🗝️")
else:
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Prepare the payload for the Together API.
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "messages": messages,
            "max_tokens": 2512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["\"\""],
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Make a POST request to Together API.
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            stream=True,
        )

        # Stream the response to the app.
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    st.write(decoded_line)
        else:
            st.error("Error: Unable to get a response from the Together API.")
