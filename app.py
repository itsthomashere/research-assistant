from pytube import YouTube

import assemblyai as aai
import langchain
import openai
import pandas
import streamlit as st

aai.settings.api_key = st.secrets["ASSEMBLYAI_TOKEN"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

def main() -> None:
    st.title("AI Research Assistant")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Block 1")

    with col2:
        st.info("Block 2")

if __name__ == "__main__":
    main()
