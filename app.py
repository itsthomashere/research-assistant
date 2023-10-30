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

if __name__ == "__main__":
    main()
