from pytube import YouTube

import assemblyai as aai
import langchain
import openai
import pandas
import streamlit as st

import src.youtube as step_1
import src.assembly step_2

aai.settings.api_key = st.secrets["ASSEMBLYAI_TOKEN"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


st.set_page_config(
        page_title="Research Assistant",
        page_icon="🤖",
        layout="wide"
        )


def main() -> None:
    st.title("AI Research Assistant")
    col1, col2 = st.columns(2)

    with col1:
        st.info("YouTube / AssemblyAI stuff goes here")

    with col2:
        st.info("ChatGPT and text area block goes here.")

if __name__ == "__main__":
    main()
