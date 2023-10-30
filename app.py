from pytube import YouTube

import assemblyai as aai
import langchain
import openai
import pandas
import streamlit as st

import src.youtube as step_1
import src.assembly as step_2

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
        video_url = st.text_input("Enter a URL:", "https://www.youtube.com/watch?v=BYHaqE81WlA&pp=ygUQYXNzZW1ibHlhaSBsZW11cg%3D%3D")
        generate = st.button("Generate!")

        if video_url and generate:
            with st.spinner("Generating"):
                video_title, save_location, thumbnail_url = step_1.save_audio(video_url)
                st.header(video_title)
                st.image(thumbnail_url)
                st.audio(save_location)
                st.write(save_location)

                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(save_location)
                st.write(transcript.text)

    with col2:
        st.info("ChatGPT and text area block goes here.")
        user_input = st.text_input("Ask a Yes / No question about the transcript.")
        st.text_area("Enter a paragraph you want transformed from the passage on the left side of the screen.", height=200)

        if user_input is not None:

            questions = [
                aai.LemurQuestion(
                question="Is this caller a qualified buyer?",
                answer_options=[
                        "Yes",
                        "No"
                    ]),
                aai.LemurQuestion(
                question=user_input,
                answer_format="Short sentence")
            ]

            result = transcript.lemur.question(questions)
            st.write(result)

if __name__ == "__main__":
    main()
