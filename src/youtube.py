import os
from pytube import YouTube
import streamlit as st

@st.cache_data
def save_audio(url):
	yt = YouTube(url)
	video = yt.streams.filter(only_audio=True).first()
	out_file = video.download()
	base, ext = os.path.splitext(out_file)
	file_name = base + '.mp3'
	os.rename(out_file, file_name)
	st.write(yt.title + " has been successfully downloaded.")
	st.write(file_name)
	return yt.title, file_name, yt.thumbnail_url


