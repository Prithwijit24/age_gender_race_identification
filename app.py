from PIL.Image import Image
import streamlit as st
from rich import print as rprint
from PIL import Image
import io
import numpy as np
import subprocess as sb
import os
import duckdb
duckdb.execute("INSTALL motherduck; LOAD motherduck;")
ac_tkn = st.secrets["MOTHERDUCK_TOKEN"]
duckdb.execute(f"SET motherduck_token='{ac_tkn}';")

# from sympy.plotting.backends.textbackend import text
st.title("Identify Gender and Race", width = "stretch")

import streamlit as st
select_media = st.radio("Please select how to upload your photo",options=("open camera", "upload from device"), horizontal=True)

if select_media == "open camera":
    camera_image = st.camera_input('Please take your photo')
    upload_image = None
else:
    upload_image = st.file_uploader("Please upload your photo")
    camera_image = None

    upload_image_to_show = Image.open((upload_image))
    st.image(upload_image_to_show)

image_to_use = camera_image if camera_image is not None else upload_image
if len(os.listdir("data")):
    sb.run("mkdir -p data", shell=True, text=True)
Image.open(image_to_use).save("data/photo.jpg","JPEG")



def highlighted_output(word):
    return f"""
    <span style='
        background-color: yellow; 
        border-radius: 0.4em; 
        padding: 0.1em 0.4em; 
        font-weight: bold; 
        color: black;'>
        {word}
    </span>
    """



select_target = st.multiselect("Please select what you want to predict : ",("Gender", "Race", "All"), default="All")

if select_target[0] in ('', 'All'):
    pred_gender = sb.run("uv run main.py --prediction_type single --target gender --image_path data/photo.jpg", shell=True, text=True, capture_output=True).stderr.split("\n")[-2].split(':')[-1]
    pred_race = sb.run("uv run main.py --prediction_type single --target race --image_path data/photo.jpg", shell=True, text=True, capture_output=True).stderr.split("\n")[-2].split(':')[-1]

    output = f"Your **Gender** is {highlighted_output(pred_gender)}  && your **Race** is {highlighted_output(pred_race)} 🙂"
    st.markdown(output, unsafe_allow_html=True)


elif select_target[0] == 'Gender':
    pred_gender = sb.run("uv run main.py --prediction_type single --target gender --image_path data/photo.jpg", shell=True, text=True, capture_output=True).stderr.split("\n")[-2].split(':')[-1]
    output = f"Your **Gender** is {highlighted_output(pred_gender)} 🙂"
    st.markdown(output, unsafe_allow_html=True)

elif select_target[0] == 'Race':
    pred_race = sb.run("uv run main.py --prediction_type single --target race --image_path data/photo.jpg", shell=True, text=True,capture_output=True).stderr.split("\n")[-2].split(':')[-1]
    output = f"Your **Race** is {highlighted_output(pred_race)} 🙂"
    st.markdown(output, unsafe_allow_html=True)
else:
    st.error('Please select target from ("All", "Gender", "Race")')

sb.run("rm data/photo.jpg", shell=True, text=True)
st.snow()






