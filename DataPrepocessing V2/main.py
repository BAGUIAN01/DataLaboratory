from streamlit_option_menu import option_menu
import streamlit as st
import dataframefunctions
import dataexploration
import plots
import runpredictions
from PIL import Image
import numpy as np
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
import sweetviz as sv

st.markdown("""
<style>
.first_titre {
    font-size:60px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    # text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
""", unsafe_allow_html=True)

# def load_data():
#     if "dataframe" not in st.session_state:
#             try:
#                 if 'csv' in st.session_state.file_details['FileName']:
#                     if st.session_state.separateur != "":
#                         st.session_state.dataframe = pd.read_csv(uploaded_file, sep=st.session_state.separateur, engine='python')
#                     else:
#                         st.session_state.dataframe = pd.read_csv(uploaded_file)
#                 else:
#                     if st.session_state.separateur != "":
#                         st.session_state.dataframe = pd.read_excel(uploaded_file, sep=st.session_state.separateur, engine='python')
#                     else:
#                         st.session_state.dataframe = pd.read_excel(uploaded_file)
#             except:
#                 pass






choose = option_menu("Data Cleaner",["Home","Prepocesing","Vizualisation","M.Learning","About"],
    icons=['house','file','bi bi-search','graph-up','person lines fill'],
    menu_icon = "None", default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "10px", "text-align": "left", "margin":"5px", "--hover-color": ""},
        "nav-link-selected": {"background-color": ""},
    },orientation = "horizontal"
    )
#*********************************************************************
if choose=="Home":
    st.markdown('<p class="first_titre">Data Cleaner Platform</p>', unsafe_allow_html=True)
    st.write("---")
    c1, c2 = st.columns((3, 2))
    with c1:
        st.write("##")
        st.markdown(
            '<p class="intro"><b>Bienvenue sur la plateforme de data science !</b></p>',
            unsafe_allow_html=True)
    with c1:
        st.subheader("Teams")
        st.write(
            "• [DIASSANA Fatoumata/GitHub](https://github.com/Diaffat)")
    # if 'a' in st.session_state:  
    #         st.session_state['a'] = "Bonjour"
    # st.write(st.session_state.a)
    # df = pd.read_csv("df")
    # pr = df.profile_report()

    # st_profile_report(pr)



if choose=="Prepocesing":
    # with st.sidebar:
    #     st.markdown("## **1.First Step** ##")
    #     data = st.sidebar.file_uploader("Please upload your dataset (CSV format):", type=['csv','xlsx'])
    #     is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
    #     if data is not None:
    #         is_loaded_dataset.success("Dataset uploaded successfully!")
    
    st.session_state.data = dataframefunctions.get_dataframe()
    if st.session_state.data is not None:
        dataexploration.load_page(st.session_state.data)

    # df = dataframe.copy()
    # df.to_csv("df")
        
    # if data is not None:

    #     dataexploration.load_page(df)
            
            
                # # is_loaded_dataset.success("Dataset uploaded successfully!")
                # dataframe = dataframefunctions.get_dataframe(st.session_state.dataframe)
                
    # if st.session_state.dataframe is not None:
    #     dataexploration.load_page(df)
    
#*********************************************************************
if choose=="Vizualisation":
    
    df = pd.read_csv("df")
    select = st.sidebar.selectbox("Select the Kind of plot",["Plot","Statistique Test"])
    if select== "Plot":
        plots.load_page(df)

if choose=="M.Learning":
    df = pd.read_csv("df")
    select1 = st.sidebar.selectbox("Choose",["Prediction","Classification"])
    if select1=="Prediction":
        runpredictions.load_page(df)
    if select1=="Classification":
        runpredictions.classification()

if choose=="About":
    st.write(st.session_state.data)

