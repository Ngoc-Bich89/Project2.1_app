import streamlit as st
st.title("Trung Tam Tin Hoc")
st.subheader("How to run streamlit app")
menu = ["Home", "About"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.subheader("Streamlit From Windows")
elif choice == 'About':
    st.subheader("[Trung Tam Tin Hoc](https://csc.edu.vn)")