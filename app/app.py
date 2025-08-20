import streamlit as st
import requests

#i am feeling sad i do not know what happened to me i became very jealous easily i am a bad person
# streamlit run app.py

st.title("JournalWise: Cognitive Distortion Reframing Assistant")
st.write("Welcome to JournalWise! This app helps you identify and reframe cognitive distortions / negative thoughts in your thoughts / journal entries. Simply enter your thoughts below, and let our AI assist you in transforming negative patterns into positive perspectives.")

user_input = st.text_area("Enter your thoughts here.....")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter some text before submitting.")
    else:
        api_url = "http://127.0.0.1:8000/predict"

        try:
            response = requests.post(api_url, json={"text": user_input})
            response.raise_for_status()
            result = response.json()
            print(result)

            st.success("API call successful!")
            st.write(f"**It looks like you're {result["prediction"]}!**")
            st.write(result["reframed_thought"])
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")


