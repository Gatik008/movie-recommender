import streamlit as st
from hybrid import hybrid_recommendation

st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1)
movie = st.text_input("Enter a Movie Name")

if st.button("Recommend"):
    results = hybrid_recommendation(user_id, movie)
    st.subheader("Recommended Movies:")
    for r in results:
        st.write("🎥", r)
