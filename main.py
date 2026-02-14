import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate Research Tool")
st.title("Real Estate Research Tool")

st.sidebar.header("Enter URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

if st.sidebar.button("Process URLs"):
    urls = [u for u in (url1, url2, url3) if u.strip()]

    if not urls:
        st.error("Please enter at least one URL.")
    else:
        for status in process_urls(urls):
            st.info(status)

st.header("Ask a Question")

query = st.text_input("Enter your question")

if query:
    try:
        answer, sources = generate_answer(query)

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                st.write(source)

    except Exception as e:
        st.error(str(e))
