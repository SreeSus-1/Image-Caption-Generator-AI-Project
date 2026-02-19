import streamlit as st
import requests
import hashlib

st.title("Image Caption Generator (LLaVA)")

@st.cache_resource
def get_session():
    return requests.Session()

session = get_session()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    st.image(img_bytes, caption="Uploaded Image", width=500)

    # hash image so we don't regenerate for the same upload
    img_hash = hashlib.md5(img_bytes).hexdigest()

    if "captions" not in st.session_state:
        st.session_state.captions = {}

    col1, col2 = st.columns(2)
    with col1:
        generate = st.button("Generate Caption", type="primary")
    with col2:
        clear = st.button("Clear cached caption")

    if clear:
        st.session_state.captions.pop(img_hash, None)
        st.rerun()

    # If already generated, show instantly
    if img_hash in st.session_state.captions:
        st.subheader("Caption:")
        st.write(st.session_state.captions[img_hash])

    if generate:
        with st.spinner("Generating caption..."):
            try:
                files = {
                    # include filename + content-type to be safe
                    "file": (uploaded_file.name, img_bytes, uploaded_file.type)
                }
                resp = session.post(
                    "http://localhost:8000/caption/",
                    files=files,
                    timeout=(5, 120),  # connect timeout, read timeout
                )
                resp.raise_for_status()
                caption = resp.json().get("caption", "Error generating caption.")
                st.session_state.captions[img_hash] = caption
                st.rerun()

            except requests.exceptions.Timeout:
                st.error("Timed out waiting for the backend. LLaVA inference may be taking too long.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            except ValueError:
                st.error("Backend did not return valid JSON.")
