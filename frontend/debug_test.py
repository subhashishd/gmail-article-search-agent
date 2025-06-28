"""Simple debug test for Streamlit button functionality."""

import streamlit as st
import requests
import os

# Configure page
st.set_page_config(page_title="Debug Test", layout="wide")

st.title("🔧 Debug Test Page")

# Backend URL
BACKEND_URL = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8000")
st.write(f"Backend URL: {BACKEND_URL}")

# Test 1: Simple button
st.subheader("Test 1: Simple Button")
if st.button("🧪 Simple Test Button"):
    st.success("✅ Simple button click registered!")
    st.balloons()

# Test 2: Network test button
st.subheader("Test 2: Network Test")
if st.button("🌐 Test Backend Connection"):
    try:
        with st.spinner("Testing connection..."):
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend connection successful!")
                st.json(response.json())
            else:
                st.error(f"❌ Backend returned status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")

# Test 3: Fetch test button
st.subheader("Test 3: Fetch Test")
if st.button("📧 Test Fetch Endpoint"):
    try:
        with st.spinner("Testing fetch endpoint..."):
            response = requests.post(f"{BACKEND_URL}/agents/fetch", json={}, timeout=10)
            if response.status_code == 200:
                st.success("✅ Fetch endpoint successful!")
                st.json(response.json())
            else:
                st.error(f"❌ Fetch endpoint returned status {response.status_code}")
                st.text(response.text)
    except Exception as e:
        st.error(f"❌ Fetch request failed: {str(e)}")

# Connection status
st.subheader("Connection Status")
st.write("If you see this page properly, Streamlit is working.")
st.write("If buttons above don't work, there's a Streamlit issue.")
st.write("If network tests fail, there's a backend connectivity issue.")
