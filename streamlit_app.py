import os
import json
import streamlit as st
from google import genai
from google.genai import types

# ——————————————————————————
# Configuration
# ——————————————————————————
# Make sure to set your GEMINI_API_KEY in the environment:
#   export GEMINI_API_KEY="YOUR_KEY"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash-preview-04-17"

# ——————————————————————————
# AI-powered question generation using latest Gemini streaming API
# ——————————————————————————
def generate_questions(topic: str, difficulty: int, n_questions: int = 5):
    """
    Generate a JSON array of quiz questions with answers using Gemini streaming.
    Returns a list of dicts: [{"question": str, "answer": str}, …].
    """
    prompt_text = (
        f"Generate {n_questions} quiz questions on the topic '{topic}', "
        f"at difficulty level {difficulty}/10. "
        "Respond in JSON array format, each element with keys 'question' and 'answer'."
    )
    contents = [
        types.Content(
            role="user",
            parts=[ types.Part.from_text(text=prompt_text) ],
        ),
    ]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    # Stream and accumulate chunks
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config,
    ):
        text_chunk = chunk.text
        full_response += text_chunk
    
    try:
        questions = json.loads(full_response)
    except json.JSONDecodeError:
        st.error("Failed to parse questions JSON. Response was:\n" + full_response)
        return []
    return questions

# ——————————————————————————
# Streamlit UI
# ——————————————————————————
st.set_page_config(page_title="AI-Powered Adaptive Quiz", layout="centered")

st.title("🧠 AI-Powered Adaptive Quiz Prototype")
st.markdown(
    """
Choose a topic and difficulty, then click **Generate**.
Answer the questions and mark whether you **know** or **don't know** each one.
"""
)

# User inputs
topic = st.text_input("Topic", placeholder="e.g. Photosynthesis, Python decorators…")
difficulty = st.slider("Difficulty level", min_value=1, max_value=10, value=5)
n_q = st.number_input("Number of questions", min_value=1, max_value=10, value=5)

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating questions…"):
            quiz = generate_questions(topic, difficulty, n_q)

        if quiz:
            st.success(f"Generated {len(quiz)} questions!")
            for idx, qa in enumerate(quiz, start=1):
                exp = st.expander(f"Q{idx}: {qa['question']}")
                col1, col2 = exp.columns(2)
                if col1.button("✅ Know", key=f"know_{idx}"):
                    exp.success("Great! Moving on.")
                if col2.button("❌ Don't Know", key=f"dontknow_{idx}"):
                    exp.info(f"Answer: {qa['answer']}")
        else:
            st.error("No questions generated. Try again with a different topic or difficulty.")
