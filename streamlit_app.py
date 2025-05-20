import os
import json
import streamlit as st
# from google import genai # No longer need this top-level import
import google.generativeai as genai # Correct import
from google.generativeai import types # This is fine, but GenerationConfig is directly under genai

# ——————————————————————————
# Configuration
# ——————————————————————————
# Make sure to set your GEMINI_API_KEY in the environment:
#   export GEMINI_API_KEY="YOUR_KEY"
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY environment variable not set! Please set it and restart.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"🚨 Error configuring Gemini API: {e}")
    st.stop()


MODEL_NAME = "gemini-1.5-pro-latest" # Using gemini-1.5-pro-latest as it's a common good model, 2.5 isn't widely available
# If "gemini-2.5-pro-preview-05-06" is specifically what you have access to and want to use, keep it.
# Otherwise, "gemini-1.5-pro-latest" or "gemini-1.5-flash-latest" are good general choices.

model = genai.GenerativeModel(MODEL_NAME)

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
        "Respond ONLY with a valid JSON array format, where each element is an object "
        "with keys 'question' (string) and 'answer' (string). Do not include any other text, "
        "markdown formatting, or explanations outside the JSON array."
    )
    contents = [
        # types.Content( # This structure is more for chat history
        #     role="user",
        #     parts=[ types.Part.from_text(text=prompt_text) ],
        # ),
        prompt_text # For simple text prompts, just the string is often enough
    ]
    # Forcing JSON output is generally better via GenerationConfig
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        # temperature=0.7 # Optional: control creativity
    )

    # Stream and accumulate chunks
    full_response = ""
    try:
        # Use model.generate_content with stream=True
        response_stream = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True
        )
        for chunk in response_stream:
            # When response_mime_type="application/json", chunk.text should contain the JSON string directly
            if chunk.text: # Ensure text part exists
                 full_response += chunk.text

    except Exception as e:
        st.error(f"Error during API call: {e}")
        st.error(f"Prompt sent: {prompt_text}") # Log the prompt for debugging
        # It's also good to see what, if anything, was received before the error
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             st.error(f"Prompt Feedback: {e.response.prompt_feedback}")
        if full_response:
            st.warning(f"Partial response received before error: {full_response}")
        return []

    try:
        # st.write("Raw response from API:", full_response) # For debugging
        questions = json.loads(full_response)
        # Basic validation of the received structure
        if not isinstance(questions, list):
            st.error(f"Expected a JSON array, but got type: {type(questions)}. Response:\n{full_response}")
            return []
        for item in questions:
            if not (isinstance(item, dict) and "question" in item and "answer" in item):
                st.error(f"Invalid item format in JSON array. Item: {item}. Full Response:\n{full_response}")
                return []
    except json.JSONDecodeError:
        st.error("Failed to parse questions JSON. Response was:\n" + full_response)
        return []
    except Exception as e: # Catch other potential errors during parsing/validation
        st.error(f"An unexpected error occurred while processing the response: {e}\nResponse:\n{full_response}")
        return []
    return questions

# ——————————————————————————
# Streamlit UI
# ——————————————————————————
st.set_page_config(page_title="AI-Powered Adaptive Quiz", layout="centered")

st.title("🧠 AI-Powered Quiz Prototype") # Removed "Adaptive" until implemented
st.markdown(
    """
Choose a topic and difficulty, then click **Generate**.
Answer the questions and mark whether you **know** or **don't know** each one.
"""
)

# Initialize session state
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "topic" not in st.session_state: # Persist topic input
    st.session_state.topic = "Photosynthesis"
if "difficulty" not in st.session_state: # Persist difficulty
    st.session_state.difficulty = 5
if "n_q" not in st.session_state: # Persist num questions
    st.session_state.n_q = 5

# User inputs
# Use session state to keep input values sticky across reruns if desired
st.session_state.topic = st.text_input(
    "Topic",
    value=st.session_state.topic,
    placeholder="e.g. Photosynthesis, Python decorators…"
)
st.session_state.difficulty = st.slider(
    "Difficulty level",
    min_value=1,
    max_value=10,
    value=st.session_state.difficulty
)
st.session_state.n_q = st.number_input(
    "Number of questions",
    min_value=1,
    max_value=20, # Increased max a bit
    value=st.session_state.n_q
)

if st.button("✨ Generate Questions"):
    if not st.session_state.topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner(f"Generating {st.session_state.n_q} questions on '{st.session_state.topic}' (difficulty {st.session_state.difficulty}/10)…"):
            st.session_state.quiz_questions = generate_questions(
                st.session_state.topic,
                st.session_state.difficulty,
                st.session_state.n_q
            )
            st.session_state.quiz_generated = True # Mark that quiz has been generated

if st.session_state.quiz_generated:
    if st.session_state.quiz_questions:
        st.success(f"Generated {len(st.session_state.quiz_questions)} questions!")
        for idx, qa in enumerate(st.session_state.quiz_questions, start=1):
            # Ensure qa is a dict and has the keys, robust error handling
            if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                exp = st.expander(f"Q{idx}: {qa['question']}")
                with exp: # Buttons and info should be within the expander's context
                    col1, col2 = st.columns(2)
                    # These buttons don't really "do" much yet other than display a message.
                    # For true interactivity (like tracking score), you'd need more session state.
                    if col1.button("✅ Know", key=f"know_{st.session_state.topic}_{idx}"): # Make key more unique
                        st.success("Great! Marked as 'Known'.") # Message inside expander
                    if col2.button("❌ Don't Know", key=f"dontknow_{st.session_state.topic}_{idx}"): # Make key more unique
                        st.info(f"**Answer:** {qa['answer']}") # Message inside expander
            else:
                st.warning(f"Question {idx} has an invalid format: {qa}")
        if st.button("Reset Quiz", key="reset_quiz"):
            st.session_state.quiz_generated = False
            st.session_state.quiz_questions = []
            st.session_state.topic = "Photosynthesis" # Reset to default or last
            st.rerun()

    elif st.session_state.quiz_generated: # If generated flag is true but no questions
        st.error("No questions were generated. Try again with a different topic or difficulty, or check the logs if errors appeared above.")
