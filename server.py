import streamlit as st
import json



with open("demo/output.json", "r") as dummy_data_file:
    dummy_data = json.load(dummy_data_file)

dummy_output_data = dummy_data

# Dummy inference function
def inference(conversation_json):
    # For now, just return a placeholder JSON
    return {
        "conversation_ID": 1,
        "conversation": conversation_json,
        "emotion_cause_pairs": [["dummy_joy", "dummy_cause"]]
    }


# Function to display conversation as a chat
# Function to display conversation as a chat
def display_chat(conversation_json, emotion_cause_pairs):
    st.subheader("Conversation Chat")

    # Build a lookup table for emotion-cause pairs
    cause_lookup = {
        pair[0].split('_')[0]: pair[1] for pair in emotion_cause_pairs
    }

    # Define color coding for emotions
    emotion_colors = {
        "anger": "#FF4500",
        "disgust": "#8B4513",
        "fear": "#8A2BE2",
        "joy": "#FFD700",
        "sadness": "#1E90FF",
        "surprise": "#32CD32",
    }

    for utterance in conversation_json:
        utterance_id = str(utterance["utterance_ID"])
        text = utterance["text"]
        speaker = utterance["speaker"]
        emotion = utterance.get("emotion", "neutral")  # Default to neutral if no emotion
        color = emotion_colors.get(emotion, "#000000")  # Default to black for undefined emotions

        # Render the main utterance with a background color
        message = st.chat_message(speaker, avatar="human")
        message.markdown(
            f'<div style="background-color:{color}; padding: 10px; border-radius: 5px;">'
            f"<strong>{speaker}:</strong> {text}</div>",
            unsafe_allow_html=True
        )

        # Check if this utterance has an emotion-cause pair
        if utterance_id in cause_lookup:
            cause_info = cause_lookup[utterance_id]
            cause_utterance_id, start_idx, end_idx = cause_info.split('_')
            start_idx, end_idx = int(start_idx), int(end_idx)

            # Find the cause utterance
            cause_utterance = next(
                (u for u in conversation_json if str(u["utterance_ID"]) == cause_utterance_id),
                None
            )

            if cause_utterance:
                cause_text = cause_utterance["text"]
                cause_speaker = cause_utterance["speaker"]
                # Highlight the cause span in red and italicize the rest
                highlighted_cause_text = (
                    f"{cause_text[:start_idx]}"
                    f":red[{cause_text[start_idx:end_idx]}]"
                    f"{cause_text[end_idx:]}"
                )
                message.markdown(
                    f"  ***{cause_speaker}:*** *{highlighted_cause_text}*",
                    unsafe_allow_html=True
                )


def process_bulk_input(bulk_text):
    conversation = []
    lines = bulk_text.strip().split("\n")
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            speaker, utterance = line.split(' ', 1)
            speaker = speaker.strip('[]')
            utterance = utterance.strip('"')
            conversation.append({
                "utterance_ID": idx + 1,
                "text": utterance,
                "speaker": speaker
            })
        except ValueError:
            st.error(f"Invalid format on line {idx + 1}: {line}")
    return conversation


# Streamlit application
st.title("Conversation Input Demo")

st.subheader("Option 1: Step-by-Step Input")
# Step-by-step input form
if "step_conversation" not in st.session_state:
    st.session_state.step_conversation = []

with st.form("step_form", clear_on_submit=True):
    speaker = st.text_input("Speaker", key="step_speaker")
    utterance = st.text_input("Utterance", key="step_utterance")
    submitted = st.form_submit_button("Add")
    if submitted:
        if speaker and utterance:
            st.session_state.step_conversation.append({
                "utterance_ID": len(st.session_state.step_conversation) + 1,
                "text": utterance,
                "speaker": speaker
            })
        else:
            st.error("Both speaker and utterance must be provided.")

if st.session_state.step_conversation:
    st.write("Current Conversation:")
    st.json(st.session_state.step_conversation)
    if st.button("Run Inference on Step-by-Step Conversation"):
        result = inference(st.session_state.step_conversation)

        #display_chat(result["conversation"])
        display_chat(dummy_output_data["conversation"], dummy_output_data["emotion-cause_pairs"])

st.subheader("Option 2: Bulk Input")
# Bulk input form
bulk_input = st.text_area(
    "Enter the conversation in the format: [person_name] \"utterance\"",
    placeholder="[Ross] \"It is funny , my birthday was seven months ago .\"\n[Joey] \"So ?\""
)
if st.button("Process and Run Inference on Bulk Input"):
    if bulk_input.strip():
        conversation = process_bulk_input(bulk_input)
        if conversation:
            st.write("Parsed Conversation:")
            st.json(conversation)
            result = inference(conversation)
            #display_chat(result["conversation"])
            display_chat(dummy_output_data["conversation"], dummy_output_data["emotion-cause_pairs"])
    else:
        st.error("Please enter some text for bulk input.")
