import streamlit as st
import json



with open("demo/output.json", "r") as dummy_data_file:
    dummy_data = json.load(dummy_data_file)

dummy_output_data = dummy_data["conversation"]

# Dummy inference function
def inference(conversation_json):
    # For now, just return a placeholder JSON
    return {
        "conversation_ID": 1,
        "conversation": conversation_json,
        "emotion_cause_pairs": [["dummy_joy", "dummy_cause"]]
    }


# Function to display conversation as a chat
def display_chat(conversation_json):
    st.subheader("Conversation Chat")
    for utterance in conversation_json:
        # Simulate the conversation in chat bubbles
        message = st.chat_message(utterance["speaker"], avatar="human")

        message.markdown(f'''
            ***{utterance["speaker"]}:*** {utterance["text"]}
        ''')


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
        display_chat(dummy_output_data)

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
            display_chat(dummy_output_data)
    else:
        st.error("Please enter some text for bulk input.")
