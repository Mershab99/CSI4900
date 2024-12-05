import json
import os

from openai import OpenAI
from tqdm.auto import tqdm

from server import HOME_DIR

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(f"{HOME_DIR}data/prompt.txt", "r") as f:
    COMMON_POMPT = f.read()


def read_json(file_name: str) -> list:
    """
    Read a JSON file and return its contents as a list.

    Args:
        file_name (str): The name of the JSON file to read.

    Returns:
        list: The contents of the JSON file as a list.
    """
    with open(file_name, "r") as f:
        return json.load(f)


def write_json(data, file_path):
    """
    Write data to a JSON file.

    Args:
        data (dict): The data to be written to the file.
        file_path (str): The path to the JSON file.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


# it is our fine-tuned model tag, you have to insert your own tag here,
# we cannot share our model due to the OpenAI policy
def get_completion(prompt: str, model: str = "ft:gpt-4o-mini-2024-07-18:personal:emotion-v2:ANj5R7eY") -> str:
    """
    Generates a completion based on the given prompt using the fine-tuned OpenAI GPT-4o mini model.

    Args:
        prompt (str): The prompt for generating the completion.
        model (str, optional): The model to use for generating the completion. Defaults to "ft:gpt-4o-mini-2024-07-18:personal:emotion-v2:ANj5R7eY".

    Returns:
        str: The generated completion.
    """
    messages = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_emotion_from_gpt(cause_edu_text: str, emotion_utterance_text: str) -> str:
    """
    Retrieves the emotion from GPT model based on the cause educational text and emotion utterance text.

    Args:
        cause_edu_text (str): The cause educational text.
        emotion_utterance_text (str): The emotion utterance text.

    Returns:
        str: The predicted emotion.
    """
    prompt = COMMON_POMPT.replace("UTT_1", cause_edu_text)
    prompt = prompt.replace("UTT_2", emotion_utterance_text)
    result = get_completion(prompt)
    return result


def emotion_prediction(dialogs: list) -> None:
    """
    Predicts emotions for each turn in a list of dialogs.
    Args:
        dialogs (list): A list of dialogs, where each dialog is a dictionary containing a "conversation" key.

    Returns:
        None
    """
    for dialog in tqdm(dialogs):
        for i, turn in enumerate(dialog["conversation"]):
            if i == 0:
                turn["emotion"] = get_emotion_from_gpt("", turn["text"])
            else:
                turn["emotion"] = get_emotion_from_gpt(dialog["conversation"][i - 1]["text"], turn["text"])
