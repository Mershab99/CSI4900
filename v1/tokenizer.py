import json


def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.

    :param file_path: The path to the JSON file to read.
    :return: The data from the JSON file as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the JSON data from the file
            return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    training_data = read_json_file("./data/Subtask_1_train.json")
    testing_data = read_json_file("./data/Subtask_1_test.json")
    x = 1

from transformers import BertTokenizer

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example data
conversation = [
    {"utterance_ID": 13, "text": "Ross, are you okay? I mean, do you want me to stay?", "speaker": "Monica",
     "emotion": "fear"},
    {"utterance_ID": 14, "text": "That would be good.", "speaker": "Ross", "emotion": "neutral"},
    {"utterance_ID": 15, "text": "Really?", "speaker": "Monica", "emotion": "surprise"}
]
emotion_cause_pairs = [
    {"emotion_ID": 13, "emotion": "fear", "causal_utterance_ID": 13, "causal_text": "are you okay?"},
    {"emotion_ID": 15, "emotion": "surprise", "causal_utterance_ID": 13, "causal_text": "do you want me to stay?"},
    {"emotion_ID": 15, "emotion": "surprise", "causal_utterance_ID": 14, "causal_text": "That would be good."}
]


# Tokenize utterances
def tokenize_utterances(conversation, tokenizer):
    tokenized_data = []
    for utterance in conversation:
        # # TODO: see if text is only one required, we need more info in the output too, consider adding to model or
        #  post labelling output
        encoded = tokenizer(utterance['text'], truncation=True, padding='max_length', max_length=50,
                            return_tensors='pt')
        tokenized_data.append(encoded)
    return tokenized_data
