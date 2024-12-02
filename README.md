# CSI4900: Textual Emotion-Cause Pair Extraction in Conversations: Cause Extraction

CSI 4900 Honours Project


## Team Members

- Ayoub El Aiboubi (300209549)
- Theeravich (Arthur) Trakulkajornsak (300192223)
- Mershab Issadien (300027272)

## Usage

Trains a cause-extractor model to predict emotion-cause pairs in conversations.

## Testing

Install dependencies:
```
pip install -r requirements.txt
```

`data/Subtask_1_train_real.json` is input to train the emotion-cause pair extraction model. 
Select the language model by setting the transformer constant in `src/data_processing.py`.
Hyperparameter weights can be adjusted in `src/training.py`.

Generate the model through running:
```
python src/training.py
```
The new model will be saved to `models/` 

The prediction requires emotion-annotated conversations such as that in `data/Subtask_1_test_gpt.json` as input
Running the following selects the model based on the transformer constant in `src/data_processing.py`.
```
python src/prediction.py
```

Predictions are saved to `data/` labelled with the model name