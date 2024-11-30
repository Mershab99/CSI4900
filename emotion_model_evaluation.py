import json

def extract_labels_and_predictions(true_json_file: str, predicted_json_file: str):
    """
    Extract true labels and predictions from JSON files, ensuring alignment by matching conversation IDs.
    
    Args:
        true_json_file (str): The path to the true labels JSON file (e.g., Subtask_1_test.json).
        predicted_json_file (str): The path to the predicted labels JSON file (e.g., annotated_test_data.json).
    
    Returns:
        tuple: A tuple containing two lists: true_labels and predictions.
    """
    # Read true labels from the file
    with open(true_json_file, "r") as f:
        true_data = json.load(f)
    
    # Read predicted labels from the file
    with open(predicted_json_file, "r") as f:
        predicted_data = json.load(f)
    
    # Extract conversation IDs
    true_conversation_ids = {dialog["conversation_ID"] for dialog in true_data}
    predicted_conversation_ids = {dialog["conversation_ID"] for dialog in predicted_data}
    
    # Find common conversation IDs
    common_conversation_ids = true_conversation_ids & predicted_conversation_ids
    
    # Filter true and predicted data by common conversation IDs
    true_data_filtered = [dialog for dialog in true_data if dialog["conversation_ID"] in common_conversation_ids]
    predicted_data_filtered = [dialog for dialog in predicted_data if dialog["conversation_ID"] in common_conversation_ids]
    
    # Extract labels and predictions
    true_labels = []
    predictions = []
    
    for true_dialog, pred_dialog in zip(true_data_filtered, predicted_data_filtered):
        # Ensure the conversations are aligned by ID
        if true_dialog["conversation_ID"] == pred_dialog["conversation_ID"]:
            for true_turn, pred_turn in zip(true_dialog["conversation"], pred_dialog["conversation"]):
                # Assuming true labels and predictions are stored in the "emotion" key
                true_labels.append(true_turn.get("emotion", "unknown"))  # Default to "unknown" if missing
                predictions.append(pred_turn.get("emotion", "unknown"))  # Default to "unknown" if missing
    
    return true_labels, predictions

# Example usage
true_labels, predictions = extract_labels_and_predictions(
    "data/labeled_testing_data.json", 
    "data/predicted_testing_data.json"
)

# Print the first few labels to verify the extraction
# print("True Labels:", true_labels)
# print("Predictions:", predictions)

def write_labels_to_file_as_list(true_labels, predictions, output_file):
    """
    Write true labels and predictions to a file in list format.

    Args:
        true_labels (list): List of true labels.
        predictions (list): List of predicted labels.
        output_file (str): Path to the output file.

    Returns:
        None
    """
    with open(output_file, "w") as f:
        f.write("True Labels:\n")
        f.write(str(true_labels) + "\n\n")  # Write true labels in list format
        
        f.write("Predictions:\n")
        f.write(str(predictions) + "\n")  # Write predictions in list format

# Write to file
write_labels_to_file_as_list(true_labels, predictions, "data/output_labels_predictions.txt")
