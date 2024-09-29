def predict_emotions_and_causes(conversation, model, tokenizer):
    tokenized_conversation = tokenize_utterances(conversation, tokenizer)

    input_ids = torch.cat([data['input_ids'] for data in tokenized_conversation])
    attention_mask = torch.cat([data['attention_mask'] for data in tokenized_conversation])

    model.eval()
    with torch.no_grad():
        emotion_preds, cause_preds = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert predictions back to labels
    predicted_emotions = torch.argmax(emotion_preds, dim=-1)
    return predicted_emotions, cause_preds


# Example prediction
predicted_emotions, predicted_causes = predict_emotions_and_causes(conversation, model, tokenizer)
print(predicted_emotions, predicted_causes)
