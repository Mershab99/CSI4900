from openai import OpenAI

# Set up your OpenAI API key
api_key = ""
client = OpenAI(api_key=api_key)

# Define your fine-tuned model's ID
fine_tuned_model_id = "ft:gpt-4o-mini-2024-07-18:personal:emotion-v2:ANj5R7eY"  # replace the fine-tuned model ID

# Define the two dialog utterances
utterance_1 = "Do not worry . I imagine he would be okay with you because really he is okay with Ethan ."
utterance_2 = "Ethan ? There is , there is an Ethan ?"

# Prepare the prompt or input for your model
prompt_text = f"Take a deep breath. Your task: given the following two dialog utterances, predict the emotion of the second utterance. Select the emotion from the following options: neutral, anger, disgust, fear, joy, sadness, surprise. Do not use any other emotions!!! Respond only with the chosen emotion, without any additional explanation. Remember that you can only use listed emotions!!!\n\nUtterance 1: {utterance_1}\nUtterance 2: {utterance_2}"

# Make the API call to your fine-tuned model
messages = [{"role": "system", "content": prompt_text}]
response = client.chat.completions.create(
    model=fine_tuned_model_id,
    messages=messages,
    temperature=0      # Adjust as needed
)

# Output the response
print(response.choices[0].message.content)
