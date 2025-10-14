
import pandas as pd
from openai import OpenAI
import os
# Load the CSV file

file_path = 'meld_train_with_answers.xlsx'
df = pd.read_excel(file_path)
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


# Define the emotion-based prompts
prompts = {
    "neutral": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is neutral. Please provide the most appropriate response based on this information and the dialogue content, maintaining a calm and objective demeanor.",
    "disgust": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is disgust. Please provide a supportive and understanding response based on this information and the dialogue content to help them alleviate their emotions.",
    "anger": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is angry. Please respond with calmness and empathy based on this information and the dialogue content to help them soothe their feelings.",
    "sadness": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is sad. Please provide comfort and encouragement based on this information and the dialogue content to help them feel better.",
    "joy": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is happy. Please provide positive feedback and praise based on this information and the dialogue content to enhance their sense of joy.",
    "surprise": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is surprised. Please respond by expressing understanding and interest based on this information and the dialogue content in their surprise.",
    "fear": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone is fearful. Please provide comfort and reassurance based on this information and the dialogue content to help them alleviate their fear.",
    "Other": "You are an emotional dialogue robot. There is a conversation with the content: ____.  The speaker's tone represents another emotion. Please provide an appropriate response based on this information and the dialogue content to meet the speaker's emotional needs."
}

# Function to generate a response based on emotion
def generate_response(row):
    emotion = row["Emotion"]
    content = row["Utterance"]
    
    
    # Select the appropriate prompt template
    prompt = prompts.get(emotion, prompts["Other"])
    
    # Fill in the content and emotion dimensions in the prompt
    filled_prompt = prompt.replace("____", content)
    client = OpenAI(
    # This is the default and can be omitted
    api_key="",)
    # Call OpenAI's GPT model
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": filled_prompt}]
    )
    
    return response.choices[0].message.content
# Initialize an empty list to collect answers
answers = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    try:
        if pd.notna(row['answer']):
            print(f"Skipping row {index + 1}, answer already exists: {row['answer']}")
            continue
        answer = generate_response(row)
        answers.append(answer)
        df.at[index, 'answer'] = answer  # Save answer in DataFrame
        print(f"Processed row {index + 1}/{len(df)}: {answer}")  # Output the current answer
        df.to_excel('meld_train_with_answers.xlsx', index=False)
        
    except Exception as e:
        print(f"Error processing row {index + 1}: {e}")
        # Export the current state of the DataFrame to Excel if an error occurs
        df.to_excel('meld_train_with_answers.xlsx', index=False)
        break  # Stop further processing if an error occurs
# If no errors occur, export the final DataFrame to Excel
else:
    df.to_excel('meld_train_with_answers.xlsx', index=False)