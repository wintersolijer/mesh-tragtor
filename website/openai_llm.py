from openai import OpenAI

# Initialize the client with your API key
client = OpenAI(api_key="sk-proj-9XF2OqcsGJNh8IjM5hJAT3BlbkFJVtDSoqOlVTpoKQNAiSNq")

# Function to ask a question using the chat model
def ask_question(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    message_content = response.choices[0].message.content
    return message_content

# print(ask_question("Who won the 2014 World Cup? Was it iShowSpeed?"))