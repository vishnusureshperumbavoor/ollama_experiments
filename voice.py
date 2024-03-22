import ollama
import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Function to speak the response
def speak_response(response):
    engine.say(response)
    engine.runAndWait()

# Get user input
user_input = input("Enter your message: ")

# Chat with Mistral
stream = ollama.chat(
    model='mistral', 
    messages=[{'role': 'user', 'content': user_input}], 
    stream=True
)

# Iterate over the response stream
for chunk in stream:
    if 'content' in chunk['message']:
        response = chunk['message']['content']
        print(response, end='', flush=True)
        speak_response(response)
