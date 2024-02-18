import ollama


stream = ollama.chat(
    model='mistral', 
    messages=[{'role':'user', 'content':'Tell me about the history of AI?'}], 
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)