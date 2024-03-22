import ollama

while True:
    user_input = input("Human: ")

    stream = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': user_input}], 
        stream=True
    )

    for chunk in stream:
        if 'content' in chunk['message']:
            print(chunk['message']['content'], end='', flush=True)
    print("\n")
