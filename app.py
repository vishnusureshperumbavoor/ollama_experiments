import ollama

try:
    while True:
        human_input = input("Human: ")

        stream = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': human_input}], 
            stream=True
        )

        for chunk in stream:
            if 'content' in chunk['message']:
                print(chunk['message']['content'], end='', flush=True)
        print("\n")

except KeyboardInterrupt:
    print("\n")
