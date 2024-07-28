import ollama 

with open('assets/VSP.jpg', 'rb') as file:
    response = ollama.chat(
        model='llava',
        messages=[
            {
                'role':'user', 
                'content':'What is strange about this image?', 
                'images':[file.read()]
            }
        ]
    )

print(response['message']['content'])