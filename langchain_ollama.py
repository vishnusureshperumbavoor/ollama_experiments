from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

try:
    while True:
        human_input = input("Human: ")
        response = llm.invoke(human_input)
        print(response)

except KeyboardInterrupt:
    print("\n")