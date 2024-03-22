from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

try:
    while True:
        conversation = ConversationChain(
            llm=llm, verbose=True, memory=ConversationBufferMemory()
        )

        human_input = input("Human: ")

        print(conversation.predict(input=human_input))

except KeyboardInterrupt:
    print("\n")