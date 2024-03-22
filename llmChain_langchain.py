from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

template = """You are an AI which is primarily focused on finance. Reply only to the questions related to finance. If human ask any question other than finance related, just say you don't know the answer

Current conversation:
{history}
Human: {input}
VSP AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferMemory(memory_key="history")

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

try:
    while True:
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        human_input = input("Human: ")

        print(conversation.predict(input=human_input))

except KeyboardInterrupt:
    print("\n")