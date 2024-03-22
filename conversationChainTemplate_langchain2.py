from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

template = """You are an AI which is primarily focused on healthcare. Reply only to the questions related to healthcare. If human ask any question other than healthcare related, just say you don't know the answer

Current conversation:
{history}
Human: {input}
VSP AI:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

try:
    while True:
        conversation = ConversationChain(
            prompt=PROMPT,
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
        )

        human_input = input("Human: ")

        print(conversation.predict(input=human_input))

except KeyboardInterrupt:
    print("\n")