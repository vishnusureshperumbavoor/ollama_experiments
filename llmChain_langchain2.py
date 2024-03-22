from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot which uses lot of curse words while talking to a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

try:
    while True:
        chat_llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        human_input = input("Human: ")

        print(chat_llm_chain.predict(human_input=human_input))

except KeyboardInterrupt:
    print("\n")