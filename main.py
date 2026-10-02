import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Gemini via Fastly ARC (using OpenAI compatibility)
llm = ChatOpenAI(
    model="gemini-3.6-flash",
    base_url="https://arc.fastly.app/v1",
    api_key=os.getenv("ARC_API_KEY"),
    temperature=0.0,
)

agent = create_agent(
    model=llm,
    system_prompt=(
        "You are a Fastly VCL expert with a knack for explaining stuff in"
        " details and mention potential risks while changing VCL configs. You"
        " explain things in a max of 300 words. You have the personality of"
        " pitbull aka mr. worldwide."
    ),
    checkpointer=InMemorySaver(),
)

critique = PromptTemplate.from_template(
    "Critique the following explanation: {result} in just 10 words, give it a"
    " score from 0 to 10 and in less than 100 words explain what is missing"
    " for the 10. You are a Fastly VCL engineer."
)
parser = StrOutputParser()
critique_chain = critique | llm | parser

while True:
    topic = input("Enter the topic you want explained: ")

    print("*********************")

    full_response = ""
    # stream_mode="messages" yields tuples of (message_chunk, metadata)
    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": topic}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages",
    ):
      # Check if the chunk contains text content
      if chunk.content:
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print("\n*********************")

    # Now run the critique using the collected full response
    critique_result = critique_chain.invoke({"result": full_response})
    print(critique_result)
    print("\n")
    # Ask the user if they want to continue or exit
    continue_choice = input("Do you want to explain another topic? (yes/no): ")
    if continue_choice.lower() != "yes":
        break