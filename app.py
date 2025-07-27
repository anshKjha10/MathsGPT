import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Setup the Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon=":robot:")
st.title("Text to Math Problem Solver using Google Gemma2")

groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue.")
    st.stop()
    
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)


# Inititalizing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for answering questions about general knowledge. Input should be a question or topic.",
)

# Initialize the math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name = "Calculator",
    func=math_chain.run,
    description="Useful for solving mathematical problems. Input should be a math problem or equation."
)

prompt = """
You are an agent tasked for solving mathematical problems. Logically arrive at the solution step by step and provide detailed explainantion and display it point wise for the question below.
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning based questions. Input should be a question or topic.",
)

# Initialize the agent with the tools
assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hello! I am your Math Problem Solver and Data Search Assistant. How can I help you today?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    
# funct to generate response
# def generate_response(question):
#     response = assistant_agent.invoke({'input':question})
#     return response

# lets staart the interaction
question = st.text_area("Enter your question or math problem here:")
if st.button("Find My Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user", "content": question})
            st.chat_message("user").write(question)
            
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            st.session_state.messages.append({"role":"assistant", "content": response})
            st.write("### Response:")
            st.success(response)
            
    else:
        st.warning("Please enter a question or math problem to get an answer.")
            