from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize RAG components
db_path = "user_4"  # Update this path
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def get_relevant_info(query):
    result = qa.invoke(query)
    result_text = result['result'] if 'result' in result else ""
    
    # Write query and result to a file
    with open("query_result.txt", "w") as file:
        file.write(f"Query: {query}\n\nResult: {result_text}\n\n")
    
    return result_text

def chat_with_gpt(user_input, user_id):
    # Create user-specific folder if it doesn't exist
    user_folder = f"memory_{user_id}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Initialize or load the conversation history
    history_file = os.path.join(user_folder, f"memory_{user_id}.txt")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            chat_with_gpt.messages = eval(f.read())
    else:
        chat_with_gpt.messages = [
            {"role": "system", "content": """
            You are an AI agent integrated into a website that helps people make meaningful connections from their personal notes about knowledge topics.
            Your primary goal is to understand the user pedagogical preferences from their answers :
             - what kind of information does he likes (eg : specific element from a note or link between notes)?
             - how does he want this information to be presented to him ?

            The most important thing is to adapt to the user. Like the way he talks, and talks about things.

            Be subtle in your approach. Don't ask these questions directly, but rather engage the user in a conversation that naturally reveals this information.
            
            You have access to the user's notes. Use this information to provide relevant examples and insights that align with the user's interests and learning style.
            Always tie your responses back to the user's preferences.

            Keep your responses concise but informative, don't repeat what the user just said. Aim to spark curiosity and encourage further exploration.
            """}
        ]
        # Add an initial question from the assistant
        initial_question = "Hello! I'm here to help you make the most of your personal notes and enhance your learning journey. To get started, could you tell me about a recent topic you've been exploring in your notes?"
        chat_with_gpt.messages.append({"role": "assistant", "content": initial_question})

    # Get relevant information from user's notes
    relevant_info = get_relevant_info(user_input)

    # Combine user input with relevant information for context
    enhanced_input = f"User input: {user_input}\nRelevant information from notes: {relevant_info}"

    # Add the enhanced user message to the conversation history
    chat_with_gpt.messages.append({"role": "user", "content": enhanced_input})

    # Send the conversation to the OpenAI API
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_with_gpt.messages,
        temperature=0.7
    )

    # Extract the assistant's reply
    assistant_reply = chat_completion.choices[0].message.content

    # Add the assistant's reply to the conversation history
    chat_with_gpt.messages.append({"role": "assistant", "content": assistant_reply})

    # Save the updated conversation history to the file
    with open(history_file, 'w') as f:
        f.write(str(chat_with_gpt.messages))

    return assistant_reply

# Example usage
if __name__ == "__main__":
    user_id = input("Enter your user ID: ")
    
    # Check if it's a new conversation
    user_folder = f"memory_{user_id}"
    history_file = os.path.join(user_folder, f"memory_{user_id}.txt")
    
    if not os.path.exists(history_file):
        print("Assistant: Hello! I'm here to help you make the most of your personal notes and enhance your learning journey. Tell me something that goes through your mind right now.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        response = chat_with_gpt(user_input, user_id)
        print("Assistant:", response)