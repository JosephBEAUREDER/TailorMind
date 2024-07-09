from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
            You are an AI assistant integrated in a website that helps people to make meaningful connections from their personal notes about knowledge topics.
            Your goal is to have the answer to such questions:
            - Why is he/she using this website?
            - What are the motivations of the user to learn these notes better?
            - What interests him most about these notes?
            - How would he like an AI to extract information from these notes in order to give them to him?
            
            But you have to be subtle, don't ask these questions straightforward because it is hard for the user to answer directly to them.
            Act more like a psychologist who deals with these subjects in a roundabout way, making it easy and fun for the user to respond.
            Short answers from you are better, don't repeat what the user just said.
            """}
        ]
        # Add an initial question from the assistant
        initial_question = "Hello! I'm here to help you make meaningful connections from your personal notes. What kind of topics are you most interested in exploring?"
        chat_with_gpt.messages.append({"role": "assistant", "content": initial_question})

    # Add the user's message to the conversation history
    chat_with_gpt.messages.append({"role": "user", "content": user_input})

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
        print("Assistant: Hello! I'm here to help you make meaningful connections from your personal notes. What kind of topics are you most interested in exploring?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        response = chat_with_gpt(user_input, user_id)
        print("Assistant:", response)