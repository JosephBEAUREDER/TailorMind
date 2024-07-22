from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.conf import settings

from .forms import TexteForm
from .models import Texte

import os
import json
import logging

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def home(request):
    return render(request,'home.html')
    



################################
# REGISTER / LOGIN / LOGOUT
################################

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create an empty database for the new user
            db_creation_response = create_empty_db(user)
            if db_creation_response.status_code == 200:
                login(request, user)
                return redirect('home')  # Redirect to a home page or any other page after registration
            else:
                # If database creation failed, delete the user and return an error
                user.delete()
                return render(request, 'register.html', {'form': form, 'error': db_creation_response.content})
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # Redirect to a home page or any other page after login
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('login')  # Redirect to the login page after logout
    return render(request, 'logout.html')

################################
# HTML LAYOUT
################################

def show_texte_sidebar(request):
    # Assuming the user is logged in
    user_texts = Texte.objects.filter(user=request.user)
    
    print(f"Number of texts: {user_texts.count()}")  # Debug print
    for text in user_texts:
        print(f"Text title: {text.title}")  # Debug print
    
    context = {
        'user_texts': user_texts,
        # ... other context variables ...
    }
    
    return render(request, 'home.html', context)

def get_texte_titles(request):
    if request.user.is_authenticated:
        titles = list(Texte.objects.filter(user=request.user).values_list('title', flat=True))
        return JsonResponse({'titles': titles})
    return JsonResponse({'titles': []})

################################
# OPEN_AI RAG
################################


def get_user_db_path(user):
    """Generate a unique path for each user's vector store."""
    return os.path.join(settings.BASE_DIR, 'user_vectorstores', f'user_{user.id}')


## CREATE EMPTY DATABASE 

def create_empty_db(user):
    """Create an empty Chroma vector store for a specific user."""
    try:
        persist_directory = get_user_db_path(user)
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create an empty vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        return JsonResponse({
            'message': f'Chroma database created successfully for user {user.username}',
            'path': persist_directory
        })
    except Exception as e:
        return JsonResponse({
            'message': f'Error creating database: {str(e)}',
        }, status=500)


## ADD TEXT TO DATABASE ###

@login_required
def save_texte(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        form = TexteForm(data)
        if form.is_valid():
            texte = form.save(commit=False)
            texte.user = request.user
            texte.save()
            
            try:
                # Add the new Texte to the user's ChromaDB database
                add_texte_to_db(request.user, texte)
                # create_and_persist_db(request.user)
                return JsonResponse({'success': True, 'message': f'Added the text "{texte.title}" to the database'})
            except Exception as e:
                # If there's an error adding to ChromaDB, we should handle it
                return JsonResponse({'success': False, 'message': f'Error adding "{texte.title}" to the database: {str(e)}'}, status=500)
        else:
            # If the form is not valid, return the form errors
            return JsonResponse({'success': False, 'message': 'Form validation failed', 'errors': form.errors}, status=400)
    
    # If the request method is not POST
    return JsonResponse({'success': False, 'message': 'Invalid request method'}, status=405)

def add_texte_to_db(user, texte):
    """Add a new Texte to the user's existing Chroma database."""
    persist_directory = get_user_db_path(user)
    
    # Check if the database exists
    if not os.path.exists(persist_directory):
        # If it doesn't exist, create it
        create_empty_db(user)
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n"])
    
    # Create a Document object and split it
    document = Document(page_content=texte.text, metadata={"title": texte.title})
    texts = text_splitter.split_documents([document])
    
    # Extract the text content and metadata from the split documents
    text_contents = [doc.page_content for doc in texts]
    metadatas = [doc.metadata for doc in texts]
    
    # Add chunk index to metadata
    for i, metadata in enumerate(metadatas):
        metadata["chunk_index"] = i

    # Load the existing database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Add the new texts
    vectorstore.add_texts(
        texts=text_contents,
        metadatas=metadatas
    )
    
    # Write the chunks to a text file
    chunks_file_path = os.path.join(persist_directory, f"chunks_{texte.title}.txt")
    with open(chunks_file_path, 'w') as chunks_file:
        for i, chunk in enumerate(text_contents):
            chunks_file.write(f"Chunk {i}:\n{chunk}\n\n{'-'*50}\n\n")
    
def create_and_persist_db(user):
    """Create and persist a Chroma vector store for a specific user."""
    persist_directory = get_user_db_path(user)
    os.makedirs(persist_directory, exist_ok=True)

    # Load and process the documents
    document_path = os.path.join(settings.BASE_DIR, 'documents', 'Bouveresse.txt')
    loader = TextLoader(document_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create the vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    return persist_directory

@login_required
def check_user_db(request):
    user = request.user
    db_path = get_user_db_path(user)
    db_exists = os.path.exists(db_path) and os.path.isdir(db_path)
    
    return JsonResponse({
        'database_exists': db_exists,
        'path': db_path if db_exists else None
    })
    


############# QUERY RAG #############

# Load environment variables from .env file
load_dotenv()

from .models import ChatHistory
@login_required
@require_POST
def query_rag(request):
    user = request.user
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)

        data = json.loads(request.body)
        query = data.get('query')

        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)

        db_path = get_user_db_path(user)

        if not os.path.exists(db_path):
            return JsonResponse({'error': 'No database found for this user'}, status=404)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Define the system prompt
        system_prompt = """You are a helpful AI assistant. Your task is to provide accurate and relevant information based on the context provided. If you're unsure about something, please say so. Your answer must be as short as possible"""

        # Create a custom prompt template
        prompt_template = """
        {context}

        Human: {question}
        AI: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Get the result and source documents
        result = qa({"query": query})
        
        # Retrieve the last 5 interactions for this user
        last_interactions = ChatHistory.objects.filter(user=user).order_by('-timestamp')[:5]
        
        # Construct the full prompt
        full_prompt = f"System: {system_prompt}\n\n"

        for i, doc in enumerate(result['source_documents']):
            full_prompt += f"Relevant Chunk {i+1} from {doc.metadata}:\n{doc.page_content}\n\n"
        full_prompt += f"Human: {query}\n"
        full_prompt += f"AI: {result['result']}"
        
        ChatHistory.objects.create(
            user=user,
            query=query,
            result=result['result']
        )

        # Write the full prompt to a text file
        log_file_path = os.path.join(settings.BASE_DIR, 'user_vectorstores', f'user_{user.id}', f"logs/query_log_{user.id}.txt")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'a') as log_file:
            log_file.write(full_prompt + "\n\n" + "-"*50 + "\n\n")

        return JsonResponse({'result': result['result']})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
    

@login_required
@require_POST
def reflexive_chatbot(request):
    try:
        # Get the OpenAI API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)

        # Parse the incoming JSON data
        data = json.loads(request.body)
        user_input = data.get('query')

        if not user_input:
            return JsonResponse({'error': 'No query provided'}, status=400)

        # Initialize the language model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant with reflexive capabilities. You can analyze your own responses and thoughts."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Initialize the memory
        memory = ConversationBufferMemory(return_messages=True)

        # Create the conversation chain
        conversation = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm
        )

        # Generate the response
        response = conversation.predict(input=user_input)

        # Add a reflexive component
        reflection = conversation.predict(input=f"Reflect on your previous response: '{response}'. How could you improve it?")

        # Combine the original response and reflection
        final_response = f"Response: {response}\n\nReflection: {reflection}"

        return JsonResponse({'result': final_response})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)