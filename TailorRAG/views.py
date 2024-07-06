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

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI

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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
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

logger = logging.getLogger(__name__)

@login_required
@require_POST
def query_rag(request):
    user = request.user
    try:
        # Get the OpenAI API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return JsonResponse({'error': 'OpenAI API key not configured'}, status=500)

        data = json.loads(request.body)
        query = data.get('query')

        if not query:
            logger.warning(f"No query provided for user {user.username}")
            return JsonResponse({'error': 'No query provided'}, status=400)

        logger.info(f"Processing query for user {user.username}: {query[:50]}...")

        # Get the user's database path
        db_path = get_user_db_path(user)
        logger.info(f"Database path for user {user.username}: {db_path}")

        # Check if the database exists
        if not os.path.exists(db_path):
            logger.error(f"No database found for user {user.username} at path {db_path}")
            return JsonResponse({'error': 'No database found for this user'}, status=404)

        # Load the persisted database
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        logger.info(f"Loaded vector store for user {user.username}")

        # Create the RAG chain
        qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        logger.info(f"Created RAG chain for user {user.username}")

        # Query the RAG system
        result = qa.invoke(query)
        logger.info(f"Obtained result for user {user.username}")

        return JsonResponse({'result': result['result']})

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for user {user.username}: {str(e)}")
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        logger.error(f"Error processing query for user {user.username}: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)