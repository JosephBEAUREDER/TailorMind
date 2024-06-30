from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
import json
from .forms import TexteForm

def home(request):
    return render(request,'home.html')
    
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')  # Redirect to a home page or any other page after registration
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


@login_required
def save_texte(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        form = TexteForm(data)
        if form.is_valid():
            texte = form.save(commit=False)
            texte.user = request.user
            texte.save()
            return JsonResponse({'success': True})
    return JsonResponse({'success': False})



################################
# OPEN_AI RAG

# import os
# from django.http import JsonResponse
# from django.views.decorators.http import require_POST
# from django.contrib.auth.decorators import login_required
# from django.conf import settings


# @login_required
# @require_POST
# def create_user_db(request):
#     user = request.user
#     db_path = os.path.join(settings.BASE_DIR, 'user_vectorstores', f'user_{user.id}')
    
#     try:
#         os.makedirs(db_path, exist_ok=True)
        
#         # Here, you would call your create_and_persist_db function
#         # For this example, we'll just create a dummy file to simulate database creation
#         with open(os.path.join(db_path, 'db_created.txt'), 'w') as f:
#             f.write('Database created')
        
#         return JsonResponse({
#             'message': f'Database created successfully for user {user.username}',
#             'path': db_path
#         })
#     except Exception as e:
#         return JsonResponse({
#             'message': f'Error creating database: {str(e)}',
#         }, status=500)
        

# @login_required
# def check_user_db(request):
#     user = request.user
#     db_path = os.path.join(settings.BASE_DIR, 'user_vectorstores', f'user_{user.id}')
#     db_exists = os.path.exists(db_path) and os.path.isfile(os.path.join(db_path, 'db_created.txt'))
    
#     return JsonResponse({
#         'database_exists': db_exists,
#         'path': db_path if db_exists else None
#     })








import os
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.conf import settings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# # Make sure to set your OpenAI API key in Django settings
# os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

def get_user_db_path(user):
    """Generate a unique path for each user's vector store."""
    return os.path.join(settings.BASE_DIR, 'user_vectorstores', f'user_{user.id}')

def create_and_persist_db(user):
    """Create and persist a Chroma vector store for a specific user."""
    persist_directory = get_user_db_path(user)
    os.makedirs(persist_directory, exist_ok=True)

    # Load and process the documents
    document_path = os.path.join(settings.BASE_DIR, 'documents', 'Bouveresse.txt')
    loader = TextLoader(document_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create the vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    return persist_directory

@login_required
@require_POST
def create_user_db(request):
    user = request.user
    try:
        db_path = create_and_persist_db(user)
        return JsonResponse({
            'message': f'Chroma database created successfully for user {user.username}',
            'path': db_path
        })
    except Exception as e:
        return JsonResponse({
            'message': f'Error creating database: {str(e)}',
        }, status=500)

@login_required
def check_user_db(request):
    user = request.user
    db_path = get_user_db_path(user)
    db_exists = os.path.exists(db_path) and os.path.isdir(db_path)
    
    return JsonResponse({
        'database_exists': db_exists,
        'path': db_path if db_exists else None
    })
    
    
@login_required
@require_POST
def query_rag(request):
    user = request.user
    data = json.loads(request.body)
    query = data.get('query')

    if not query:
        return JsonResponse({'error': 'No query provided'}, status=400)

    try:
        # Get the user's database path
        db_path = get_user_db_path(user)

        # Check if the database exists
        if not os.path.exists(db_path):
            return JsonResponse({'error': 'No database found for this user'}, status=404)

        # Load the persisted database
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Create the RAG chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Query the RAG system
        result = qa.invoke(query)

        return JsonResponse({'result': result['result']})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)