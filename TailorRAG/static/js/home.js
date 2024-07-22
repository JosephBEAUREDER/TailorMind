const csrfToken = "{{ csrf_token }}";

///////////////////////////////////// SIDEBAR - LEFT ////////////////////////////////////////

// show texts titles in sidebar
function showTextsSidebar() {
    fetch('/get_texte_titles/')
        .then(response => response.json())
        .then(data => {
            const texteItems = document.getElementById('texteItems');
            texteItems.innerHTML = ''; // Clear existing items
            data.titles.forEach(title => {
                const li = document.createElement('li');
                li.textContent = title;
                texteItems.appendChild(li);
            });
        })
        .catch(error => console.error('Error:', error));
    }

// Initial load of Texte items
showTextsSidebar();

// show the window to add text
document.getElementById('addText').addEventListener('click', function() {
    // Show Bootstrap modal
    $('#modalAddText').modal('show');
});

// Function to show temporary alert
function showTemporaryAlert(message, duration = 1000) {
    const alertElement = document.createElement('div');
    alertElement.textContent = message;
    alertElement.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        z-index: 1000;
    `;
    document.body.appendChild(alertElement);
    setTimeout(() => {
        alertElement.remove();
    }, duration);
}

// save a text in database
document.getElementById('saveText').addEventListener('click', function() {
    const title = document.getElementById('newTitle').textContent;
    const text = document.getElementById('newText').value;

    fetch("/save_texte/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie('csrftoken')
        },
        body: JSON.stringify({
            title: title,
            text: text
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response from server:', data);
        if (data.success) {
            console.log('Success message:', data.message);
            showTextsSidebar(); // Refresh the list
            showTemporaryAlert('Text successfully saved');
            $('#modalAddText').modal('hide');  // Close the modal
        } else {
            console.log('Error message:', data.message);
            if (data.errors) {
                console.log('Form errors:', data.errors);
                let errorMessage = 'Form validation failed:\n';
                for (const [field, errors] of Object.entries(data.errors)) {
                    errorMessage += `${field}: ${errors.join(', ')}\n`;
                }
                alert(errorMessage);
            } else {
                alert('Failed to save texte: ' + data.message);
            }
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        alert('An error occurred while saving the texte.');
    });
});





///////////////////////////////////// CHATBOX ////////////////////////////////////////

const chatContainer = document.getElementById('chat-container');
const queryBtn = document.getElementById('queryBtn');

function addMessage(message, isUser) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.classList.add(isUser ? 'user-message' : 'bot-message');
    messageElement.textContent = message;
    chatContainer.prepend(messageElement);
}

queryBtn.addEventListener('click', function() {
    const query = document.getElementById('queryInput').value;
    if (query) {

        addMessage(query, true);
        queryInput.value = '';

        addMessage('Processing query...', false);

        fetch('/query_rag/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({query: query})
        })
        .then(response => response.json())
        .then(data => {
            // Remove the "Processing query..." message
            chatContainer.removeChild(chatContainer.firstChild);

            if (data.error) {
                addMessage('Error: ' + data.error, false);
            } else {
                addMessage(data.result, false);
            }
        })
        .catch(error => {
            console.error('Error from server:', error);
            // Remove the "Processing query..." message
            chatContainer.removeChild(chatContainer.firstChild);
            addMessage('An error occurred while processing your query.', false);
        });
    }
});

queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        queryBtn.click();
    }
});


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}








///////////////////////////////////// CHATBOX-RIGHT ////////////////////////////////////////


document.addEventListener('DOMContentLoaded', function() {
    const chatContainerRight = document.getElementById('chat-container-right');
    const queryInputRight = document.getElementById('queryInputRight');
    const queryBtnRight = document.getElementById('queryBtnRight');

    function addMessageRight(message, isUser) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message-right');
        messageElement.classList.add(isUser ? 'user-message-right' : 'bot-message-right');
        messageElement.textContent = message;
        chatContainerRight.prepend(messageElement);
    }

    queryBtnRight.addEventListener('click', function() {
        const query = queryInputRight.value.trim();
        if (query) {
            addMessageRight(query, true);
            queryInputRight.value = '';

            addMessageRight('Processing query...', false);

            // Here you would typically send the query to your backend
            // For demonstration, we'll just simulate a response after a delay
            setTimeout(() => {
                chatContainerRight.removeChild(chatContainerRight.firstChild); // Remove 'Processing' message
                addMessageRight('This is a simulated response to: ' + query, false);
            }, 1000);
        }
    });

    // Allow sending message with Enter key
    queryInputRight.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            queryBtnRight.click();
        }
    });
});