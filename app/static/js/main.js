document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const conversationList = document.getElementById('conversation-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const currentConvIdInput = document.getElementById('current-conversation-id');
    const attachmentBtn = document.getElementById('attachment-btn');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const emptyState = document.getElementById('empty-state');
    const interruptionAlert = document.getElementById('interruption-alert');
    const questionText = document.getElementById('question-text');
    const statusBadge = document.getElementById('status-badge');

    // --- Event Listeners ---

    attachmentBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        filePreview.innerHTML = '';
        Array.from(fileInput.files).forEach(file => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-secondary';
            badge.textContent = file.name;
            filePreview.appendChild(badge);
        });
    });

    newChatBtn.addEventListener('click', () => {
        currentConvIdInput.value = '';
        chatMessages.innerHTML = '';
        chatMessages.appendChild(emptyState);
        emptyState.style.display = 'block';
        document.getElementById('chat-title').textContent = 'New Consultation';
        interruptionAlert.classList.add('d-none');
        statusBadge.textContent = 'Ready';
        statusBadge.className = 'badge bg-info text-dark';

        // Remove active class from list
        document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
    });

    document.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const id = item.dataset.id;
            loadConversation(id);

            // Highlight active
            document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
        });
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = messageInput.value.trim();
        const files = fileInput.files;

        if (!message && files.length === 0) return;

        // Optimistic UI update
        if (message) {
            appendMessage('user', message);
        }

        // Prepare FormData
        const formData = new FormData(chatForm);

        // Clear input
        messageInput.value = '';
        fileInput.value = ''; // Reset file input
        filePreview.innerHTML = '';

        if (emptyState) emptyState.style.display = 'none';

        // Show loading state
        statusBadge.textContent = 'Processing...';
        statusBadge.className = 'badge bg-warning text-dark';
        interruptionAlert.classList.add('d-none');

        try {
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                if (data.status === 'completed') {
                    appendMessage('assistant', data.response);
                    statusBadge.textContent = 'Completed';
                    statusBadge.className = 'badge bg-success';
                } else if (data.status === 'interrupted') {
                    // Handle system question
                    appendMessage('system_question', data.question);
                    interruptionAlert.classList.remove('d-none');
                    questionText.textContent = data.question;
                    messageInput.placeholder = "Please answer the question above...";
                    messageInput.focus();

                    statusBadge.textContent = 'Waiting for input';
                    statusBadge.className = 'badge bg-danger';
                }

                if (data.conversation_id) {
                    currentConvIdInput.value = data.conversation_id;
                    // TODO: Update or add to conversation list if new
                }
            } else {
                appendMessage('error', data.error || 'An error occurred.');
                statusBadge.textContent = 'Error';
                statusBadge.className = 'badge bg-danger';
            }
        } catch (error) {
            console.error('Error:', error);
            appendMessage('error', 'Network error. Please try again.');
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge bg-danger';
        }
    });

    // --- Helper Functions ---

    async function loadConversation(id) {
        try {
            const res = await fetch(`/api/conversations/${id}`);
            const data = await res.json();

            if (res.ok) {
                currentConvIdInput.value = data.id;
                document.getElementById('chat-title').textContent = data.title;
                chatMessages.innerHTML = '';

                if (data.messages && data.messages.length > 0) {
                    data.messages.forEach(msg => {
                        appendMessage(msg.role, msg.content);
                    });
                } else {
                    chatMessages.appendChild(emptyState);
                }

                // Check if last message was a system question (interrupted state)
                const lastMsg = data.messages[data.messages.length - 1];
                if (lastMsg && lastMsg.role === 'system_question') {
                     interruptionAlert.classList.remove('d-none');
                     questionText.textContent = lastMsg.content;
                     statusBadge.textContent = 'Waiting for input';
                     statusBadge.className = 'badge bg-danger';
                } else {
                     interruptionAlert.classList.add('d-none');
                     statusBadge.textContent = 'Ready';
                     statusBadge.className = 'badge bg-info text-dark';
                }
            }
        } catch (err) {
            console.error(err);
        }
    }

    function appendMessage(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `d-flex mb-3 ${role === 'user' ? 'justify-content-end' : 'justify-content-start'}`;

        let cardClass = role === 'user' ? 'bg-primary text-white' : 'bg-white border';
        if (role === 'system_question') cardClass = 'bg-warning text-dark';
        if (role === 'error') cardClass = 'bg-danger text-white';

        // Markdown-like simple formatting (newlines)
        const formattedContent = content.replace(/\n/g, '<br>');

        msgDiv.innerHTML = `
            <div class="card ${cardClass}" style="max-width: 75%;">
                <div class="card-body p-2">
                    ${formattedContent}
                </div>
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
