document.addEventListener('DOMContentLoaded', function () {
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

    // --- Markdown Parser ---
    function parseMarkdown(text) {
        let html = text;

        // Escape HTML
        html = html.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Headers (## and ###)
        html = html.replace(/^### (.+)$/gm, '<h3 class="mt-3 mb-2">$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2 class="mt-4 mb-3">$1</h2>');

        // Bold (**text**)
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Horizontal rules (***)
        html = html.replace(/^\*\*\*$/gm, '<hr class="my-3">');
        html = html.replace(/^---$/gm, '<hr class="my-3">');

        // Bullet lists (lines starting with * or -)
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

        // Wrap consecutive <li> in <ul>
        html = html.replace(/(<li>.*<\/li>\n?)+/g, function (match) {
            return '<ul class="mb-2">' + match + '</ul>';
        });

        // Paragraphs (double newlines)
        html = html.replace(/\n\n+/g, '</p><p class="mb-2">');
        html = '<p class="mb-2">' + html + '</p>';

        // Single newlines to <br>
        html = html.replace(/\n/g, '<br>');

        // Clean up empty paragraphs
        html = html.replace(/<p class="mb-2"><\/p>/g, '');

        return html;
    }

    // --- Event Listeners ---

    attachmentBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        filePreview.innerHTML = '';
        Array.from(fileInput.files).forEach(file => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-secondary me-1';
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

        document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
    });

    document.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const id = item.dataset.id;
            loadConversation(id);

            document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
        });
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = messageInput.value.trim();
        const files = fileInput.files;

        if (!message && files.length === 0) return;

        if (message) {
            appendMessage('user', message);
        }

        const formData = new FormData(chatForm);

        messageInput.value = '';
        fileInput.value = '';
        filePreview.innerHTML = '';

        if (emptyState) emptyState.style.display = 'none';

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

        // Apply markdown parsing for assistant messages
        const formattedContent = (role === 'assistant')
            ? parseMarkdown(content)
            : content.replace(/\n/g, '<br>');

        msgDiv.innerHTML = `
            <div class="card ${cardClass}" style="max-width: 75%;">
                <div class="card-body p-3">
                    ${formattedContent}
                </div>
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});