document.addEventListener('DOMContentLoaded', function () {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const conversationList = document.getElementById('conversation-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const currentConvIdInput = document.getElementById('current-conversation-id');
    const attachmentBtn = document.getElementById('attachment-btn');
    const fileInput = document.getElementById('file-input');
    const fileStagingArea = document.getElementById('file-staging-area');
    const emptyState = document.getElementById('empty-state');
    const interruptionAlert = document.getElementById('interruption-alert');
    const questionText = document.getElementById('question-text');
    const statusBadge = document.getElementById('status-badge');

    // State for staged files
    // Array of { id: string, file: File, customName: string, ext: string }
    let stagedFiles = [];

    // --- Markdown Parser ---
    function parseMarkdown(text) {
        if (!text) return '';
        let html = text;

        // Escape HTML
        html = html.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Horizontal rules
        html = html.replace(/^\*\*\*$/gm, '<hr>');
        html = html.replace(/^---$/gm, '<hr>');

        // Lists
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, match => `<ul>${match}</ul>`);

        // Paragraphs
        html = html.replace(/\n\n+/g, '</p><p>');
        html = '<p>' + html + '</p>';

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        // Cleanup
        html = html.replace(/<p><\/p>/g, '');

        return html;
    }

    // --- File Handling Functions ---

    attachmentBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        if (files.length === 0) return;

        files.forEach(file => {
            // Check limits (example: max 20 files total, max 10MB each)
            if (stagedFiles.length >= 20) {
                alert('Maximum 20 files allowed.');
                return;
            }
            if (file.size > 10 * 1024 * 1024) {
                alert(`File ${file.name} is too large (Max 10MB).`);
                return;
            }

            const lastDotIndex = file.name.lastIndexOf('.');
            const name = lastDotIndex !== -1 ? file.name.substring(0, lastDotIndex) : file.name;
            const ext = lastDotIndex !== -1 ? file.name.substring(lastDotIndex) : '';

            const fileObj = {
                id: Math.random().toString(36).substring(7),
                file: file,
                customName: name,
                ext: ext
            };

            stagedFiles.push(fileObj);
        });

        renderStagedFiles();

        // Reset input so same file can be selected again
        fileInput.value = '';
    });

    function renderStagedFiles() {
        fileStagingArea.innerHTML = '';

        stagedFiles.forEach(fileObj => {
            const el = document.createElement('div');
            el.className = 'staged-file';
            el.innerHTML = `
                <i class="bi bi-file-earmark-text staged-file-icon"></i>
                <div class="staged-file-info">
                    <input type="text" class="staged-file-input" value="${fileObj.customName}" placeholder="Filename">
                    <span class="staged-file-ext">${fileObj.ext}</span>
                </div>
                <i class="bi bi-x staged-file-remove" data-id="${fileObj.id}"></i>
            `;

            // Handle rename
            const input = el.querySelector('input');
            input.addEventListener('input', (e) => {
                fileObj.customName = e.target.value.trim();
            });

            // Handle remove
            const removeBtn = el.querySelector('.staged-file-remove');
            removeBtn.addEventListener('click', () => {
                stagedFiles = stagedFiles.filter(f => f.id !== fileObj.id);
                renderStagedFiles();
            });

            fileStagingArea.appendChild(el);
        });
    }

    // --- Chat Logic ---

    newChatBtn.addEventListener('click', () => {
        currentConvIdInput.value = '';
        chatMessages.innerHTML = '';
        chatMessages.appendChild(emptyState);
        emptyState.style.display = 'block';
        document.getElementById('chat-title').textContent = 'New Consultation';
        interruptionAlert.classList.add('d-none');
        statusBadge.textContent = 'Ready';
        statusBadge.className = 'badge bg-secondary rounded-pill';

        document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
        stagedFiles = [];
        renderStagedFiles();
    });

    // Handle Conversation Click
    // Use event delegation for dynamically added items if needed,
    // but here we just attach to existing ones on load.
    // Ideally, if we add new convs dynamically, we should refactor this.
    // For now, let's keep it simple.
    document.getElementById('conversation-list').addEventListener('click', (e) => {
        const item = e.target.closest('.conversation-item');
        if (item) {
            e.preventDefault();
            const id = item.dataset.id;
            loadConversation(id);

            document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = messageInput.value.trim();

        if (!message && stagedFiles.length === 0) return;

        // Optimistic UI Update
        if (message) {
            appendMessage('user', message);
        }
        if (stagedFiles.length > 0) {
            const fileNames = stagedFiles.map(f => f.customName + f.ext).join(', ');
            appendMessage('user', `*Attached files: ${fileNames}*`);
        }

        const formData = new FormData();
        formData.append('message', message);

        const convId = currentConvIdInput.value;
        if (convId) {
            formData.append('conversation_id', convId);
        }

        // Append files with renamed titles
        stagedFiles.forEach(f => {
            const finalName = (f.customName || 'file') + f.ext;
            formData.append('files', f.file, finalName);
        });

        // Clear input
        messageInput.value = '';
        stagedFiles = [];
        renderStagedFiles();

        if (emptyState) emptyState.style.display = 'none';

        statusBadge.textContent = 'Processing...';
        statusBadge.className = 'badge bg-warning text-dark rounded-pill';
        interruptionAlert.classList.add('d-none');

        // Disable input while processing? Maybe just button.
        const submitBtn = chatForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;

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
                    statusBadge.className = 'badge bg-success rounded-pill';
                } else if (data.status === 'interrupted') {
                    appendMessage('system_question', data.question);
                    interruptionAlert.classList.remove('d-none');
                    questionText.textContent = data.question;
                    messageInput.placeholder = "Please answer the question above...";
                    messageInput.focus();

                    statusBadge.textContent = 'Waiting for input';
                    statusBadge.className = 'badge bg-danger rounded-pill';
                }

                if (data.conversation_id) {
                    currentConvIdInput.value = data.conversation_id;
                    // Ideally, refresh sidebar here to show new conv or update timestamp
                }
            } else {
                appendMessage('error', data.error || 'An error occurred.');
                statusBadge.textContent = 'Error';
                statusBadge.className = 'badge bg-danger rounded-pill';
            }
        } catch (error) {
            console.error('Error:', error);
            appendMessage('error', 'Network error. Please try again.');
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge bg-danger rounded-pill';
        } finally {
            submitBtn.disabled = false;
        }
    });

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
                        if (msg.attachments && msg.attachments.length > 0) {
                           // Could append a separate message or info about attachments
                           // For now, let's assume content covers it or user remembers
                        }
                    });
                } else {
                    chatMessages.appendChild(emptyState);
                    emptyState.style.display = 'block';
                }

                const lastMsg = data.messages[data.messages.length - 1];
                if (lastMsg && lastMsg.role === 'system_question') {
                    interruptionAlert.classList.remove('d-none');
                    questionText.textContent = lastMsg.content;
                    statusBadge.textContent = 'Waiting for input';
                    statusBadge.className = 'badge bg-danger rounded-pill';
                } else {
                    interruptionAlert.classList.add('d-none');
                    statusBadge.textContent = 'Ready';
                    statusBadge.className = 'badge bg-secondary rounded-pill';
                }

                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        } catch (err) {
            console.error(err);
        }
    }

    function appendMessage(role, content) {
        // Wrapper
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${role}`;

        // Card
        const card = document.createElement('div');
        card.className = 'message-card';

        // Body
        const body = document.createElement('div');
        body.className = 'message-body';

        // Content
        if (role === 'assistant') {
            body.innerHTML = parseMarkdown(content);
        } else {
            body.innerHTML = content.replace(/\n/g, '<br>');
        }

        card.appendChild(body);
        wrapper.appendChild(card);
        chatMessages.appendChild(wrapper);

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

});
