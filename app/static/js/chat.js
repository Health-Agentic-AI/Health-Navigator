// Chat Page JavaScript - Modern Health Navigator
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const conversationList = document.querySelector('.conversations-wrapper');
    const newChatBtn = document.getElementById('newChatBtn');
    const currentConvIdInput = document.getElementById('currentConversationId');
    const attachmentBtn = document.getElementById('attachmentBtn');
    const fileInput = document.getElementById('fileInput');
    const filePreviewArea = document.getElementById('filePreviewArea');
    const emptyState = document.getElementById('emptyState');
    const interruptionAlert = document.getElementById('interruptionAlert');
    const questionText = document.getElementById('questionText');
    const statusBadge = document.querySelector('.status-badge');
    const chatTitle = document.getElementById('chatTitle');
    const sendBtn = document.getElementById('sendBtn');
    const toggleSidebar = document.getElementById('toggleSidebar');
    const chatSidebar = document.getElementById('chatSidebar');

    // State
    let stagedFiles = [];
    let isProcessing = false;

    // ==================
    // SIDEBAR FUNCTIONS
    // ==================

    // Toggle sidebar on mobile
    if (toggleSidebar && chatSidebar) {
        toggleSidebar.addEventListener('click', () => {
            chatSidebar.classList.toggle('show');
        });
    }

    // New chat button
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            startNewChat();
        });
    }

    function startNewChat() {
        currentConvIdInput.value = '';
        chatMessages.innerHTML = '';
        if (emptyState) {
            chatMessages.appendChild(emptyState);
            emptyState.style.display = 'flex';
        }
        chatTitle.textContent = 'New Consultation';
        updateStatus('ready', 'Ready');
        hideInterruptionAlert();

        // Clear active conversation
        document.querySelectorAll('.conversation-card').forEach(el => {
            el.classList.remove('active');
        });

        // Clear staged files
        stagedFiles = [];
        renderFilePreview();

        // Clear message input
        messageInput.value = '';
        autoResizeTextarea();
    }

    // Load conversation
    if (conversationList) {
        conversationList.addEventListener('click', (e) => {
            const card = e.target.closest('.conversation-card');
            if (card) {
                const id = card.dataset.id;
                loadConversation(id);

                // Update active state
                document.querySelectorAll('.conversation-card').forEach(el => {
                    el.classList.remove('active');
                });
                card.classList.add('active');

                // Close sidebar on mobile
                if (window.innerWidth <= 1024) {
                    chatSidebar.classList.remove('show');
                }
            }
        });
    }

    async function loadConversation(id) {
        try {
            showLoading('Loading conversation...');

            const res = await fetch(`/api/conversations/${id}`);
            const data = await res.json();

            hideLoading();

            if (res.ok) {
                currentConvIdInput.value = data.id;
                chatTitle.textContent = data.title;
                chatMessages.innerHTML = '';

                if (data.messages && data.messages.length > 0) {
                    data.messages.forEach(msg => {
                        appendMessage(msg.role, msg.content, false);
                    });

                    // Check last message for interruption
                    const lastMsg = data.messages[data.messages.length - 1];
                    if (lastMsg && lastMsg.role === 'system_question') {
                        hideInterruptionAlert();
                        updateStatus('waiting', 'Waiting for input');
                    } else {
                        hideInterruptionAlert();
                        updateStatus('ready', 'Ready');
                    }
                } else {
                    if (emptyState) {
                        chatMessages.appendChild(emptyState);
                        emptyState.style.display = 'flex';
                    }
                }

                scrollToBottom();
            } else {
                showError('Failed to load conversation');
            }
        } catch (err) {
            hideLoading();
            console.error('Error loading conversation:', err);
            showError('Network error while loading conversation');
        }
    }

    // ==================
    // FILE HANDLING
    // ==================

    if (attachmentBtn) {
        attachmentBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    function handleFileSelect(e) {
        const files = Array.from(e.target.files);

        if (files.length === 0) return;

        // Check limits
        if (stagedFiles.length + files.length > 20) {
            alert('Maximum 20 files allowed.');
            return;
        }

        files.forEach(file => {
            // Check size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert(`File ${file.name} is too large (Max 10MB).`);
                return;
            }

            const lastDotIndex = file.name.lastIndexOf('.');
            const name = lastDotIndex !== -1 ? file.name.substring(0, lastDotIndex) : file.name;
            const ext = lastDotIndex !== -1 ? file.name.substring(lastDotIndex) : '';

            stagedFiles.push({
                id: generateId(),
                file: file,
                customName: name,
                ext: ext
            });
        });

        renderFilePreview();
        fileInput.value = ''; // Reset input
    }

    function renderFilePreview() {
        filePreviewArea.innerHTML = '';

        if (stagedFiles.length === 0) {
            filePreviewArea.style.display = 'none';
            return;
        }

        filePreviewArea.style.display = 'flex';

        stagedFiles.forEach(fileObj => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            item.innerHTML = `
                <i class="bi bi-file-earmark-text file-preview-icon"></i>
                <div class="file-preview-name-wrapper">
                    <input type="text" 
                           class="file-preview-name-input" 
                           value="${fileObj.customName}" 
                           placeholder="Filename"
                           data-id="${fileObj.id}">
                    <span class="file-preview-ext">${fileObj.ext}</span>
                </div>
                <button type="button" class="file-preview-remove" data-id="${fileObj.id}">
                    <i class="bi bi-x"></i>
                </button>
            `;

            // Handle filename editing
            const nameInput = item.querySelector('.file-preview-name-input');
            nameInput.addEventListener('input', (e) => {
                const file = stagedFiles.find(f => f.id === fileObj.id);
                if (file) {
                    file.customName = e.target.value.trim();
                }
            });

            // Handle remove
            const removeBtn = item.querySelector('.file-preview-remove');
            removeBtn.addEventListener('click', () => {
                stagedFiles = stagedFiles.filter(f => f.id !== fileObj.id);
                renderFilePreview();
            });

            filePreviewArea.appendChild(item);
        });
    }

    // ==================
    // MESSAGE HANDLING
    // ==================

    // Auto-resize textarea
    if (messageInput) {
        messageInput.addEventListener('input', autoResizeTextarea);

        // Handle Enter key
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
    }

    function autoResizeTextarea() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    }

    // Form submission
    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (isProcessing) return;

            const message = messageInput.value.trim();

            if (!message && stagedFiles.length === 0) return;

            // Disable form
            isProcessing = true;
            sendBtn.disabled = true;

            // Hide empty state
            if (emptyState) {
                emptyState.style.display = 'none';
            }

            // Show user message
            if (message) {
                appendMessage('user', message, true);
            }

            // Show file attachments
            if (stagedFiles.length > 0) {
                const fileNames = stagedFiles.map(f => f.customName + f.ext).join(', ');
                appendMessage('user', `📎 Attached files: ${fileNames}`, true);
            }

            // Prepare form data
            const formData = new FormData();
            formData.append('message', message);

            const convId = currentConvIdInput.value;
            if (convId) {
                formData.append('conversation_id', convId);
            }

            // Append files
            stagedFiles.forEach(f => {
                const finalName = (f.customName || 'file') + f.ext;
                formData.append('files', f.file, finalName);
            });

            // Clear input
            messageInput.value = '';
            autoResizeTextarea();
            stagedFiles = [];
            renderFilePreview();

            // Update status
            updateStatus('analyzing', 'Analyzing...');
            hideInterruptionAlert();

            // Show loading
            showLoading('Analyzing medical data...');

            try {
                const response = await fetch('/api/chat/message', {
                    method: 'POST',
                    body: formData
                });

                let data = {};
                try {
                    data = await response.json();
                } catch (parseErr) {
                    const rawText = await response.text();
                    data = { error: rawText || 'Unexpected server response' };
                }

                console.log('Backend response:', data);

                hideLoading();

                if (response.ok) {
                    if (data.status === 'completed') {
                        appendMessage('assistant', data.response, true);
                        updateStatus('ready', 'Completed');
                        hideInterruptionAlert();
                    } else if (data.status === 'interrupted') {
                        appendMessage('system_question', data.question, true);
                        hideInterruptionAlert();
                        updateStatus('waiting', 'Waiting for input');
                        messageInput.placeholder = "Please answer the question above...";
                        messageInput.focus();
                    }

                    if (data.conversation_id) {
                        currentConvIdInput.value = data.conversation_id;
                        // Update chat title if new conversation
                        if (!convId && message) {
                            chatTitle.textContent = message.substring(0, 30) + (message.length > 30 ? '...' : '');
                        }
                    }
                } else {
                    appendMessage('error', data.error || 'An error occurred. Please try again.', true);
                    updateStatus('ready', 'Error');
                }
            } catch (error) {
                hideLoading();
                console.error('Error:', error);
                appendMessage('error', 'Network error. Please check your connection and try again.', true);
                updateStatus('ready', 'Error');
            } finally {
                isProcessing = false;
                sendBtn.disabled = false;
            }
        });
    }

    // ==================
    // MESSAGE RENDERING
    // ==================

    function appendMessage(role, content, animate = true) {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${role}`;
        if (animate) {
            wrapper.style.opacity = '0';
        }

        const card = document.createElement('div');
        card.className = 'message-card';

        const body = document.createElement('div');
        body.className = 'message-body';

        // Parse content
        if (role === 'assistant') {
            body.innerHTML = parseMarkdown(content);
        } else if (role === 'system_question') {
            body.innerHTML = parseMarkdown(formatSystemQuestion(content));
        } else if (role === 'error') {
            body.innerHTML = `<strong>Error:</strong> ${escapeHtml(content)}`;
        } else {
            body.innerHTML = escapeHtml(content).replace(/\n/g, '<br>');
        }

        card.appendChild(body);
        wrapper.appendChild(card);
        chatMessages.appendChild(wrapper);

        if (animate) {
            setTimeout(() => {
                wrapper.style.opacity = '1';
            }, 10);
        }

        scrollToBottom();
    }

    // ==================
    // LOADING INDICATOR
    // ==================

    let loadingElement = null;

    function showLoading(text = 'Loading...') {
        hideLoading(); // Remove any existing loader

        loadingElement = document.createElement('div');
        loadingElement.className = 'message-wrapper assistant';
        loadingElement.innerHTML = `
            <div class="message-card">
                <div class="loading-indicator">
                    <div class="loading-circle"></div>
                    <div class="loading-text">${text}</div>
                </div>
            </div>
        `;

        chatMessages.appendChild(loadingElement);
        scrollToBottom();
    }

    function hideLoading() {
        if (loadingElement) {
            loadingElement.remove();
            loadingElement = null;
        }
    }

    // ==================
    // STATUS MANAGEMENT
    // ==================

    function updateStatus(type, text) {
        const statusDot = statusBadge.querySelector('.status-dot');
        const statusText = statusBadge.querySelector('.status-text');

        // Remove all status classes
        statusBadge.classList.remove('ready', 'analyzing', 'waiting', 'error');

        // Add new status class
        statusBadge.classList.add(type);
        statusText.textContent = text;
    }

    function showInterruptionAlert(question) {
        questionText.textContent = question;
        interruptionAlert.classList.add('show');
    }

    function hideInterruptionAlert() {
        interruptionAlert.classList.remove('show');
    }

    // ==================
    // UTILITY FUNCTIONS
    // ==================

    function parseMarkdown(text) {
        if (!text) return '';

        // Normalize excessive blank lines from backend output
        const normalized = String(text)
            .replace(/\r\n/g, '\n')
            .replace(/\n{2,}/g, '\n')
            .trim();

        let html = escapeHtml(normalized);

        // Headers
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // Bold and italics
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(?!\s)(.+?)\*/g, '<em>$1</em>');

        // Horizontal rules
        html = html.replace(/^\*\*\*$/gm, '<hr>');
        html = html.replace(/^---$/gm, '<hr>');

        // Lists - handle bullet points
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

        // Wrap consecutive <li> in <ul>
        html = html.replace(/(<li>.*?<\/li>\n?)+/g, match => `<ul>${match}</ul>`);

        // Paragraphs
        const lines = html.split('\n\n');
        html = lines.map(line => {
            if (line.trim() && !line.startsWith('<') && !line.endsWith('>')) {
                return `<p>${line}</p>`;
            }
            return line;
        }).join('\n');

        // Remove newlines between tags, then add line breaks for remaining newlines
        html = html.replace(/>\n</g, '><');
        html = html.replace(/\n/g, '<br>');

        // Cleanup empty paragraphs
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p><br><\/p>/g, '');

        return html;
    }

    function formatSystemQuestion(text) {
        const content = text ? String(text).trim() : '';
        return `**▲ Additional information required**\n\n${content}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    function generateId() {
        return Math.random().toString(36).substring(2, 9);
    }

    function showError(message) {
        // Could implement a toast notification here
        console.error(message);
    }

    // ==================
    // QUICK ACTIONS
    // ==================

    const quickActionBtns = document.querySelectorAll('.quick-action-btn');
    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            const text = this.querySelector('span').textContent;

            if (text.includes('symptoms')) {
                messageInput.value = 'I would like to describe my symptoms.';
                messageInput.focus();
            } else if (text.includes('Upload')) {
                fileInput.click();
            } else if (text.includes('question')) {
                messageInput.value = 'I have a medical question: ';
                messageInput.focus();
            }
        });
    });

    // ==================
    // EXPORT CHAT
    // ==================

    const exportBtn = document.querySelector('.btn-header-action[title="Export chat"]');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            exportChatAsText();
        });
    }

    function exportChatAsText() {
        const messages = chatMessages.querySelectorAll('.message-wrapper');

        if (messages.length === 0) {
            if (window.showNotification) {
                showNotification('No messages to export', 'warning');
            } else {
                alert('No messages to export');
            }
            return;
        }

        let exportText = `Health Navigator Chat Export\n`;
        exportText += `Date: ${new Date().toLocaleString()}\n`;
        exportText += `Chat: ${chatTitle.textContent}\n`;
        exportText += `${'='.repeat(60)}\n\n`;

        messages.forEach(msg => {
            const role = msg.classList.contains('user') ? 'You' :
                msg.classList.contains('assistant') ? 'AI Assistant' :
                    'System';
            const content = msg.querySelector('.message-body').textContent.trim();

            exportText += `[${role}]\n${content}\n\n`;
        });

        exportText += `${'='.repeat(60)}\n`;
        exportText += `End of chat export\n`;

        // Create download
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `health-navigator-chat-${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (window.showNotification) {
            showNotification('Chat exported successfully', 'success');
        }
    }

    // ==================
    // INITIALIZATION
    // ==================

    // Focus message input on load
    if (messageInput && !currentConvIdInput.value) {
        messageInput.focus();
    }

    // Smooth scroll to bottom on load
    scrollToBottom();

    console.log('Chat interface initialized successfully');
});
