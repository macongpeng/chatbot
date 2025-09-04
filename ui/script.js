// MediRecords Chatbot UI JavaScript
class MedibotChat {
    constructor() {
        this.apiUrl = 'http://localhost:8080/medibot/chat';
        this.healthUrl = 'http://localhost:8080/health/liveness';
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.apiStatus = document.getElementById('apiStatus');
        
        this.isTyping = false;
        this.messageHistory = [];
        
        this.init();
    }
    
    init() {
        // Set welcome message time
        document.getElementById('welcome-time').textContent = this.formatTime(new Date());
        
        // Event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => this.updateCharCount());
        
        // Check API health
        this.checkApiHealth();
        setInterval(() => this.checkApiHealth(), 30000); // Check every 30 seconds
        
        // Focus input
        this.messageInput.focus();
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch(this.healthUrl, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                this.updateStatus('connected', 'Connected');
                this.apiStatus.textContent = 'Ready';
            } else {
                this.updateStatus('warning', 'API Issues');
                this.apiStatus.textContent = 'Warning';
            }
        } catch (error) {
            this.updateStatus('disconnected', 'Disconnected');
            this.apiStatus.textContent = 'Offline';
        }
    }
    
    updateStatus(status, text) {
        this.statusText.textContent = text;
        this.statusDot.className = 'status-dot';
        
        switch(status) {
            case 'connected':
                this.statusDot.style.background = '#2ecc71';
                break;
            case 'warning':
                this.statusDot.style.background = '#f39c12';
                break;
            case 'disconnected':
                this.statusDot.style.background = '#e74c3c';
                break;
        }
    }
    
    updateCharCount() {
        const length = this.messageInput.value.length;
        this.charCount.textContent = length;
        
        if (length > 450) {
            this.charCount.style.color = '#e74c3c';
        } else if (length > 400) {
            this.charCount.style.color = '#f39c12';
        } else {
            this.charCount.style.color = '#7f8c8d';
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageHistory.push({ role: 'user', content: message });
        
        // Clear input and show typing
        this.messageInput.value = '';
        this.updateCharCount();
        this.showTyping(true);
        
        try {
            // Send to API
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ usermessage: message })
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add bot response
            setTimeout(() => {
                this.showTyping(false);
                this.addMessage(data.data, 'bot');
                this.messageHistory.push({ role: 'assistant', content: data.data });
            }, 1000); // Small delay for better UX
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.showTyping(false);
            this.addMessage(
                `Sorry, I'm having trouble connecting to the server. Please check that the MediRecords chatbot service is running on localhost:8080.\n\nError: ${error.message}`, 
                'bot', 
                'error'
            );
        }
    }
    
    addMessage(text, sender, type = 'normal') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? '👤' : '🤖';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        if (type === 'error') {
            messageText.style.color = '#e74c3c';
            messageText.innerHTML = this.formatText(text);
        } else {
            messageText.innerHTML = this.formatText(text);
        }
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = this.formatTime(new Date());
        
        content.appendChild(messageText);
        content.appendChild(messageTime);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatText(text) {
        // Simple text formatting
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    }
    
    formatTime(date) {
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
    }
    
    showTyping(show) {
        this.isTyping = show;
        this.sendButton.disabled = show;
        this.typingIndicator.style.display = show ? 'flex' : 'none';
        
        if (show) {
            this.sendButton.innerHTML = '<div class="loading"></div> Thinking...';
        } else {
            this.sendButton.innerHTML = '<span class="send-icon">📤</span><span class="send-text">Send</span>';
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Sample queries for easy testing
const sampleQueries = [
    "How do I create an appointment in MediRecords?",
    "Where is patient registration located?",
    "How to generate Medicare reports?",
    "What are the billing options available?",
    "How do I set up reminders for patients?",
    "How to manage clinical consultations?",
    "What system settings can I configure?",
    "How to export patient data reports?",
    "EMR system troubleshooting help",
    "How to integrate with third party systems?"
];

// Add sample query buttons (optional enhancement)
function addSampleQueries() {
    const welcomeMessage = document.querySelector('.bot-message .message-text');
    
    const samplesDiv = document.createElement('div');
    samplesDiv.innerHTML = `
        <div style="margin-top: 15px;">
            <strong>💡 Try these sample queries:</strong>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                ${sampleQueries.slice(0, 4).map(query => 
                    `<button class="sample-query-btn" onclick="document.getElementById('messageInput').value='${query}'; document.getElementById('messageInput').focus();">${query}</button>`
                ).join('')}
            </div>
        </div>
    `;
    
    // Add CSS for sample query buttons
    const style = document.createElement('style');
    style.textContent = `
        .sample-query-btn {
            background: rgba(52, 152, 219, 0.1);
            border: 1px solid rgba(52, 152, 219, 0.2);
            color: #3498db;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .sample-query-btn:hover {
            background: rgba(52, 152, 219, 0.2);
            transform: translateY(-1px);
        }
    `;
    document.head.appendChild(style);
    
    welcomeMessage.appendChild(samplesDiv);
}

// Initialize the chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const chat = new MedibotChat();
    
    // Add sample queries after a brief delay
    setTimeout(addSampleQueries, 1000);
    
    // Add some helpful keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.getElementById('messageInput').focus();
        }
        
        // Esc to clear input
        if (e.key === 'Escape') {
            document.getElementById('messageInput').value = '';
            document.getElementById('messageInput').blur();
        }
    });
});