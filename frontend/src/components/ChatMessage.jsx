import React from 'react';
import { AlertCircle } from 'lucide-react';
import CaseResults from './CaseResults';

export default function ChatMessage({ message }) {
  const isUser = message.sender === 'user';

  if (isUser) {
    return (
      <div className="chat-message-row user-row">
        <div className="user-label-title">you do:</div>
        <div className="user-text-content">{message.content}</div>
      </div>
    );
  }

  if (message.isError) {
    return (
      <div className="chat-message-row ai-row">
        <div className="ai-header-block">
          <span className="ai-label-title">Legal AI</span>
        </div>
        <div className="error-message-box">
          <AlertCircle size={16} className="error-icon" />
          <div className="error-text">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-message-row ai-row">
      <div className="ai-header-block">
        <span className="ai-label-title">Legal AI</span>
      </div>
      <CaseResults 
        results={message.results} 
        searchTime={message.searchTime}
        answer={message.answer}
        notice={message.notice}
        llmAvailable={message.llmAvailable}
        disclaimer={message.disclaimer}
      />
    </div>
  );
}
