import React from 'react';

export default function LoadingMessage({ stage }) {
  const loadingText = stage === 'llm' 
    ? 'Generating grounded legal explanation...' 
    : 'Searching legal case database...';

  return (
    <div className="chat-message-row ai-row">
      <div className="ai-header-block">
        <span className="ai-label-title">Legal AI</span>
      </div>
      <div className="loading-row-box">
        <span className="loading-text">{loadingText}</span>
        <div className="dots-pulse">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </div>
      </div>
    </div>
  );
}
