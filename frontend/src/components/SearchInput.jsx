import React, { useRef, useEffect } from 'react';
import { Plus, Brain, Mic, ArrowUp, ChevronDown } from 'lucide-react';

export default function SearchInput({ 
  query, 
  setQuery, 
  onSend, 
  isLoading,
  showScrollDown,
  onScrollDown
}) {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSend(query);
      }
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSend(query);
    }
  };

  return (
    <div className="bottom-input-container">
      {showScrollDown && (
        <button 
          className="scroll-down-btn" 
          onClick={onScrollDown}
          title="Scroll to bottom"
          type="button"
        >
          <ChevronDown size={18} />
        </button>
      )}

      <div className="input-capsule-wrapper">
        <form onSubmit={handleSubmit} className="input-capsule-form">
          <button type="button" className="attach-plus-btn" title="Add attachment">
            <Plus size={18} />
          </button>

          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            rows={1}
            disabled={isLoading}
          />

          <div className="capsule-right-actions">
            <button type="button" className="think-pill-btn">
              <Brain size={14} />
              <span>Think</span>
            </button>
            
            <button type="button" className="mic-btn" title="Voice input">
              <Mic size={16} />
            </button>

            <button
              type="submit"
              className="blue-submit-circle"
              disabled={!query.trim() || isLoading}
              title="Send query"
            >
              <ArrowUp size={16} />
            </button>
          </div>
        </form>
      </div>

      <div className="input-disclaimer">
        Legal AI can make mistakes. Verify retrieved cases using authoritative legal sources.
      </div>
    </div>
  );
}
