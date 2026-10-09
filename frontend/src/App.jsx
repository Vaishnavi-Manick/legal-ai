import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import WelcomeScreen from './components/WelcomeScreen';
import ChatMessage from './components/ChatMessage';
import LoadingMessage from './components/LoadingMessage';
import SearchInput from './components/SearchInput';
import AboutModal from './components/AboutModal';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('retrieval');
  const [recentSearches, setRecentSearches] = useState([
    "right to privacy under Article 21",
    "constitutional validity of preventive detention",
    "principles of natural justice",
    "OCR Retrieval Flow",
    "Set briefing preferences"
  ]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [showScrollDown, setShowScrollDown] = useState(false);

  const messagesEndRef = useRef(null);
  const streamContainerRef = useRef(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('legal_ai_recent_searches');
      if (saved) {
        setRecentSearches(JSON.parse(saved));
      }
    } catch (e) {
      console.error('Failed to load recent searches:', e);
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleScroll = () => {
    if (streamContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = streamContainerRef.current;
      setShowScrollDown(scrollHeight - scrollTop - clientHeight > 150);
    }
  };

  const saveRecentSearch = (queryText) => {
    const trimmed = queryText.trim();
    if (!trimmed) return;
    setRecentSearches((prev) => {
      const filtered = prev.filter((item) => (item.query || item) !== trimmed);
      const updated = [{ query: trimmed, timestamp: Date.now() }, ...filtered].slice(0, 15);
      try {
        localStorage.setItem('legal_ai_recent_searches', JSON.stringify(updated));
      } catch (e) {
        console.error('Failed to save recent search:', e);
      }
      return updated;
    });
  };

  const handleSendQuery = async (queryText) => {
    const text = (queryText || inputQuery).trim();
    if (!text || isLoading) return;

    setInputQuery('');
    setActiveQuery(text);
    setIsLoading(true);
    setLoadingStage('retrieval');
    saveRecentSearch(text);

    const userMsgId = `user-${Date.now()}`;
    const userMessage = {
      id: userMsgId,
      sender: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await axios.post(
        `${API_URL}/search`,
        { query: text, top_k: 5 },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 60000
        }
      );

      const data = response.data || {};
      const aiMsgId = `ai-${Date.now()}`;
      const aiMessage = {
        id: aiMsgId,
        sender: 'ai',
        results: data.results || [],
        searchTime: data.search_time || 0,
        answer: data.answer,
        notice: data.notice,
        supportingCases: data.supporting_cases,
        disclaimer: data.disclaimer,
        llmAvailable: data.llm_available,
        query: text
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error('Search API request error:', err);
      let errorText = 'An unexpected error occurred during legal search.';
      if (err.code === 'ERR_NETWORK' || !err.response) {
        errorText = 'Unable to connect to the Legal AI backend server. Please verify that the backend is running at http://localhost:8000.';
      } else if (err.response?.data?.detail) {
        errorText = typeof err.response.data.detail === 'string'
          ? err.response.data.detail
          : 'Server error during legal case retrieval.';
      }

      const aiErrorMsg = {
        id: `ai-err-${Date.now()}`,
        sender: 'ai',
        isError: true,
        content: errorText
      };
      setMessages((prev) => [...prev, aiErrorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSearch = () => {
    setMessages([]);
    setInputQuery('');
    setActiveQuery('');
    setIsSidebarOpen(false);
  };

  const handleSelectRecent = (queryText) => {
    setIsSidebarOpen(false);
    handleSendQuery(queryText);
  };

  return (
    <div className="chat-app-root">
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewSearch={handleNewSearch}
        recentSearches={recentSearches}
        onSelectRecent={handleSelectRecent}
        activeQuery={activeQuery}
        onOpenAbout={() => setIsAboutOpen(true)}
      />

      <div className="chat-main-layout">
        <Navbar 
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          onOpenAbout={() => setIsAboutOpen(true)}
        />

        <div 
          ref={streamContainerRef}
          onScroll={handleScroll}
          className="chat-stream-container"
        >
          {messages.length === 0 ? (
            <WelcomeScreen onSelectExample={handleSendQuery} />
          ) : (
            <div className="conversation-thread">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              {isLoading && <LoadingMessage stage={loadingStage} />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <SearchInput
          query={inputQuery}
          setQuery={setInputQuery}
          onSend={handleSendQuery}
          isLoading={isLoading}
          showScrollDown={showScrollDown}
          onScrollDown={() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })}
        />
      </div>

      <AboutModal
        isOpen={isAboutOpen}
        onClose={() => setIsAboutOpen(false)}
      />
    </div>
  );
}
