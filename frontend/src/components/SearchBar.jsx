import React from 'react';
import { Search, X, Sparkles } from 'lucide-react';

const EXAMPLE_QUERIES = [
  'right to privacy under Article 21',
  'constitutional validity of preventive detention',
  'principles of natural justice'
];

export default function SearchBar({ query, setQuery, onSearch, isLoading }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSearch(query);
      }
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  const handleExampleClick = (example) => {
    setQuery(example);
    onSearch(example);
  };

  return (
    <div className="search-section">
      <div className="search-header">
        <h1 className="hero-title">AI Legal Case Retrieval</h1>
        <p className="hero-subtitle">Find relevant Indian Supreme Court cases from AILA 2019 dataset using 2-stage Transformer Cross-Encoder ranking</p>
      </div>

      <div className="search-box-card">
        <div className="input-wrapper">
          <textarea
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. constitutional validity of preventive detention"
            rows={3}
            disabled={isLoading}
          />
          {query && !isLoading && (
            <button 
              className="clear-button"
              onClick={handleClear}
              title="Clear text"
              type="button"
            >
              <X size={18} />
            </button>
          )}
        </div>

        <div className="search-actions">
          <button
            className="search-button"
            onClick={() => onSearch(query)}
            disabled={!query.trim() || isLoading}
            type="button"
          >
            <Search size={18} />
            <span>{isLoading ? 'Searching...' : 'Search Cases'}</span>
          </button>
        </div>

        <div className="example-queries">
          <span className="example-label">
            <Sparkles size={14} /> Try example query:
          </span>
          <div className="example-chips">
            {EXAMPLE_QUERIES.map((example, idx) => (
              <button
                key={idx}
                className="chip"
                onClick={() => handleExampleClick(example)}
                disabled={isLoading}
                type="button"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

