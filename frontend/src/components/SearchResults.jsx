import React from 'react';
import CaseCard from './CaseCard';
import EmptyState from './EmptyState';
import { Clock, CheckCircle2 } from 'lucide-react';

export default function SearchResults({ results, searchTime, query }) {
  if (!results || results.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="results-container">
      <div className="results-meta-header">
        <div className="meta-left">
          <CheckCircle2 size={18} className="meta-icon" />
          <h2 className="results-title">Search Results</h2>
        </div>
        <div className="meta-right">
          <span className="results-count">
            {results.length} relevant case{results.length > 1 ? 's' : ''} found
          </span>
          <span className="meta-divider">•</span>
          <span className="search-time">
            <Clock size={14} /> {searchTime}s
          </span>
        </div>
      </div>

      <div className="results-list">
        {results.map((result) => (
          <CaseCard key={result.case_id || result.rank} result={result} />
        ))}
      </div>
    </div>
  );
}

