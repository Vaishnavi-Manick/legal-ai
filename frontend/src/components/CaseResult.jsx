import React, { useState } from 'react';
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';

export default function CaseResult({ result }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const textToCopy = `Case ID: ${result.case_id}\nRelevance Score: ${result.relevance_score}\nCase Name: ${result.case_name || 'N/A'}\nRelevant Passage: ${result.snippet}`;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="case-result-codebox">
      <div className="codebox-top-bar">
        <span className="codebox-rank-title">
          #{result.rank} &nbsp; Case ID: {result.case_id}
        </span>
        <div className="codebox-actions">
          <button className="codebox-copy-btn" onClick={handleCopy} type="button">
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      <div className="codebox-body">
        {result.case_name && (
          <div className="case-info-line">{result.case_name}</div>
        )}

        <div className="case-score-line">
          Relevance Score: <strong>{result.relevance_score.toFixed(4)}</strong>
        </div>

        {result.citation && (
          <div className="result-citation">{result.citation}</div>
        )}

        <div className="passage-box">
          "{result.snippet}"
        </div>

        <button
          className="passage-toggle-btn"
          onClick={() => setIsExpanded(!isExpanded)}
          type="button"
        >
          <span>{isExpanded ? 'Hide full document' : 'View full document'}</span>
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {isExpanded && (
          <div className="full-doc-container">
            <textarea
              className="full-doc-text"
              value={result.document_text}
              readOnly
              rows={8}
            />
          </div>
        )}
      </div>
    </div>
  );
}
