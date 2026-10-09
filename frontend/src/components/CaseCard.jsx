import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Award, Layers } from 'lucide-react';

const RANK_BADGES = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];

export default function CaseCard({ result }) {
  const [expanded, setExpanded] = useState(result.rank === 1);

  const rankEmoji = RANK_BADGES[result.rank - 1] || `#${result.rank}`;

  return (
    <div className={`case-card ${expanded ? 'is-expanded' : ''}`}>
      <div className="case-card-header" onClick={() => setExpanded(!expanded)}>
        <div className="header-left">
          <span className="rank-badge">{rankEmoji} Rank {result.rank}</span>
          <span className="case-id-badge">
            <FileText size={14} /> Case ID: {result.case_id}
          </span>
          {result.case_name && (
            <span className="case-name">{result.case_name}</span>
          )}
        </div>

        <div className="header-right">
          <div className="score-badge" title="Transformer Cross-Encoder MaxP Relevance Score">
            <Award size={14} />
            <span>Score: {result.relevance_score.toFixed(4)}</span>
          </div>
          <button 
            className="expand-toggle"
            aria-label={expanded ? 'Collapse card' : 'Expand card'}
            type="button"
          >
            {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>
      </div>

      <div className="case-card-body">
        {result.citation && (
          <div className="metadata-row">
            <strong>Citation:</strong> {result.citation}
          </div>
        )}

        <div className="snippet-box">
          <div className="snippet-label">Relevant Passage:</div>
          <p className="snippet-text">"{result.snippet}"</p>
        </div>

        <div className="card-actions">
          <button
            className="view-passage-button"
            onClick={() => setExpanded(!expanded)}
            type="button"
          >
            <Layers size={14} />
            <span>{expanded ? 'Hide Full Case Document' : 'View Relevant Passage & Full Document'}</span>
          </button>
        </div>

        {expanded && result.document_text && (
          <div className="full-document-section">
            <div className="document-header">
              <span className="document-title">Full Retrieved Case Document ({result.case_id})</span>
              <span className="document-length">{result.document_text.length} characters</span>
            </div>
            <textarea
              className="full-document-text"
              value={result.document_text}
              readOnly
              rows={12}
            />
          </div>
        )}
      </div>
    </div>
  );
}

