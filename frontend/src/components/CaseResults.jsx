import React from 'react';
import CaseResult from './CaseResult';

function parseInline(text) {
  if (!text) return text;

  const parts = [];
  const regex = /(\*\*.*?\*\*|\[.*?\])/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={match.index}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('[') && token.endsWith(']')) {
      parts.push(
        <span key={match.index} className="rag-citation-tag">
          {token}
        </span>
      );
    } else {
      parts.push(token);
    }
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }
  return parts.length > 0 ? parts : text;
}

function FormattedAnswer({ content }) {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];

  let currentList = null;
  let listType = null; // 'ul' | 'ol' | 'table'
  let paragraphLines = [];

  const flushList = (key) => {
    if (currentList && currentList.length > 0) {
      if (listType === 'ul') {
        elements.push(
          <ul key={`ul-${key}`} className="rag-answer-list">
            {currentList}
          </ul>
        );
      } else if (listType === 'ol') {
        elements.push(
          <ol key={`ol-${key}`} className="rag-answer-list">
            {currentList}
          </ol>
        );
      } else if (listType === 'table') {
        const headerRow = currentList[0];
        const bodyRows = currentList.slice(1);
        elements.push(
          <div key={`tbl-${key}`} className="rag-table-wrapper">
            <table className="rag-answer-table">
              <thead>
                <tr>
                  {headerRow.map((cell, cIdx) => (
                    <th key={cIdx}>{parseInline(cell)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bodyRows.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((cell, cIdx) => (
                      <td key={cIdx}>{parseInline(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      currentList = null;
      listType = null;
    }
  };

  const flushParagraph = (key) => {
    if (paragraphLines.length > 0) {
      const fullText = paragraphLines.join(' ');
      elements.push(
        <p key={`p-${key}`} className="rag-answer-paragraph">
          {parseInline(fullText)}
        </p>
      );
      paragraphLines = [];
    }
  };

  const flushAll = (key) => {
    flushList(key);
    flushParagraph(key);
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    // Empty line
    if (trimmed === '') {
      flushAll(idx);
      return;
    }

    // Horizontal Rule: --- or *** or ___
    if (trimmed === '---' || trimmed === '***' || trimmed === '___') {
      flushAll(idx);
      elements.push(<hr key={`hr-${idx}`} className="rag-answer-divider" />);
      return;
    }

    // Table row: | cell1 | cell2 |
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      // Skip markdown separator row like |---|---|
      if (/^\|[\s\-:|]+\|$/.test(trimmed)) {
        return;
      }
      flushParagraph(idx);
      if (listType !== 'table') {
        flushList(idx);
        listType = 'table';
        currentList = [];
      }
      const cells = trimmed
        .slice(1, -1)
        .split('|')
        .map((c) => c.trim());
      currentList.push(cells);
      return;
    }

    // Headers: # Title, ## Title, ### Title
    if (trimmed.startsWith('#')) {
      flushAll(idx);
      const match = trimmed.match(/^(#+)\s*(.*)/);
      if (match) {
        const level = match[1].length;
        const titleText = match[2];
        const hClass = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3';
        elements.push(
          <h4 key={`h-${idx}`} className={`rag-answer-heading ${hClass}`}>
            {parseInline(titleText)}
          </h4>
        );
        return;
      }
    }

    // Bold standalone line section title: **Title** or **Title:**
    const boldTitleMatch = trimmed.match(/^\*\*(.*?)\*\*:?$/);
    if (boldTitleMatch && !trimmed.includes('  ')) {
      flushAll(idx);
      elements.push(
        <h4 key={`bh-${idx}`} className="rag-answer-heading h2">
          {boldTitleMatch[1]}
        </h4>
      );
      return;
    }

    // Unordered List: - Item or * Item or • Item
    const ulMatch = trimmed.match(/^[-*•]\s+(.*)/);
    if (ulMatch) {
      flushParagraph(idx);
      if (listType !== 'ul') {
        flushList(idx);
        listType = 'ul';
        currentList = [];
      }
      currentList.push(<li key={`li-${idx}`}>{parseInline(ulMatch[1])}</li>);
      return;
    }

    // Ordered List: 1. Item or 1) Item
    const olMatch = trimmed.match(/^(\d+)[.)]\s+(.*)/);
    if (olMatch) {
      flushParagraph(idx);
      if (listType !== 'ol') {
        flushList(idx);
        listType = 'ol';
        currentList = [];
      }
      currentList.push(<li key={`li-${idx}`}>{parseInline(olMatch[2])}</li>);
      return;
    }

    // Regular text line -> append to current paragraph
    flushList(idx);
    paragraphLines.push(trimmed);
  });

  flushAll(lines.length);

  return <div className="formatted-rag-content">{elements}</div>;
}

export default function CaseResults({ results, searchTime, answer, notice, llmAvailable, disclaimer }) {
  if (!results || results.length === 0) {
    return (
      <div className="results-summary-text">
        <p>No relevant cases found for your query. Try asking a more specific legal question.</p>
      </div>
    );
  }

  return (
    <div className="case-results-container">
      {/* Grounded LLM Answer Block if available */}
      {answer && (
        <div className="rag-answer-card">
          <div className="rag-answer-header">Grounded AI Legal Synthesis</div>
          <div className="rag-answer-body">
            <FormattedAnswer content={answer} />
          </div>
          {disclaimer && (
            <div className="rag-disclaimer-text">{disclaimer}</div>
          )}
        </div>
      )}

      {/* Fallback Notice if LLM is unavailable */}
      {!answer && notice && (
        <div className="llm-notice-box">
          <span className="notice-text">{notice}</span>
        </div>
      )}

      {/* Supporting Cases Header */}
      <div className="results-heading-title">
        Supporting Cases ({results.length})
      </div>

      <div className="results-list-block">
        {results.map((item) => (
          <CaseResult key={item.case_id || item.rank} result={item} />
        ))}
      </div>
    </div>
  );
}

