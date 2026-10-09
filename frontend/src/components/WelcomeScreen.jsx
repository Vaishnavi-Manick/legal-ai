import React from 'react';
import { Scale, ArrowRight } from 'lucide-react';

const EXAMPLE_PROMPTS = [
  "Right to privacy under Article 21",
  "Constitutional validity of preventive detention",
  "Principles of natural justice"
];

export default function WelcomeScreen({ onSelectExample }) {
  return (
    <div className="welcome-container">
      <div className="welcome-hero">
        <div className="welcome-avatar">
          <Scale size={28} />
        </div>
        <h1 className="welcome-title">Legal AI Research Assistant</h1>
        <p className="welcome-subtitle">
          Search Indian Supreme Court cases using AI-powered semantic legal retrieval.
        </p>
      </div>

      <div className="welcome-examples-section">
        <div className="examples-header">Try asking</div>
        <div className="examples-grid">
          {EXAMPLE_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              className="example-card"
              onClick={() => onSelectExample(prompt)}
              type="button"
            >
              <span className="example-text">"{prompt}"</span>
              <ArrowRight size={16} className="example-arrow" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
