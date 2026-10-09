import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState() {
  return (
    <div className="loading-state">
      <div className="spinner-wrapper">
        <Loader2 className="spinner" size={36} />
      </div>
      <h3 className="loading-title">Searching relevant legal cases...</h3>
      <p className="loading-subtitle">
        Running Stage-1 TF-IDF candidate retrieval over 2,914 documents and Stage-2 Transformer Cross-Encoder re-ranking...
      </p>

      <div className="skeleton-container">
        <div className="skeleton-card"></div>
        <div className="skeleton-card"></div>
        <div className="skeleton-card"></div>
      </div>
    </div>
  );
}

