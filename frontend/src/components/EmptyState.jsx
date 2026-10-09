import React from 'react';
import { SearchX } from 'lucide-react';

export default function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon-wrapper">
        <SearchX size={40} className="empty-icon" />
      </div>
      <h3 className="empty-title">No relevant cases found</h3>
      <p className="empty-subtitle">Try using a more specific legal query or check for spelling errors.</p>
    </div>
  );
}

