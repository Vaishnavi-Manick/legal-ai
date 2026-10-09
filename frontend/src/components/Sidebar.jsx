import React from 'react';
import { 
  Plus, 
  Search, 
  PanelLeft, 
  Scale, 
  ShieldCheck, 
  Gavel, 
  Database, 
  Info,
  Clock
} from 'lucide-react';

const LEGAL_CATEGORIES = [
  { name: "Constitutional Law", query: "right to privacy under Article 21", icon: Scale },
  { name: "Preventive Detention", query: "constitutional validity of preventive detention", icon: ShieldCheck },
  { name: "Natural Justice", query: "principles of natural justice", icon: Gavel },
  { name: "AILA 2019 Corpus", query: "supreme court precedent cases", icon: Database }
];

export default function Sidebar({
  isOpen,
  onClose,
  onNewSearch,
  recentSearches,
  onSelectRecent,
  activeQuery,
  onOpenAbout
}) {
  return (
    <>
      {isOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside className={`sidebar ${isOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <Scale size={18} className="brand-scale-icon" />
            <span className="sidebar-brand-title">Legal AI</span>
          </div>
          <div className="sidebar-header-actions">
            <button className="sidebar-icon-btn" onClick={onOpenAbout} title="System Details" type="button">
              <Info size={16} />
            </button>
            <button className="sidebar-icon-btn" onClick={onClose} title="Close sidebar" type="button">
              <PanelLeft size={16} />
            </button>
          </div>
        </div>

        <div className="sidebar-action">
          <button 
            className="new-chat-btn"
            onClick={onNewSearch}
            type="button"
          >
            <div className="new-chat-left">
              <Plus size={16} />
              <span>New Search</span>
            </div>
          </button>
        </div>

        <div className="sidebar-content">
          <div className="sidebar-section-header">LEGAL DOMAINS</div>
          <div className="legal-categories-list">
            {LEGAL_CATEGORIES.map((cat, idx) => {
              const IconComp = cat.icon;
              return (
                <button
                  key={idx}
                  className="legal-category-item"
                  onClick={() => onSelectRecent(cat.query)}
                  type="button"
                >
                  <IconComp size={14} className="cat-icon" />
                  <span className="cat-name">{cat.name}</span>
                </button>
              );
            })}
          </div>

          <div className="sidebar-section-header">RECENT SEARCHES</div>
          <div className="recent-list">
            {recentSearches.length === 0 ? (
              <div className="recent-empty">No recent legal queries</div>
            ) : (
              recentSearches.map((item, idx) => {
                const qText = item.query || item;
                const isActive = qText === activeQuery;
                return (
                  <button
                    key={idx}
                    className={`recent-item ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectRecent(qText)}
                    title={qText}
                    type="button"
                  >
                    <Clock size={13} className="recent-clock-icon" />
                    <span className="recent-text">{qText}</span>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="user-profile-block" onClick={onOpenAbout} style={{ cursor: 'pointer' }}>
            <div className="user-avatar-circle">LA</div>
            <div className="user-profile-info">
              <span className="user-name">Legal AI Engine</span>
              <span className="user-plan">AILA 2019 • MAP 0.1886</span>
            </div>
          </div>
          <button className="upgrade-pill-btn" onClick={onOpenAbout} type="button">
            Info
          </button>
        </div>
      </aside>
    </>
  );
}
