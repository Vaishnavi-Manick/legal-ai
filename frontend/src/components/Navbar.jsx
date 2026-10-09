import React from 'react';
import { Sparkles, Share2, MoreHorizontal, PanelLeft } from 'lucide-react';

export default function Navbar({ onToggleSidebar, onOpenAbout }) {
  return (
    <div className="top-header-bar">
      <div className="header-left">
        <button className="sidebar-icon-btn mobile-toggle" onClick={onToggleSidebar} type="button">
          <PanelLeft size={18} />
        </button>
      </div>

      <div className="header-actions-right">
        <button className="header-btn upgrade" onClick={onOpenAbout} type="button">
          <Sparkles size={15} />
          <span>Upgrade</span>
        </button>
        <button className="header-btn" type="button">
          <Share2 size={15} />
          <span>Share</span>
        </button>
        <button className="header-btn" onClick={onOpenAbout} type="button">
          <MoreHorizontal size={18} />
        </button>
      </div>
    </div>
  );
}
