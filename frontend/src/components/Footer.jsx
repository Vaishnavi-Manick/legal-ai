import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="disclaimer-box">
          <ShieldAlert size={16} className="disclaimer-icon" />
          <div className="disclaimer-text">
            <strong>Research Prototype</strong> — This system is intended for legal research assistance and educational purposes. Retrieved cases should be independently verified using authoritative legal sources.
          </div>
        </div>
        <div className="footer-copyright">
          AILA 2019 Dataset • Two-Stage Legal Retrieval Engine
        </div>
      </div>
    </footer>
  );
}

