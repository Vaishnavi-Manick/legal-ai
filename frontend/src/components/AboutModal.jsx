import React from 'react';
import { X, Database, Cpu, Layers, Trophy, FileCheck } from 'lucide-react';

export default function AboutModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-title-block">
            <h2 className="modal-title">System Information & Model Metrics</h2>
            <span className="modal-subtitle">Legal AI Case Retrieval Pipeline (AILA 2019)</span>
          </div>
          <button className="modal-close" onClick={onClose} type="button">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <section className="about-section">
            <h3 className="section-title">
              <Database size={16} /> System Architecture Overview
            </h3>
            <div className="info-grid">
              <div className="info-card">
                <span className="info-label">Dataset</span>
                <span className="info-value">AILA 2019</span>
              </div>
              <div className="info-card">
                <span className="info-label">Documents</span>
                <span className="info-value">2,914 Legal Cases</span>
              </div>
              <div className="info-card">
                <span className="info-label">Stage 1 Retrieval</span>
                <span className="info-value">TF-IDF Index</span>
              </div>
              <div className="info-card">
                <span className="info-label">Candidate Pool</span>
                <span className="info-value">Top 50 Candidates</span>
              </div>
              <div className="info-card">
                <span className="info-label">Stage 2 Ranking</span>
                <span className="info-value">Transformer Cross-Encoder</span>
              </div>
              <div className="info-card">
                <span className="info-label">Output Results</span>
                <span className="info-value">Top 5 Cases</span>
              </div>
            </div>
          </section>

          <section className="about-section">
            <h3 className="section-title">
              <Trophy size={16} /> Models Evaluated Comparison
            </h3>
            <p className="section-description">
              Verified evaluation metrics on AILA 2019 test queries (Q11-Q50).
            </p>
            <div className="metrics-table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>MAP (Mean Avg Precision)</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Siamese BiLSTM + Attention</td>
                    <td className="metric-score">0.0068</td>
                    <td><span className="status-tag status-baseline">Baseline</span></td>
                  </tr>
                  <tr className="highlight-row">
                    <td>
                      <strong className="model-name">Legal Transformer Cross-Encoder</strong>
                      <div className="model-subtext">all-MiniLM-L6-v2 + Chunk MaxP</div>
                    </td>
                    <td className="metric-score highlight-score">0.1886</td>
                    <td><span className="status-tag status-active">Active Verified Model</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="about-section">
            <h3 className="section-title">
              <FileCheck size={16} /> 2-Stage Retrieval Flow
            </h3>
            <div className="pipeline-flow">
              <div className="pipeline-step">Query Input</div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">TF-IDF Stage 1 (Top 50)</div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">Transformer Stage 2</div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">Top 5 Output</div>
            </div>
          </section>
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose} type="button">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
