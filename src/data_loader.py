import os
import re
from typing import Dict, List, Tuple, Set

class AILADataLoader:
    def __init__(self, data_dir: str = os.path.join("data", "raw", "AILA2019")):
        # If the requested directory does not exist, check common relative locations
        if not os.path.exists(data_dir):
            candidates = [
                os.path.join("data", "raw", "AILA2019"),
                os.path.join("..", "data", "raw", "AILA2019"),
                os.path.join("data", "AILA2019"),
                os.path.join("..", "data", "AILA2019")
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    data_dir = cand
                    break

        self.data_dir = data_dir
        self.queries_file = os.path.join(data_dir, "Query_doc.txt")
        self.casedocs_dir = os.path.join(data_dir, "Object_casedocs")
        self.statutes_dir = os.path.join(data_dir, "Object_statutes")
        self.qrels_priorcases_file = os.path.join(data_dir, "relevance_judgments_priorcases.txt")
        self.qrels_statutes_file = os.path.join(data_dir, "relevance_judgments_statutes.txt")

    def load_queries(self) -> Dict[str, str]:
        """Loads queries mapping qid -> query_text"""
        queries = {}
        if not os.path.exists(self.queries_file):
            raise FileNotFoundError(f"Queries file not found at {self.queries_file}")
        
        with open(self.queries_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "||" in line:
                    parts = line.split("||", 1)
                    qid = parts[0].strip()
                    text = parts[1].strip()
                    queries[qid] = text
        return queries

    def load_case_docs(self) -> Dict[str, str]:
        """Loads case documents mapping doc_id -> doc_text"""
        casedocs = {}
        if not os.path.exists(self.casedocs_dir):
            raise FileNotFoundError(f"Case docs directory not found at {self.casedocs_dir}")
        
        for filename in os.listdir(self.casedocs_dir):
            if filename.endswith(".txt"):
                doc_id = filename[:-4] # e.g. C1 from C1.txt
                filepath = os.path.join(self.casedocs_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    casedocs[doc_id] = text
        return casedocs

    def load_qrels_priorcases(self) -> Dict[str, Set[str]]:
        """Loads ground truth relevance mapping qid -> set of relevant doc_ids"""
        qrels = {}
        if not os.path.exists(self.qrels_priorcases_file):
            raise FileNotFoundError(f"Qrels file not found at {self.qrels_priorcases_file}")
        
        with open(self.qrels_priorcases_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    qid = parts[0]
                    doc_id = parts[2]
                    rel = int(parts[3])
                    if qid not in qrels:
                        qrels[qid] = set()
                    if rel > 0:
                        qrels[qid].add(doc_id)
        return qrels

    def get_query_splits(self, queries: Dict[str, str], qrels: Dict[str, Set[str]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits queries into Train (Q1-Q8), Validation (Q9-Q10), and Test (Q11-Q50).
        Ensures queries with positive ground truth are used for training.
        """
        train_qids, val_qids, test_qids = [], [], []
        
        for qid in sorted(queries.keys(), key=lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) else 0):
            q_num = int(re.sub(r'\D', '', qid)) if re.sub(r'\D', '', qid) else 0
            if q_num >= 11:
                test_qids.append(qid)
            elif q_num in [9, 10]:
                val_qids.append(qid)
            else:
                train_qids.append(qid)
                
        train_qids = [q for q in train_qids if len(qrels.get(q, set())) > 0]
        val_qids = [q for q in val_qids if len(qrels.get(q, set())) > 0]
        test_qids = [q for q in test_qids if len(qrels.get(q, set())) > 0]
        
        return train_qids, val_qids, test_qids
