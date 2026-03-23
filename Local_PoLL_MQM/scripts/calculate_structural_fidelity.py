import json
import os
import re
import pandas as pd
from pathlib import Path

def extract_structural_elements(text):
    # Extract markdown links, brackets, tags, bold/italic, code blocks
    elements = []
    # Markdown links
    elements.extend(re.findall(r'\[([^\]]+)\]\([^\)]+\)', text))
    # Code blocks
    elements.extend(re.findall(r'`([^`]+)`', text))
    # Brackets
    elements.extend(re.findall(r'【([^】]+)】', text))
    # HTML tags
    elements.extend(re.findall(r'<[^>]+>', text))
    return elements

def calculate_fidelity():
    audit_root = Path("Local_PoLL_MQM/output/elite_five_integrated/audited_reports")
    output_dir = Path("Local_PoLL_MQM/analysis_infra/raw_stats")
    
    results = []
    if not audit_root.exists():
        print("Audit root not found:", audit_root)
        return

    for model_dir in audit_root.iterdir():
        if not model_dir.is_dir(): continue
        model_id = model_dir.name
        
        total_samples = 0
        preserved_samples = 0
        
        for audit_file in model_dir.glob("*_poll_mqm_audit.json"):
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {audit_file}: {e}")
                continue
                
            records = data.get("results", [])
            for rec in records:
                source = rec.get("source", "")
                hypothesis = rec.get("hypothesis", "")
                if not source or not hypothesis: continue
                
                src_elements = extract_structural_elements(source)
                hyp_elements = extract_structural_elements(hypothesis)
                
                total_samples += 1
                
                # Check preservation
                # A simple metric: proportion of structural elements preserved
                is_preserved = True
                for el in src_elements:
                    if el not in hypothesis:
                        is_preserved = False
                        break
                
                if is_preserved:
                    preserved_samples += 1
                    
        if total_samples > 0:
            score = (preserved_samples / total_samples) * 100
        else:
            score = 100.0
            
        results.append({
            "model_id": model_id,
            "fidelity_score": score,
            "sample_size": total_samples
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by="fidelity_score", ascending=False)
    out_csv = output_dir / "structural_fidelity.csv"
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Calculated true structural fidelity. Saved to {out_csv}")

if __name__ == "__main__":
    calculate_fidelity()
