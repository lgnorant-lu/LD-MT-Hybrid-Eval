import json
from pathlib import Path
from collections import Counter, defaultdict

def deep_audit():
    root = Path("output/elite_five_integrated/checkpoints")
    targets = ['gpt_5_01', 'claude_4_6_01', 'qwen_3_5_sf', 'minimax_m2_5_sf', 'deepseek_v3_2_sf']
    
    # Metrics
    total_decisions = 0
    valid_json_count = 0
    suspicious_long_count = 0
    keyword_filtered_count = 0
    
    judge_health = {jid: Counter() for jid in targets}
    
    print(f"Starting Content-Level Audit for Elite Five in: {root}")

    for json_file in root.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except: continue
            
            for tid, judges in data.items():
                for jid in targets:
                    if jid not in judges: continue
                    total_decisions += 1
                    
                    decision = judges[jid]
                    raw = str(decision.get("raw_text", ""))
                    ok = decision.get("ok", False)
                    
                    # 1. Length Check (Max 15000 chars for a batch of 20 is reasonable)
                    if len(raw) > 20000:
                        suspicious_long_count += 1
                        judge_health[jid]["SUSPICIOUS_LONG"] += 1
                        continue

                    # 2. Key-logic Check (Hidden errors)
                    low_raw = raw.lower()
                    if any(k in low_raw for k in ["error", "limit", "quota", "504 gateway", "doctype html"]):
                        # Check if it's just mentioning an error in MQM context
                        if not ("results" in low_raw or "errors" in low_raw):
                            keyword_filtered_count += 1
                            judge_health[jid]["HIDDEN_ERROR"] += 1
                            continue

                    # 3. Structure Check
                    if "results" in raw and "test_id" in raw:
                        valid_json_count += 1
                        judge_health[jid]["VALID_MQM_JSON"] += 1
                    elif ok:
                        judge_health[jid]["OTHER_SUCCESS"] += 1
                    else:
                        judge_health[jid]["KNOWN_FAIL"] += 1

    print("\n" + "="*60)
    print(f"Global Data Integrity: {(valid_json_count/total_decisions*100):.2f}% Normal")
    print(f"Total Decisions: {total_decisions}")
    print(f"Valid MQM JSON: {valid_json_count}")
    print(f"Suspicious Long: {suspicious_long_count}")
    print(f"Hidden Errors: {keyword_filtered_count}")
    print("="*60)
    
    for jid in targets:
        print(f"\nJudge: {jid}")
        for cat, count in judge_health[jid].most_common():
            print(f"  - {cat:<20}: {count}")

if __name__ == "__main__":
    deep_audit()
