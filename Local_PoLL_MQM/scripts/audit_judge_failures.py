import json
from pathlib import Path
from collections import Counter

def audit_checkpoints():
    root = Path("output/full_seven_integrated/checkpoints")
    if not root.exists():
        print("Checkpoint root not found.")
        return

    # Statistics
    judge_stats = Counter()
    error_types = Counter()
    model_failure_map = {} # model -> failed_test_ids_count

    print(f"Auditing checkpoints in: {root}")
    
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        model_failure_map[model_name] = 0
        
        for json_file in model_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
                
                for test_id, judges in data.items():
                    test_failed = False
                    for judge_id, decision in judges.items():
                        if not decision.get("ok", True):
                            test_failed = True
                            judge_stats[judge_id] += 1
                            err_msg = decision.get("error_message", "Unknown Error")
                            
                            # Categorize error
                            if "429" in err_msg:
                                error_types["429_RateLimit"] += 1
                            elif "403" in err_msg:
                                error_types["403_Forbidden/Balance"] += 1
                            elif "connection" in err_msg.lower():
                                error_types["ConnectionError"] += 1
                            else:
                                error_types["Other_Error"] += 1
                    
                    if test_failed:
                        model_failure_map[model_name] += 1

    print("\n=== Judge Failure Statistics (How many times each judge failed) ===")
    for jid, count in judge_stats.most_common():
        print(f"Judge: {jid:<20} | Failures: {count}")

    print("\n=== Error Type Distribution ===")
    for etype, count in error_types.items():
        print(f"Type: {etype:<25} | Occurrences: {count}")

    print("\n=== Impacted Models (Top 10 models with most corrupted test items) ===")
    sorted_models = sorted(model_failure_map.items(), key=lambda x: x[1], reverse=True)
    for mname, count in sorted_models[:10]:
        print(f"Model: {mname:<40} | Corrupted Items: {count}")

if __name__ == "__main__":
    audit_checkpoints()
