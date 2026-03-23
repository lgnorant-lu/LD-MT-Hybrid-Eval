import json
import argparse
from pathlib import Path
from collections import defaultdict

def integrity_check(checkpoint_dir: str, fix: bool = False):
    root = Path(checkpoint_dir)
    if not root.exists():
        print(f"Error: Path {root} does not exist.")
        return

    # Tracking
    total_items = 0
    corrupted_items = 0
    reasons = defaultdict(int)
    
    print(f"--- Integrity Check: {root} ---")
    
    # We iterate through all JSON files in the directory
    for json_file in root.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")
                continue
        
        modified = False
        new_data = {}
        
        for tid, judges in data.items():
            valid_judges_for_item = {}
            for jid, decision in judges.items():
                total_items += 1
                is_corrupted = False
                reason = ""
                
                # Check 1: Explicit ok=False
                if not decision.get("ok", True):
                    is_corrupted = True
                    reason = "EXPLICIT_FAIL"
                
                # Check 2: Content keywords (In case they were missed by in-flight guard)
                raw_text = decision.get("raw_text", "").lower()
                if "<!doctype html" in raw_text or "请求次数过多" in raw_text:
                    is_corrupted = True
                    reason = "HTML_INTERCEPT"
                
                # Check 3: Empty results but ok was true (Suspicious)
                if decision.get("ok") and not decision.get("errors") and not raw_text:
                    is_corrupted = True
                    reason = "EMPTY_SUCCESS"

                if is_corrupted:
                    corrupted_items += 1
                    reasons[reason] += 1
                    if fix:
                        modified = True
                        # By NOT adding this judge to valid_judges_for_item, 
                        # we effectively "clean" it if it's the only judge,
                        # or keep the others.
                        print(f"  [FIX] Removing corrupted judge {jid} for {tid} ({reason})")
                        continue
                
                valid_judges_for_item[jid] = decision
            
            if valid_judges_for_item:
                new_data[tid] = valid_judges_for_item
        
        if fix and modified:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            print(f"Saved cleaned file: {json_file}")

    print("\nSummary:")
    print(f"  Total Judge Decisions Scanned: {total_items}")
    print(f"  Corrupted Items Found: {corrupted_items}")
    for r, count in reasons.items():
        print(f"    - {r}: {count}")
    
    if fix:
        print("\nIntegrity check and cleaning complete.")
    else:
        print("\nRun with --fix to automatically remove corrupted items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and clean corrupted PoLL checkpoints.")
    parser.get_default("checkpoint_dir")
    parser.add_argument("dir", help="Directory to scan")
    parser.add_argument("--fix", action="store_true", help="Remove corrupted entries")
    args = parser.parse_args()
    
    integrity_check(args.dir, args.fix)
