import json
import glob
from pathlib import Path

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def main():
    cwd = Path.cwd()
    root = cwd / "output"
    target_root = root / "elite_five_integrated" / "checkpoints"
    target_root.mkdir(parents=True, exist_ok=True)
    
    # 锁定五大精英评审团目录
    elite_sources = [
        "gpt_single_judge_test",
        "claude_single_judge_test",
        "qwen_single_judge_test",
        "minimax_single_judge_test",
        "deepseek_single_judge_test"
    ]
    
    print(f"Scanning for Elite Five sources in: {root}")
    source_dirs = []
    for d_name in elite_sources:
        d = root / d_name
        if d.is_dir():
            cp_dir = d / "checkpoints"
            if cp_dir.exists():
                for model_folder in cp_dir.iterdir():
                    if model_folder.is_dir():
                        source_dirs.append(model_folder)
    
    print(f"Found {len(source_dirs)} elite source model directories.")
    
    # model_block_map: model_folder -> { block_filename -> aggregated_json }
    aggregated = {}

    for s_dir in source_dirs:
        s_path = Path(s_dir)
        model_folder = s_path.name
        
        if model_folder not in aggregated:
            aggregated[model_folder] = {}
            
        json_files = s_path.glob("*.json")
        for j_file in json_files:
            block_name = j_file.name
            print(f"Merging {s_path.parent.parent.name}/{model_folder}/{block_name}...")
            
            with open(j_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if block_name not in aggregated[model_folder]:
                aggregated[model_folder][block_name] = {}
            
            # 执行合并
            deep_merge(aggregated[model_folder][block_name], data)

    # 写入最终目录
    for m_folder, blocks in aggregated.items():
        dest_dir = target_root / m_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for b_name, b_data in blocks.items():
            dest_file = dest_dir / b_name
            with open(dest_file, 'w', encoding='utf-8') as f:
                json.dump(b_data, f, indent=2, ensure_ascii=False)
            print(f"Saved integrated cache: {dest_file}")

if __name__ == "__main__":
    main()
