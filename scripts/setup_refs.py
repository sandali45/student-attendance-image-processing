import os
import shutil

REF_BASE = "data/reference_signatures"
os.makedirs(REF_BASE, exist_ok=True)

ROW_TO_INDEX = {
    1: "10000409",
    2: "10009301",
    3: "10009302",
    4: "10009303",
    5: "10009304",
    6: "10009306",
}
    
for sheet_num in range(1, 5):
    src_dir = f"output/signature_regions/sheet{sheet_num}_students"
    
    for row_num, student_index in ROW_TO_INDEX.items():
        src_file = os.path.join(src_dir, f"row_{row_num:02d}.png")
        dst_dir = os.path.join(REF_BASE, student_index)
        os.makedirs(dst_dir, exist_ok=True)
        
        if os.path.exists(src_file):
            dst_file = os.path.join(dst_dir, f"ref_{sheet_num}.png")
            shutil.copy2(src_file, dst_file)
            print(f"Copied sheet{sheet_num} row{row_num} -> {dst_file}")

print("\nDone.")