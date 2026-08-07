import os
import shutil
import xml.etree.ElementTree as ET

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_XML_PATH = os.path.join(SCRIPT_DIR, "..", "data", "info.xml")
REF_BASE = os.path.join(SCRIPT_DIR, "..", "data", "reference_signatures")

os.makedirs(REF_BASE, exist_ok=True)

# --- Parse XML ---
tree = ET.parse(INFO_XML_PATH)
root = tree.getroot()

# Build ROW_TO_INDEX dynamically from XML
ROW_TO_INDEX = {}
students = root.find("students")
for i, student in enumerate(students.findall("student"), start=1):
    student_index = student.find("index").text
    ROW_TO_INDEX[i] = student_index

print("Loaded mapping:", ROW_TO_INDEX)

# --- Copy reference signatures ---
for sheet_num in range(1, 5):
    src_dir = os.path.join(SCRIPT_DIR, "..", "output", "signature_regions", f"sheet{sheet_num}_students")
    
    for row_num, student_index in ROW_TO_INDEX.items():
        src_file = os.path.join(src_dir, f"row_{row_num:02d}.png")
        dst_dir = os.path.join(REF_BASE, student_index)
        os.makedirs(dst_dir, exist_ok=True)
        
        if os.path.exists(src_file):
            dst_file = os.path.join(dst_dir, f"ref_{sheet_num}.png")
            shutil.copy2(src_file, dst_file)
            print(f"Copied sheet{sheet_num} row{row_num} -> {dst_file}")

print("\nDone.")