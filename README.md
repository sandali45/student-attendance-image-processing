# Student Attendance Image Processing System

A Python-based student attendance management system developed for **CS402.3 – Computer Graphics and Visualization**.

The system processes photographed attendance signing sheets, identifies whether each student has signed, maps detected rows to student information from XML, stores attendance in SQLite, and generates attendance visualizations.

## Main Features

- Load attendance-sheet images captured using a smartphone
- Correct and preprocess images
- Convert images to grayscale
- Reduce noise and enhance contrast
- Apply thresholding and morphological cleaning
- Detect the attendance table and student rows
- Extract student signature regions
- Detect **Present / Absent** from signature regions
- Read student information from `info.xml`
- Store attendance records in SQLite
- Generate student attendance graphs
- Compare signatures with previous reference signatures
- Simple Tkinter UI for processing attendance and viewing student summaries
- Automated testing using `pytest`

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- SQLite
- XML
- Tkinter
- Pillow
- scikit-image
- pytest

## Project Structure

```text
student-attendance-image-processing/
│
├── app.py
├── sams.py
├── infovis.py
├── investigate.py
├── requirements.txt
├── pytest.ini
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── image_loader.py
│   ├── image_preprocessing.py
│   ├── thresholding.py
│   ├── table_detection.py
│   ├── signature_extraction.py
│   ├── attendance_detection.py
│   ├── xml_parser.py
│   ├── database.py
│   ├── visualization.py
│   └── signature_verification.py
│
├── data/
│   ├── info.xml
│   ├── signing_sheets/
│   └── reference_signatures/
│
├── database/
│   └── attendance.db
│
├── output/
│   ├── corrected_images/
│   ├── grayscale_images/
│   ├── binary_images/
│   ├── detected_tables/
│   ├── signature_regions/
│   ├── attendance_results/
│   ├── graphs/
│   └── signature_reports/
│
└── tests/
    ├── test_image_loader.py
    ├── test_image_preprocessing.py
    ├── test_thresholding.py
    ├── test_table_detection.py
    ├── test_signature_extraction.py
    ├── test_attendance_detection.py
    ├── test_xml_parser.py
    ├── test_database.py
    ├── test_visualization.py
    └── test_signature_verification.py
```

## System Workflow

```text
Attendance Sheet Image
        ↓
Image Loading and Correction
        ↓
Grayscale + Noise Removal
        ↓
Thresholding
        ↓
Table Detection
        ↓
Signature Region Extraction
        ↓
Present / Absent Detection
        ↓
XML Student Mapping
        ↓
SQLite Database
        ↓
Attendance Visualization
```

Signature verification is available separately through `investigate.py`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sandali45/student-attendance-image-processing.git
cd student-attendance-image-processing
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the User Interface

```bash
python app.py
```

The UI contains two main sections:

### Process Attendance

- Select a signing-sheet image
- Select the student XML file
- Enter the attendance date
- Process the sheet
- View image-processing stages
- View Present / Absent results

### Student Summary

- Enter a student index
- View total Present and Absent counts
- View attendance percentage
- View attendance graphs

## Run the Main Attendance Program

```bash
python sams.py <signing_sheet_image> <info_xml>
```

Example:

```bash
python sams.py data/signing_sheets/1.jpeg data/info.xml
```

If the current version uses a date argument:

```bash
python sams.py data/signing_sheets/2.jpeg data/info.xml --date 2019-06-21
```

## Attendance Visualization

```bash
python infovis.py <student_index>
```

Example:

```bash
python infovis.py 10009301
```

Generated graphs are stored in:

```text
output/graphs/
```

## Signature Verification

```bash
python investigate.py <student_index>
```

Reference signatures should be stored under:

```text
data/reference_signatures/<student_index>/
```

When testing a signature, do not include the exact same signature image in its reference set.

## Student Information XML

Student information is stored in:

```text
data/info.xml
```

Example:

```xml
<attendance_info>
    <subject>
        <subject_name>Computer Graphics and Visualization</subject_name>
        <subject_code>CS402.3</subject_code>
    </subject>

    <students>
        <student>
            <index>10000409</index>
            <name>Student Name</name>
        </student>
    </students>
</attendance_info>
```

The student order in the XML should match the student-row order on the signing sheet.

## Testing

Run all tests:

```bash
python -m pytest -q
```

Run only the attendance-detection tests:

```bash
python -m pytest tests/test_attendance_detection.py -v
```

The project includes tests for:

- Image loading and correction
- Image preprocessing
- Thresholding
- Table detection
- Signature extraction
- Present / Absent detection
- XML parsing
- Database operations
- Visualization
- Signature verification

## Output Files

Processed results are stored in the `output/` directory:

```text
output/corrected_images/
output/grayscale_images/
output/binary_images/
output/detected_tables/
output/signature_regions/
output/attendance_results/
output/graphs/
output/signature_reports/
```

Attendance records are stored in:

```text
database/attendance.db
```

## Group Module Distribution

| Member | Responsibility |
|---|---|
| 1 | Image loading and smartphone-photo correction |
| 2 | Grayscale, contrast enhancement and noise removal |
| 3 | Binarization and morphological cleaning |
| 4 | Attendance-table and grid-line detection |
| 5 | Student-row and signature-box extraction |
| 6 | Present / Absent signature detection |
| 7 | XML reading and student-row mapping |
| 8 | SQLite database |
| 9 | Attendance visualization |
| 10 | Signature verification |

## Coursework Deliverables

The final coursework submission contains:

- **Prototype ZIP** – the complete runnable project
- **Report** – screenshots, testing results, discussion, technologies, and individual contributions

## Notes

- The project is designed for the provided static signing-sheet layout.
- The XML student count and order should match the detected student rows.
- The UI is an additional front end; `sams.py`, `infovis.py`, and `investigate.py` remain available as command-line programs.
- Keep test signatures separate from their own reference images when evaluating signature similarity.

## Repository

https://github.com/sandali45/student-attendance-image-processing
