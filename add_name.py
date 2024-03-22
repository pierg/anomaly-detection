import os
from datetime import datetime

def add_header_to_file(file_path, header):
    with open(file_path, 'r+') as f:
        original_content = f.read()
        # Check if the header is already present
        if "Author: Piergiuseppe Mallozzi" not in original_content:
            f.seek(0, 0)
            f.write(header.rstrip('\r\n') + '\n\n' + original_content)

def add_header_to_all_py_files(start_dir, header):
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                add_header_to_file(file_path, header)

# Customize the header information here
header = """
\"\"\"
Author: Piergiuseppe Mallozzi
Date: 2024
\"\"\"
""".strip().format(datetime.now().strftime('%Y-%m-%d'))

# Specify your project directory here
project_directory = './anomaly_detection'

add_header_to_all_py_files(project_directory, header)
