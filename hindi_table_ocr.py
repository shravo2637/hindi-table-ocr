
!pip install easyocr pdf2image --quiet
!apt-get install -y poppler-utils > /dev/null

import cv2
import numpy as np
import easyocr
from pdf2image import convert_from_path
from google.colab import files


uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]


images = convert_from_path(pdf_path, dpi=300)

reader = easyocr.Reader(['hi', 'en'], gpu=True)


for page_num, img in enumerate(images):
    print(f"\n📄 --- Page {page_num + 1} ---")

    image = np.array(img)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    table_mask = cv2.add(horizontal_lines, vertical_lines)

    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    table_boxes = []
    table_mask_overlay = np.zeros_like(gray)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40 and h > 20:
            table_boxes.append((x, y, w, h))
            cv2.rectangle(table_mask_overlay, (x, y), (x+w, y+h), 255, -1) 

    
    table_boxes = sorted(table_boxes, key=lambda b: (b[1] // 20, b[0]))
    rows = []
    current_row = []
    last_y = -100
    for box in table_boxes:
        x, y, w, h = box
        if abs(y - last_y) > 20 and current_row:
            rows.append(current_row)
            current_row = []
        current_row.append(box)
        last_y = y
    if current_row:
        rows.append(current_row)

    
    structured_output = []

    for row in rows:
        row = sorted(row, key=lambda b: b[0])
        row_text = []
        min_y = row[0][1]
        for x, y, w, h in row:
            cell = image[y:y+h, x:x+w]
            result = reader.readtext(cell, detail=0)
            cell_text = " ".join(result).strip()
            row_text.append(cell_text if cell_text else " ")
        structured_output.append((min_y, 'table', row_text))

    
    all_text = reader.readtext(image, detail=1)
    for box, text, _ in all_text:
        
        x0, y0 = box[0]
        x1, y1 = box[2]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        
        if table_mask_overlay[int(cy), int(cx)] == 255:
            continue
        structured_output.append((int(cy), 'text', text.strip()))

    
    structured_output.sort(key=lambda x: x[0])

    for _, content_type, content in structured_output:
        if content_type == 'table':
            print(" | ".join(content))
        else:
            print(content)
