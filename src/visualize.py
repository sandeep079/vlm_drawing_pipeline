import json
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="VLM Drawing Pipeline Visualizer", layout="wide")

OUTPUT_DIR = Path("output")
SAMPLES_DIR = Path("reference_samples")

st.title("🛠️ VLM Drawing Pipeline Visualizer")

# 1. Select Drawing
json_files = sorted(list(OUTPUT_DIR.glob("*_regions.json")))
if not json_files:
    st.error("No processed outputs found in output/ directory. Run batch pipeline first.")
    st.stop()

drawing_ids = [f.stem.replace("_regions", "") for f in json_files]
selected_id = st.sidebar.selectbox("Select Drawing ID", drawing_ids)

# Paths
pdf_path = SAMPLES_DIR / f"{selected_id}.pdf"
regions_file = OUTPUT_DIR / f"{selected_id}_regions.json"
class_file = OUTPUT_DIR / f"{selected_id}_classification.json"
bends_file = OUTPUT_DIR / f"{selected_id}_bends.json"

# Load JSON Data
regions_data = json.load(open(regions_file)) if regions_file.exists() else {}
class_data = json.load(open(class_file)) if class_file.exists() else {}
bends_data = json.load(open(bends_file)) if bends_file.exists() else {}

# Load PDF Image
@st.cache_data
def load_pdf_image(path):
    images = convert_from_path(path, dpi=300)
    return images[0]

page_image = load_pdf_image(pdf_path)

# --- Top Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Class", class_data.get("class", "N/A").upper())
col2.metric("Classification Confidence", class_data.get("class_confidence", "N/A"))
col3.metric("Bend Count", str(bends_data.get("num_bends", "N/A")))

st.markdown("---")

# --- 1. Full-Page Drawing & Grounding Overlay ---
st.subheader("1. Full-Page Drawing & Grounding Overlay")

overlay_img = page_image.copy()
draw = ImageDraw.Draw(overlay_img)
regions = regions_data.get("regions", [])

colors = {
    "flat_pattern": "#FF0000",
    "orthographic_view": "#00FF00",
    "section_view": "#0000FF",
    "title_block": "#FFA500",
}

for r in regions:
    bbox = r["bbox"]  # [xmin, ymin, xmax, ymax]
    box_type = r["type"]
    color = colors.get(box_type, "#FF00FF")
    draw.rectangle(bbox, outline=color, width=5)

st.image(overlay_img, width="stretch", caption="Full Page Grounding Overlay")

st.markdown("---")

# --- 2. Cropped Regions Gallery ---
st.subheader("2. Extracted Cropped Regions")

if regions:
    cols = st.columns(min(len(regions), 3))
    for idx, r in enumerate(regions):
        bbox = r["bbox"]
        box_type = r["type"]
        conf = r.get("conf", 0.0)

        # Crop region from original high-res image
        cropped_part = page_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        with cols[idx % 3]:
            st.image(
                cropped_part,
                caption=f"Type: {box_type} | Conf: {conf}",
                width="stretch",
            )
else:
    st.info("No localized regions found for this drawing.")

st.markdown("---")

# --- 3. Reasoning & Evidence ---
st.subheader("3. Reasoning & Evidence")
st.write(f"**Classification Evidence:** {class_data.get('class_evidence', 'N/A')}")
st.write(f"**Bend Count Evidence:** {bends_data.get('bend_evidence', 'N/A')}")