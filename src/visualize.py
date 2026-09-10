import json
import os
from pathlib import Path
import pdf2image
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="VLM Pipeline Visualizer", layout="wide")

# Directory Paths
BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = BASE_DIR / "reference_samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
CROPS_DIR = OUTPUTS_DIR / "crops"
JSON_DIR = OUTPUTS_DIR / "json"

st.title("🛠️ VLM Drawing Pipeline Visualizer")

# Sidebar: Select Drawing ID
pdf_files = list(SAMPLES_DIR.glob("*.pdf"))
drawing_ids = [f.stem for f in pdf_files]

selected_id = st.sidebar.selectbox("Select Drawing ID", sorted(drawing_ids))

if selected_id:
    st.header(f"Drawing Analysis: `{selected_id}`")
    
    # Load JSON Outputs
    region_file = JSON_DIR / f"{selected_id}_regions.json"
    class_file = JSON_DIR / f"{selected_id}_classification.json"
    count_file = JSON_DIR / f"{selected_id}_bends.json"
    
    regions_data = json.load(open(region_file)) if region_file.exists() else None
    class_data = json.load(open(class_file)) if class_file.exists() else None
    count_data = json.load(open(count_file)) if count_file.exists() else None

    # Top Metrics Bar
    col1, col2, col3 = st.columns(3)
    with col1:
        part_class = class_data.get("class", "N/A").upper() if class_data else "N/A"
        st.metric("Predicted Class", part_class)
    with col2:
        class_conf = class_data.get("class_confidence", 0.0) if class_data else 0.0
        st.metric("Classification Confidence", f"{class_conf:.2f}")
    with col3:
        bends = count_data.get("num_bends", "N/A") if count_data else "N/A"
        st.metric("Bend Count", str(bends))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("1. Full-Page Drawing & Localized Bounding Boxes")
        pdf_path = SAMPLES_DIR / f"{selected_id}.pdf"
        
        if pdf_path.exists():
            # Convert PDF first page to Image
            images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=1)
            base_img = images[0].convert("RGB")
            draw_img = base_img.copy()
            draw = ImageDraw.Draw(draw_img)
            w, h = draw_img.size

            # Color palette for regions
            color_map = {
                "title_block": "#FF3366",
                "flat_pattern": "#33CC66",
                "orthographic_view": "#3399FF",
                "isometric_view": "#FFCC00",
                "section_view": "#CC33FF"
            }

            if regions_data and "regions" in regions_data:
                for region in regions_data["regions"]:
                    label = region.get("type", "region")
                    bbox = region.get("bbox", [0, 0, 0, 0])  # Normalized 0-1000
                    
                    # Convert 0-1000 normalized to absolute pixel values
                    x0, y0 = (bbox[0] / 1000) * w, (bbox[1] / 1000) * h
                    x1, y1 = (bbox[2] / 1000) * w, (bbox[3] / 1000) * h
                    
                    color = color_map.get(label, "#FFFFFF")
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                    draw.text((x0 + 5, y0 + 5), label, fill=color)

            st.image(draw_img, use_container_width=True, caption="Grounding Overlay (Stage 1)")

    with col_right:
        st.subheader("2. Crop Artifacts")
        crop_files = sorted(list(CROPS_DIR.glob(f"{selected_id}_*.png")))
        if crop_files:
            for crop_path in crop_files:
                st.caption(f"Crop: `{crop_path.name}`")
                st.image(str(crop_path), width=300)
        else:
            st.warning("No crops found for this drawing ID.")

        st.subheader("3. Reasoning & Evidence")
        if class_data:
            st.markdown(f"**Classification Evidence:**\n> {class_data.get('class_evidence', 'N/A')}")
        if count_data:
            st.markdown(f"**Bend Count Evidence:**\n> {count_data.get('bend_evidence', 'N/A')}")