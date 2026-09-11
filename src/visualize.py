import json
import os
from pathlib import Path
import pdf2image
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="VLM Pipeline Visualizer", layout="wide")

BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = BASE_DIR / "reference_samples"
OUTPUT_DIR = BASE_DIR / "output"

st.title("🛠️ VLM Drawing Pipeline Visualizer")

pdf_files = list(SAMPLES_DIR.glob("*.pdf"))
drawing_ids = [f.stem for f in pdf_files]

selected_id = st.sidebar.selectbox("Select Drawing ID", sorted(drawing_ids))

if selected_id:
    st.header(f"Drawing Analysis: `{selected_id}`")

    region_file = OUTPUT_DIR / f"{selected_id}_regions.json"
    class_file = OUTPUT_DIR / f"{selected_id}_classification.json"
    count_file = OUTPUT_DIR / f"{selected_id}_bends.json"

    regions_data = (
        json.load(open(region_file)) if region_file.exists() else None
    )
    class_data = json.load(open(class_file)) if class_file.exists() else None
    count_data = json.load(open(count_file)) if count_file.exists() else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Predicted Class",
            class_data.get("class", "N/A").upper() if class_data else "N/A",
        )
    with col2:
        st.metric(
            "Classification Confidence",
            f"{class_data.get('class_confidence', 0.0):.2f}"
            if class_data
            else "0.00",
        )
    with col3:
        st.metric(
            "Bend Count",
            str(count_data.get("num_bends", "N/A")) if count_data else "N/A",
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("1. Full-Page Drawing & Localized Bounding Boxes")
        pdf_path = SAMPLES_DIR / f"{selected_id}.pdf"

        if pdf_path.exists():
            # MATCH DPI WITH LOCALIZE.PY (300 DPI)
            images = pdf2image.convert_from_path(
                pdf_path, first_page=1, last_page=1, dpi=300
            )
            base_img = images[0].convert("RGB")
            draw_img = base_img.copy()
            draw = ImageDraw.Draw(draw_img)
            w, h = draw_img.size

            color_map = {
                "title_block": "#FF3366",
                "flat_pattern": "#33CC66",
                "orthographic_view": "#3399FF",
                "isometric_view": "#FFCC00",
                "section_view": "#CC33FF",
            }

            if regions_data and "regions" in regions_data:
                for region in regions_data["regions"]:
                    label = region.get("type", "region")
                    bbox = region.get("bbox", [0, 0, 0, 0])

                    if max(bbox) <= 1000:
                        x0, y0 = (bbox[0] / 1000) * w, (bbox[1] / 1000) * h
                        x1, y1 = (bbox[2] / 1000) * w, (bbox[3] / 1000) * h
                    else:
                        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

                    color = color_map.get(label, "#FF0000")
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=8)
                    draw.text(
                        (x0 + 15, y0 + 15),
                        label,
                        fill=color,
                    )

            st.image(
                draw_img,
                use_container_width=True,
                caption="Grounding Overlay (Stage 1)",
            )

    with col_right:
        st.subheader("2. Reasoning & Evidence")
        if class_data:
            st.markdown(
                f"**Classification Evidence:**\n> {class_data.get('class_evidence', 'N/A')}"
            )
        if count_data:
            st.markdown(
                f"**Bend Count Evidence:**\n> {count_data.get('bend_evidence', 'N/A')}"
            )