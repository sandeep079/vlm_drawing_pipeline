import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("reference_samples", exist_ok=True)


def create_cad_pdf(filename, title, material, process, is_tube=False, bends=0):
    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300)
    ax.set_xlim(0, 297)
    ax.set_ylim(0, 210)
    ax.axis("off")

    # Outer Border
    ax.add_patch(
        patches.Rectangle(
            (5, 5), 287, 200, fill=False, edgecolor="black", lw=2
        )
    )

    # Title Block
    ax.add_patch(
        patches.Rectangle(
            (170, 5), 122, 45, fill=False, edgecolor="black", lw=1.5
        )
    )
    ax.plot([170, 292], [35, 35], color="black", lw=1)
    ax.plot([170, 292], [20, 20], color="black", lw=1)
    ax.plot([225, 225], [5, 35], color="black", lw=1)

    ax.text(173, 40, f"TITLE: {title}", fontsize=9, weight="bold")
    ax.text(173, 26, f"MAT: {material}", fontsize=8)
    ax.text(228, 26, "THICKNESS: 2.0 MM", fontsize=8)
    ax.text(173, 11, f"DWG NO: {filename.split('.')[0].upper()}", fontsize=8)
    ax.text(228, 11, f"PROCESS: {process}", fontsize=8)

    # Views
    if is_tube:
        # Tube Profile View
        ax.add_patch(
            patches.Rectangle(
                (40, 80), 70, 70, fill=False, edgecolor="black", lw=2
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (48, 88), 54, 54, fill=False, edgecolor="black", lw=1.5
            )
        )
        ax.text(40, 160, "SQUARE TUBE SECTION (ROHR 80X80)", fontsize=9, weight="bold")
    else:
        # Orthographic View
        ax.add_patch(
            patches.Rectangle(
                (25, 75), 80, 80, fill=False, edgecolor="black", lw=1.5
            )
        )
        ax.text(25, 160, "ORTHOGRAPHIC VIEW", fontsize=9, weight="bold")

        # Flat Pattern
        ax.add_patch(
            patches.Rectangle(
                (135, 75), 110, 80, fill=False, edgecolor="black", lw=1.5
            )
        )
        ax.text(
            135, 160, "FLAT PATTERN (ABWICKLUNG)", fontsize=9, weight="bold"
        )

        # Draw Bend Lines if specified
        if bends == 2:
            ax.plot([165, 165], [75, 155], color="red", linestyle="--", lw=1.5)
            ax.plot([215, 215], [75, 155], color="red", linestyle="--", lw=1.5)
        elif bends == 4:
            ax.plot([155, 155], [75, 155], color="red", linestyle="--", lw=1.5)
            ax.plot([180, 180], [75, 155], color="red", linestyle="--", lw=1.5)
            ax.plot([205, 205], [75, 155], color="red", linestyle="--", lw=1.5)
            ax.plot([230, 230], [75, 155], color="red", linestyle="--", lw=1.5)

    path = os.path.join("reference_samples", filename)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


# 1. Sheet Metal U-Bracket (2 Bends)
create_cad_pdf(
    "001_sample_bracket.pdf",
    "BRACKET MOUNT",
    "1.4301",
    "ABWICKLUNG",
    bends=2,
)

# 2. Square Tube Profile (0 Bends, Tube Class)
create_cad_pdf(
    "002_sample_tube.pdf",
    "SQUARE TUBE PROFILE",
    "1.0038",
    "ROHR 80X80",
    is_tube=True,
)

# 3. Flat Adapter Plate (0 Bends, Chamfered)
create_cad_pdf(
    "003_sample_plate.pdf",
    "ADAPTER PLATE 5X45",
    "ALMG3",
    "FLAT BLECH",
    bends=0,
)

# 4. Complex Sheet Component (4 Bends)
create_cad_pdf(
    "004_sample_complex.pdf",
    "CHASSIS BRACKET",
    "1.4301",
    "ABWICKLUNG",
    bends=4,
)
