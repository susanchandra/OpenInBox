import json
import re

files = [
    "c:/Users/SRI/Downloads/Bellman_breakers/Bellman_breakers/colab_training/cells_1_to_4.py",
    "c:/Users/SRI/Downloads/Bellman_breakers/Bellman_breakers/colab_training/cells_5_to_9.py"
]

cells = []

# Add an intro markdown cell
cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "intro"},
    "source": [
        "# OpenInbox — GRPO Training Pipeline\n",
        "\n",
        "This notebook contains the complete end-to-end training pipeline for the OpenInbox environment using HuggingFace TRL (GRPO) and a 4-bit quantized Qwen2.5-1.5B model.\n",
        "\n",
        "It connects directly to the live HuggingFace Space environment endpoint."
    ]
})

for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split by the # ======================= CELL X separator
    parts = re.split(r'# ===========================================================================\n# CELL \d+ — (.*?)\n# ===========================================================================\n# %%', content)
    
    # The first part is the header (which we can skip if empty or just comment)
    # The subsequent parts alternate: title, code, title, code...
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        code = parts[i+1].strip()
        
        # Add markdown cell for the title
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}"]
        })
        
        # Format code lines
        code_lines = [line + "\n" for line in code.split("\n")]
        # Remove trailing newline from last line to be clean
        if code_lines:
            code_lines[-1] = code_lines[-1].rstrip("\n")
            
        # Add code cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        })

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("c:/Users/SRI/Downloads/Bellman_breakers/Bellman_breakers/OpenInbox_GRPO_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
