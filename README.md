# MMSS-SAM

## Overview
<p align="center"><strong>MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation</strong></p>
---

## Preparation

### Environment Setup

1. Create a new Conda environment and activate it:
    ```bash
    conda create -n MMSS_SAM python=3.10 
    conda activate MMSS_SAM
    ```

2. Download SAM2 and upload it into the `semseg/models/` directory. [Facebook Research SAM2 Repository](https://github.com/facebookresearch/sam2)
   Add the `sam_lora_image_encoder_seg.py` to the `semseg/models/sam2/sam2/` directory.
   Add the `sam2_base` to the `semseg/models/sam2/sam2/modeling/` directory.

2. Install PyTorch and related libraries:
    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```

3. Install additional dependencies:
    ```bash
    pip install pycocotools
    pip install safetensors
    pip install icecream
    pip install -r requirements.txt
    ```

4. Navigate to the model directory and install:
    ```bash
    cd semseg/models/sam2
    pip install -e .
    ```

---

## Run

### Data Preparation

1. Download the DELIVER dataset and place it into the `data/` directory.

### Running the Model

2. Execute the following command to start the model:
    ```bash
    sh run_sam.sh
    ```
