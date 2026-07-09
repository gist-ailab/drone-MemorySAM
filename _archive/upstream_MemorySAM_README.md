# <p align="center"><strong>MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation</strong></p>

<div align="center">
Chenfei Liao<sup>1</sup>, Xu Zheng<sup>1,2</sup><sup></sup> (Project lead), Yuanhuiyi Lyu<sup>1</sup>, Haiwei Xue<sup>5</sup>, Yihong Cao<sup>4</sup>, 
    
Jiawen Wang<sup>6</sup>, Kailun Yang<sup>4</sup>, Xuming Hu<sup>1,3</sup><sup></sup> (Corresponding author)
</div>

<div align="center">
<sup>1</sup>HKUST(GZ), <sup>2</sup>INSAIT, <sup>3</sup>HKUST, <sup>4</sup>HNU, <sup>5</sup>THU, <sup>6</sup>CUMTB
</div>

<div align="center">
    
[![arXiv](https://img.shields.io/badge/arXiv-2503.06700-brown?style=flat-square)](https://arxiv.org/abs/2503.06700)

</div>

## Abstract

Research has focused on Multi-Modal Semantic Segmentation (MMSS), where pixel-wise predictions are derived from multiple visual modalities captured by diverse sensors. Recently, the large vision model, Segment Anything Model 2 (SAM2), has shown strong zero-shot segmentation performance on both images and videos. When extending SAM2 to MMSS, two issues arise: 

🔥1. How can SAM2 be adapted to multi-modal data?

🔥2. How can SAM2 better understand semantics?

Inspired by cross-frame correlation in videos, we propose to treat multi-modal data as a sequence of frames representing the same scene. Our key idea is to **"memorize"** the modality-agnostic information and **"memorize"** the semantics related to the targeted scene. To achieve this, we apply SAM2’s memory mechanisms across multi-modal data to capture modality-agnostic features. Meanwhile, to memorize the semantic knowledge, we propose a training-only Semantic Prototype Memory Module (SPMM) to store category-level prototypes across training for facilitating SAM2’s transition from instance to semantic segmentation. A prototypical adaptation loss is imposed between global and local prototypes iteratively to align and refine SAM2's semantic understanding. 
Extensive experimental results demonstrate that our proposed MemorySAM outperforms SoTA methods by large margins on both synthetic and real-world benchmarks (65.38% on DELIVER, 52.88% on MCubeS). 
<div align="center">
    <img src="Figure/Figure_Overview.png" alt="Overview" width="600"/>
</div>

## News
⭐ If you find any problems in our code, please contact us! We will fix them as soon as possible! 

📧 lcfgreat624@gmail.com, cliao127@connect.hkust-gz.edu.cn

🚩 2025/3/10 Our paper has been online on Arxiv: https://arxiv.org/pdf/2503.06700

🚩 2025/3/13 We release the first version of our souce code! The weight will be released soon~

🚩 2025/4/23 We release the weights of MemorySAM on DELIVER dataset! Click this: [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/cliao127_connect_hkust-gz_edu_cn/ElwQ8vuvX7dKmxWVsYiWpSkBtwI4ErJV7grSUqKyRdVysw?e=cJbLcI)

🚩 2025/8/20 We release the weights of MemorySAM on MCubes dataset! Click this: [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/cliao127_connect_hkust-gz_edu_cn/ElwQ8vuvX7dKmxWVsYiWpSkBtwI4ErJV7grSUqKyRdVysw?e=cJbLcI)

## Framework

<div align="center">
    <img src="Figure/Figure_Framework.jpg" alt="Framework" width="1000"/>
</div>

## Code Structure Illustration

About the entire model part, we use the same code as standard SAM2, which is in  "MemorySAM/semseg/models/sam2". We clone these files from SAM's official code at the beginning of our project. The MemorySAM model code is mainly in "MemorySAM/semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py", with the model named as LoRA_Sam. Finally, in "train_sam2_lora_paper.py" (current main trainer; the original minimal "train_sam2_lora.py" is archived under "_archive/trainers/") we import this model and train.

## Preparation

### Environment Setup

1. Create a new Conda environment and activate it:
    ```bash
    conda create -n MMSS_SAM python=3.10 
    conda activate MMSS_SAM
    ```

2. Download SAM2's weight and upload it into the `semseg/models/sam2/checkpoints` directory. [Facebook Research SAM2 Repository](https://github.com/facebookresearch/sam2)

2. Install PyTorch and related libraries:
    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```

3. Install additional dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Navigate to the model directory and install:
    ```bash
    cd semseg/models/sam2
    pip install -e .
    ```

---

### B200 GPU (NVIDIA Blackwell, sm_100) 환경 설정

> **주의**: PyTorch ≤ 2.5 (cu121 빌드)는 B200 GPU(sm_100 아키텍처)를 지원하지 않습니다.  
> 아래 절차를 따르지 않으면 `no kernel image is available for execution on the device` 에러가 발생합니다.

#### 요구사항
- NVIDIA 드라이버 ≥ 555 (CUDA runtime 12.8 이상 포함)
- Python 3.10+

#### 설치 절차

1. 기본 환경 생성은 동일:
    ```bash
    conda create -n MMSS_SAM python=3.10
    conda activate MMSS_SAM
    ```

2. **PyTorch를 B200 호환 버전(cu128 빌드)으로 설치**:
    ```bash
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

3. 추가 의존성 설치:
    ```bash
    pip install -r requirements.txt
    ```

4. SAM2 설치:
    ```bash
    cd semseg/models/sam2
    pip install -e .
    ```

5. `timm` 버전 확인 (0.4.12 기준):
    ```bash
    # requirements.txt에 timm==0.4.12 지정되어 있어야 함
    python -c "import timm; print(timm.__version__)"
    ```

6. GPU 인식 확인:
    ```bash
    python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
    # 출력 예: 1  NVIDIA B200
    ```

#### 주요 차이점 (A100/H100 환경 vs B200 환경)

| 항목 | A100/H100 (기존) | B200 |
|------|-----------------|------|
| CUDA Compute Capability | sm_80 / sm_90 | sm_100 |
| PyTorch 버전 | `torch==2.3.1+cu121` | `torch==2.7.0+cu128` |
| `--index-url` | `whl/cu121` | `whl/cu128` |
| Flash Attention | `flash-attn` 기설치 | `pip install flash-attn` 재빌드 필요 |

---

## Run

### Data Preparation

1. Download the DELIVER/MCubes dataset and place it into the `data/` directory.

### Running the Model

2. Execute the following command to start the model:
    ```bash
    sh run_sam.sh
    ```
3. 🚨 <span style="color:red;">ATTENTION!!!</span> 🚨
 Line 233 in MemorySAM/semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py needs to be consistent with the number of modalities.

### MaCVi / MULTIAQUA 제출 (Multimodal Semantic Segmentation Challenge)
- [MaCVi 리더보드](https://macvi.org/workshop/cvpr/challenges/multimodal_semantic)는 **1-indexed** (1=Static, 2=Dynamic, 3=Water, 4=Sky)를 기대합니다.
- `--macvi` 플래그로 실행하면 eval_macvi/에 세그멘테이션 마스크만 저장됩니다 (val/test 공통 폴더):
  ```bash
  python val_multiaqua.py --cfg ... --mode val --model_path ... --macvi
  python val_multiaqua.py --cfg ... --mode test --model_path ... --macvi
  ```

## Acknowledgements

🤝 Our work is based on project of [DELIVER](https://github.com/jamycheung/DELIVER) and [SAM2](https://github.com/facebookresearch/sam2). Thanks to their contributions to this community!!!

🤝 Also, thanks to [DELIVER](https://github.com/jamycheung/DELIVER) and [MCubes](https://github.com/kyotovision-public/multimodal-material-segmentation) for their efforts to build such valuable datasets!!!

🤝 Moreover, thanks to [Xu Zheng](https://github.com/zhengxuJosh) (zhengxu128@gmail.com) for his great guidance and help for this project, who is the lead of this project!!!

## References

If you find this project helpful, please consider citing the following paper:
```
@misc{liao2025memorysammemorizemodalitiessemantics,
      title={MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation}, 
      author={Chenfei Liao and Xu Zheng and Yuanhuiyi Lyu and Haiwei Xue and Yihong Cao and Jiawen Wang and Kailun Yang and Xuming Hu},
      year={2025},
      eprint={2503.06700},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06700}, 
}
```


Thank you for your interest and support!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chenfei-Liao/MemorySAM&type=date&logscale&legend=top-left)](https://www.star-history.com/#Chenfei-Liao/MemorySAM&type=date&logscale&legend=top-left)
