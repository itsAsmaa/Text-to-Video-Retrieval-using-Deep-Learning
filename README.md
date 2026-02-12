# Video-Text Retrieval with CLIP on MSR-VTT

A comprehensive video-text retrieval system using CLIP embeddings and contrastive learning on the MSR-VTT dataset. This project implements both baseline and fine-tuned approaches for retrieving relevant videos based on text queries.

## Overview

This project tackles the video-text retrieval task using the Microsoft Research Video to Text (MSR-VTT) dataset. It leverages pre-trained CLIP models to encode both video frames and text captions into a shared embedding space, then fine-tunes the text encoder using contrastive learning to improve retrieval performance.

## Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the MSR-VTT dataset including caption statistics, video categories, and feature distributions
- **Dual CLIP Encoding**: Support for both standard CLIP (ViT-Base-Patch32) and OpenCLIP (ViT-H-14) models
- **Contrastive Fine-tuning**: Custom PyTorch implementation with projection layers and InfoNCE loss
- **Comprehensive Evaluation**: Metrics including Recall@1, Recall@5, Recall@10, Median Rank, and Mean Rank
- **Rich Visualizations**: Performance analysis, similarity distributions, category breakdowns, and qualitative examples
- **Query Length Analysis**: Performance evaluation across different caption lengths

## Dataset

**MSR-VTT** (Microsoft Research Video to Text)
- 10,000 web video clips
- 200,000 natural language descriptions (20 captions per video)
- 20 video categories
- Pre-extracted CLIP features (ViT-H-14)
- Test set evaluation

## Project Structure

```
├── Data Loading & EDA
│   ├── Video metadata analysis
│   ├── Caption statistics
│   └── Category distribution visualization
│
├── Text Encoding
│   ├── CLIP ViT-Base-Patch32 encoding
│   └── OpenCLIP ViT-H-14 encoding (1024-d)
│
├── Fine-tuning Pipeline
│   ├── Custom projection layers (512-d → 1024-d)
│   ├── Contrastive learning with InfoNCE loss
│   └── Training with hard negative mining
│
├── Evaluation
│   ├── Retrieval metrics (R@1, R@5, R@10, MedR, MeanR)
│   ├── Per-category performance analysis
│   └── Query length stratification
│
└── Visualization
    ├── Performance comparison plots
    ├── Similarity distribution analysis
    └── Qualitative success/failure cases
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install numpy pandas torch torchvision
pip install transformers accelerate
pip install open_clip_torch
pip install matplotlib seaborn tqdm
pip install pillow scikit-learn
```

## Usage

### 1. Data Preparation

Ensure you have the MSR-VTT dataset with the following structure:
```
MSR-VTT/
├── clip-features-vit-h14/    # Pre-extracted video features
├── test_videodatainfo.json   # Caption metadata
├── category.txt               # Category mappings
└── keyframes/                 # Video keyframes (optional)
```

### 2. Text Encoding

Encode captions using CLIP or OpenCLIP:
```python
# Using OpenCLIP ViT-H-14 (recommended)
text_embeds = encode_texts_openclip(captions, batch_size=256)
```

### 3. Fine-tuning

Train the text encoder with contrastive learning:
```python
model = TwoTowerFineTuneH14(base_text_dim=512, target_dim=1024)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_one_epoch(
    model, train_loader, optimizer, 
    device, temperature=0.07, 
    num_hard_negatives=8
)
```

### 4. Evaluation

Evaluate retrieval performance:
```python
metrics = compute_recall_at_k(text_feats, video_feats, ground_truth)
print(f"R@1: {metrics['R@1']:.2%}")
print(f"R@5: {metrics['R@5']:.2%}")
print(f"R@10: {metrics['R@10']:.2%}")
```

## Model Architecture

### Text Encoder Pipeline
1. **Base Model**: CLIP ViT-Base-Patch32 (frozen)
2. **Projection Layer**: MLP (512 → 1024 dimensions)
3. **Normalization**: L2 normalization for cosine similarity

### Training Details
- **Loss**: InfoNCE with temperature scaling (τ = 0.07)
- **Optimizer**: AdamW (lr = 1e-4)
- **Hard Negatives**: 8 per batch
- **Batch Size**: 256
- **Epochs**: 10-15

## Results

### Overall Performance

| Model | R@1 | R@5 | R@10 | MedR | MeanR |
|-------|-----|-----|------|------|-------|
| Baseline (CLIP ViT-Base) | 15.3% | 38.2% | 52.1% | 12.0 | 45.3 |
| Fine-tuned (+ Projection) | **22.7%** | **47.9%** | **61.5%** | **8.0** | **32.1** |

**Improvements**: +7.4% R@1, +9.7% R@5, +9.4% R@10

### Performance by Category

Top performing categories (Fine-tuned R@1):
- **Music & Performance**: 28.3%
- **Food & Cooking**: 26.5%
- **Sports & Fitness**: 24.1%

### Query Length Analysis

Performance improves with query length:
- **Short queries (1-10 words)**: R@1 = 18.2%
- **Medium queries (11-20 words)**: R@1 = 24.6%
- **Long queries (21+ words)**: R@1 = 27.9%

## Visualization Outputs

The project generates the following visualizations:

1. **video_category_distribution_test.png** - Category distribution in test set
2. **recall_comparison_chart.png** - Baseline vs fine-tuned performance
3. **category_recall_comparison.png** - Per-category retrieval performance
4. **query_length_recall.png** - Performance stratified by caption length
5. **baseline_similarity_distribution.png** - Similarity distributions (baseline)
6. **finetuned_similarity_distribution.png** - Similarity distributions (fine-tuned)
7. **success_case_examples.png** - Qualitative success cases
8. **failure_case_examples.png** - Qualitative failure analysis

## Technical Highlights

- **Dimension Alignment**: Projection layer bridges 512-d text embeddings to 1024-d video features
- **Hard Negative Mining**: Selects most confusing negatives per batch for robust training
- **Efficient Batching**: Handles large-scale embeddings with batch processing
- **Comprehensive Analysis**: Breaks down performance by multiple dimensions (category, length, etc.)

## Future Improvements

- [ ] Implement cross-modal attention mechanisms
- [ ] Experiment with larger CLIP models (ViT-L, ViT-G)
- [ ] Add temporal modeling for video sequences
- [ ] Implement bi-directional retrieval (text-to-video and video-to-text)
- [ ] Test on additional datasets (MSVD, ActivityNet Captions)

## Citation

If you use this code in your research, please cite the MSR-VTT dataset:

```bibtex
@inproceedings{xu2016msr,
  title={MSR-VTT: A large video description dataset for bridging video and language},
  author={Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5288--5296},
  year={2016}
}
```

## License

This project is available for educational and research purposes. Please ensure compliance with the MSR-VTT dataset license.

## Acknowledgments

- OpenAI CLIP team for pre-trained models
- Microsoft Research for the MSR-VTT dataset
- OpenCLIP contributors for expanded model variants

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project was developed for video-text retrieval research and demonstrates the effectiveness of contrastive fine-tuning for cross-modal alignment tasks.
