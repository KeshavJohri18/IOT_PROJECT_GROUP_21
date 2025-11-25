# SGLATrack Project Structure Summary

## Project Overview

**SGLATrack** (Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking) is an object tracking system optimized for aerial/UAV videos. It uses Vision Transformer (ViT) backbone with tracking-specific enhancements.

---

## Root Level Files

| File                    | Purpose                                                 |
| ----------------------- | ------------------------------------------------------- |
| `environment_sgla1.yml` | Conda environment specification for dependencies        |
| `README.md`             | Project documentation with setup and usage instructions |

---

## `/data` Directory

**Purpose**: Stores all training and testing datasets

### Subdirectories:

- `lasot/` - LaSOT training dataset
- `got10k/` - GOT-10K training dataset (train/ and val/ splits)
- `trackingnet/` - TrackingNet training dataset
- `UAV123/`, `UAV123_10fps/` - UAV tracking benchmark datasets
- `uavdt/`, `uavtrack/`, `V4RFlight112/` - Other UAV tracking datasets
- `otb/`, `vot*/` - General tracking benchmarks
- `nfs/`, `tc128/`, `tnl2k/` - Specialized tracking datasets
- `*_lmdb/` - LMDB-format versions of datasets (faster I/O)

---

## `/experiments` Directory

**Purpose**: Configuration files for different model variants

### Subdirectories:

- `sglatrack/` - SGLATrack configurations
  - `deit_distilled.yaml` - **Main config used by your project**
    - Specifies model hyperparameters (backbone type, training schedule, data augmentation)
    - Template/search region sizes
    - Loss weights and other training parameters

---

## `/lib` Directory

**Purpose**: Core library code for training and testing

### `/lib/config`

Configuration loading and management

- `sglatrack/config.py` - Config parser for SGLATrack YAML files

### `/lib/models`

Neural network model definitions

#### `/lib/models/layers`

**Reusable neural network components**
| File | Purpose |
|------|---------|
| `patch_embed.py` | Converts image into patch embeddings (divide image into patches) |
| `attn.py` | Multi-head self-attention mechanism |
| `attn_blocks.py` | Transformer blocks (attention + MLP) |
| `frozen_bn.py` | Frozen batch normalization layers |
| `head.py` | Prediction heads (corner predictor or center predictor) |
| `rpe.py` | Relative position encoding |

#### `/lib/models/sglatrack`

**Main tracker model architecture**
| File | Purpose |
|------|---------|
| `__init__.py` | Exports `build_sglatrack` function |
| `sglatrack.py` | **Core tracker class** - Combines backbone + head, runs forward/forward_test |
| `base_backbone.py` | **Vision Transformer backbone base** - Implements feature extraction and layer-adaptive mechanism |
| `deit.py` | **DeiT backbone variants** - Data-efficient distilled Vision Transformer (your config uses this) |
| `vit.py` | **ViT backbone variants** - Standard Vision Transformer (alternative backbone) |
| `utils.py` | Token manipulation utilities (combine_tokens, recover_tokens, window_partition) |

### `/lib/train`

**Training pipeline and utilities**
| File/Folder | Purpose |
|------|---------|
| `run_training.py` | Main training entry point - loads config and launches trainer |
| `train_script.py` | Training script for standard SGLATrack training |
| `train_script_distill.py` | Training script with knowledge distillation (teacher-student) |
| `base_functions.py` | Helper functions for building datasets, dataloaders, loss functions |
| `_init_paths.py` | Python path initialization |

#### `/lib/train/admin`

Training environment and metadata management
| File | Purpose |
|------|---------|
| `local.py` | Local environment paths (dataset dirs, checkpoint dirs, etc.) |
| `environment.py` | Environment configuration class |
| `tensorboard.py` | TensorBoard logging utilities |

#### `/lib/train/data`

Data processing for training
| File | Purpose |
|------|---------|
| `processing_utils.py` | Image cropping, padding, resizing utilities |
| `loader.py` | Data batch loading utilities |
| `transforms.py` | Image augmentation (flip, normalize, jitter, etc.) |
| Other files | Dataset-specific loaders and utilities |

#### `/lib/train/dataset`

Dataset implementations for training
| File | Purpose |
|------|---------|
| `base_video_dataset.py` | Base class for video tracking datasets |
| `lasot.py` | LaSOT dataset loader |
| `got10k.py` | GOT-10K dataset loader |
| `trackingnet.py` | TrackingNet dataset loader |
| `*_lmdb.py` | LMDB-format versions of above datasets (faster) |

#### `/lib/train/trainers`

Training orchestration
| File | Purpose |
|------|---------|
| `base_trainer.py` | Base trainer class - handles epoch loop, checkpoint save/load, gradient updates |
| `ltr_trainer.py` | LTR trainer implementation (Learning to Track) |

#### `/lib/train/actors`

Forward/backward pass implementation
| File | Purpose |
|------|---------|
| `sglatrack_actor.py` | Loss computation and backward pass for SGLATrack |

#### `/lib/train/data_specs`

Dataset specifications and splits
| File | Purpose |
|------|---------|
| `got10k_train_full_split.txt` | All 9335 GOT-10K training videos |
| `got10k_vot_train_split.txt` | GOT-10K subset allowed by VOT challenge |
| `lasot_train_split.txt` | LaSOT training video list |
| `trackingnet_classmap.txt` | TrackingNet sequence-to-class mapping |
| `README.md` | Documentation of data split files |

### `/lib/test`

**Testing/inference pipeline**

#### `/lib/test/tracker`

Core inference code
| File | Purpose |
|------|---------|
| `basetracker.py` | Base tracker interface (initialize, track, predict) |
| `sglatrack.py` | **SGLATrack inference implementation** - Runs forward_test, generates predictions |
| `data_utils.py` | Preprocessor for image normalization during inference |
| `vis_utils.py` | Visualization utilities |

#### `/lib/test/evaluation`

Dataset and evaluation infrastructure
| File | Purpose |
|------|---------|
| `data.py` | Sequence/dataset base classes |
| `datasets.py` | Dataset registry and loader |
| `tracker.py` | Tracker runner - loops through frames, saves predictions |
| `running.py` | Dataset evaluation orchestration |
| `environment.py` | Evaluation environment configuration |
| `local.py` | Local paths for test datasets and results |
| `*dataset.py` | Specific dataset loaders (OTB, LaSOT, UAV123, etc.) |

#### `/lib/test/analysis`

Result analysis and visualization
| File | Purpose |
|------|---------|
| `plot_results.py` | Generate evaluation plots and metrics (precision, success rate graphs) |
| `extract_results.py` | Extract raw tracking results |

#### `/lib/test/parameter`

Test configuration
| File | Purpose |
|------|---------|
| `sglatrack.py` | Test parameters for SGLATrack (template size, search size, checkpoint path) |

#### `/lib/test/utils`

Utility functions for testing
| File | Purpose |
|------|---------|
| `hann.py` | Hann window for response map smoothing |
| `load_text.py` | Load text annotation files |
| `TrackerParams` class | Parameter container for tracker |

### `/lib/utils`

General utilities used across project
| File | Purpose |
|------|---------|
| `box_ops.py` | Bounding box operations (conversion, IoU, GIoU loss) |
| `ce_utils.py` | Contrastive learning utilities |
| `focal_loss.py` | Focal loss for classification |
| `heapmap_utils.py` | Heatmap processing utilities |
| `lmdb_utils.py` | LMDB dataset utilities |
| `misc.py` | Miscellaneous utilities |
| `merge.py` | Model merging utilities |
| `tensor.py` | Tensor utilities |
| `variable_hook.py` | PyTorch gradient hook utilities |

### `/lib/vis`

Visualization utilities
| File | Purpose |
|------|---------|
| `plotting.py` | Plot generation utilities |
| `utils.py` | Visualization helper functions |
| `visdom_cus.py` | Visdom visualization server integration (deprecated) |

---

## `/output` Directory

**Purpose**: Stores all trained models and test results

### `/output/checkpoints/train`

- **Stores trained model weights** (`.pth.tar` files)
- Structure: `train/sglatrack/sglatrack/sglatrack_ep*.pth.tar`
- Each checkpoint contains: model weights, optimizer state, epoch, training stats

### `/output/test`

**Test results and visualizations**

- `tracking_results/` - Raw bounding box predictions for each video frame
- `result_plots/` - Generated evaluation plots (precision, success curves, comparisons)
- `segmentation_results/` - Segmentation masks (if applicable)

---

## `/tracking` Directory

**Purpose**: User-facing scripts (entry points you run directly)

| Script                           | Purpose                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `test.py`                        | **Test tracker on benchmark datasets** - Main inference script |
| `train.py`                       | **Launch training** - Handles single/multi-GPU setup           |
| `analysis_results.py`            | **Analyze test results** - Generate plots and metrics          |
| `analysis_results.ipynb`         | Jupyter notebook version of analysis                           |
| `video_demo.py`                  | Run tracker on custom video files                              |
| `vis_results.py`                 | Visualize tracking results with bounding boxes                 |
| `profile_model.py`               | Measure model speed and computational efficiency               |
| `convert_transt.py`              | Convert model formats                                          |
| `create_default_local_file.py`   | Setup initial local configuration                              |
| `download_pytracking_results.py` | Download benchmark results                                     |
| `pre_read_datasets.py`           | Pre-process datasets for faster loading                        |
| `_init_paths.py`                 | Python path initialization                                     |

### Typical Usage:

```bash
# Training
python tracking/train.py --script sglatrack --config deit_distilled --save_dir ./output

# Testing
python tracking/test.py sglatrack sglatrack --dataset_name uav123_10fps --threads 8

# Analysis
python tracking/analysis_results.py
```

---

## Key File Relationships

### Training Flow:

```
tracking/train.py
  └─> lib/train/run_training.py
       └─> lib/train/train_script.py
            ├─> lib/train/base_functions.py (build datasets & dataloaders)
            ├─> lib/models/sglatrack/build_sglatrack() (create model)
            ├─> lib/train/trainers/LTRTrainer (training loop)
            └─> lib/train/actors/sglatrackActor (loss computation)
```

### Testing Flow:

```
tracking/test.py
  └─> lib/test/evaluation/running.py
       └─> lib/test/evaluation/tracker.py (frame-by-frame tracking)
            └─> lib/test/tracker/sglatrack.py (inference)
                 └─> lib/models/sglatrack/sglatrack.py (forward_test)
```

### Analysis Flow:

```
tracking/analysis_results.py
  └─> lib/test/analysis/plot_results.py (generate evaluation metrics)
```

---

## Configuration System

**All behavior controlled via YAML files** in `/experiments/sglatrack/`:

Example: `deit_distilled.yaml` contains:

- `MODEL.BACKBONE.TYPE` - Which backbone (deit_tiny_distilled, vit_base, etc.)
- `DATA.TEMPLATE.SIZE`, `DATA.SEARCH.SIZE` - Input sizes
- `TRAIN.LR`, `TRAIN.EPOCH` - Training hyperparameters
- `TEST.TEMPLATE_FACTOR`, `TEST.SEARCH_FACTOR` - Inference parameters

**No code changes needed** - just modify YAML to experiment!

---

## Quick Reference

### To Train:

- Edit config: `/experiments/sglatrack/deit_distilled.yaml`
- Run: `python tracking/train.py --script sglatrack --config deit_distilled`
- Checkpoints saved to: `/output/checkpoints/train/`

### To Test:

- Ensure checkpoint exists at: `/output/checkpoints/train/sglatrack/sglatrack/sglatrack_ep*.pth.tar`
- Ensure dataset in: `/data/UAV123_10fps/` (or other dataset)
- Run: `python tracking/test.py sglatrack sglatrack --dataset_name uav123_10fps`
- Results saved to: `/output/test/tracking_results/`

### To Analyze:

- Run: `python tracking/analysis_results.py`
- Plots generated in: `/output/test/result_plots/`

---

## Model Architecture

```
Input Images (template + search)
  ├─ Patch Embedding (lib/models/layers/patch_embed.py)
  ├─ Vision Transformer Backbone (lib/models/sglatrack/deit.py or vit.py)
  │   ├─ Self-Attention Layers (lib/models/layers/attn.py)
  │   ├─ Transformer Blocks (lib/models/layers/attn_blocks.py)
  │   └─ Relative Position Encoding (lib/models/layers/rpe.py)
  └─ Prediction Head (lib/models/layers/head.py)
      ├─ Corner Predictor (predicts corner coordinates)
      └─ Center Predictor (predicts center + width/height)

Output: Bounding Box Predictions
```

---

## Important Environment Variables

- `CUDA_VISIBLE_DEVICES` - Which GPUs to use (empty = CPU only)
- `MPLBACKEND` - Matplotlib backend (set to 'Agg' for non-interactive plotting)

---

## Dependencies

Key Python packages (see `environment_sgla1.yml`):

- `torch` - Deep learning framework
- `torchvision` - Vision utilities
- `timm` - Transformer models library
- `opencv` - Image processing
- `matplotlib` - Plotting
- `numpy`, `scipy` - Numerical computing
- `pyyaml` - YAML config parsing
