# Fine-tuning Pipeline Documentation

This README explains the comprehensive data processing and model fine-tuning pipeline implemented in `Whisper-ST1.ipynb`. The pipeline consists of three main phases: **Data Preprocessing**, **Dataset Splitting**, and **Model Training/Fine-tuning**.

## Overview

The pipeline processes multiple audio datasets (TORGO clinical data and YouTube naturalistic data) for dysarthria severity classification using deep learning models. It implements a sophisticated patient-level data splitting strategy and supports multiple pre-trained audio models with configurable fine-tuning approaches.

---

## Phase 1: Data Preprocessing & Metadata Creation

### Purpose
Creates unified metadata and audio file listings from multiple data sources with consistent labeling and file structure organization.

### Input Directory Structure
```
audio_datasets/
├── Youtube/
│   └── diarization outputs filtered consolidated/
│       ├── [audio_category]/           # e.g., "mild", "severe", etc.
│       │   └── [patient_folder]/       # Contains severity in folder name
│       │       └── segments _=2s/      # Audio segments ≥2 seconds
│       │           └── *.wav           # Audio files
└── TORGO/
    └── [speaker_id]/                   # e.g., "F03", "M01"
        └── Session[N]/                 # e.g., "Session1", "Session2"
            ├── wav_headMic/            # Primary audio source
            ├── wav_arrayMic/           # Fallback audio source
            └── prompts/                # Text transcriptions
                └── *.txt               # Corresponding transcripts
```

### Key Functions

#### `extract_patient_code(name)`
Generates unique patient identifiers from audio filenames:
- **TORGO**: Extracts patient codes like `M01_Session1` → `M01`
- **YouTube**: Converts `audio 1_speaker 3` → `A1S3`
- **Unknown**: Returns "Unknown" for unrecognized patterns

### Processing Logic

#### YouTube Data Processing
1. **Severity Extraction**: Scans folder names for severity keywords
2. **Label Mapping**: `{"normal":0, "mild":0, "mod":1, "moderate":1, "sev":2, "severe":2, "profound":3}`
3. **File Filtering**: Only processes `.wav` files in `segments _=2s` folders
4. **Quality Control**: Validates audio files using `librosa.load()`

#### TORGO Data Processing
1. **Manual Mapping**: Uses predefined severity labels for known patients:
   ```python
   togo_data = {
       "F3S1": {"severity": "Moderate", "label": 1},
       "F4S1": {"severity": "Mild", "label": 0},
       "M1S1": {"severity": "Profound", "label": 3},
       # ... etc
   }
   ```
2. **Audio Source Priority**: Prefers `wav_headMic/` over `wav_arrayMic/`
3. **Transcript Processing**:
   - Matches `.wav` files with `.txt` transcripts
   - Filters out image references (`.jpg`, `.jpeg`, `xxx`)
   - Removes bracketed annotations `[...]`
   - Requires transcripts >10 characters
4. **Duration Filtering**: Accepts audio between 0.5-40 seconds

### Output Files
- `audio_datasets/dataset_all_audio.csv`: Complete audio file inventory
- `audio_datasets/dataset_meta.csv`: Patient-level metadata summary

### Output Schema
**dataset_all_audio.csv**:
```csv
speaker_id,audio_name,label,path,session_code,source,transcript
```

**dataset_meta.csv**:
```csv
speaker_id,severity,label,count,session_code,source
```

---

## Phase 2: Advanced Dataset Splitting

### Purpose
Implements sophisticated patient-level train/validation/test splitting with multiple constraints to ensure balanced, representative splits while preserving all patients.

### Key Features
- **Patient-Level Splitting**: Each patient assigned to exactly one split
- **Class Balance Protection**: Prevents any severity class from dominating
- **Dominance Capping**: Limits class sizes to prevent skewed distributions
- **Split Ratio Enforcement**: Maintains approximate 80/10/10 train/val/test ratios
- **Universal Patient Preservation**: No patients are dropped from the dataset

### Algorithm Parameters

#### Core Ratios
- `train_ratio = 0.8` (80% training data)
- `valid_ratio = 0.1` (10% validation data)
- `test_ratio = 0.1` (10% test data)

#### Balancing Controls
- `class_slack = 0.50` (50% tolerance for class size variations)
- `dominance_cap = 1.35` (Limit class sizes to 135% of average)
- `split_slack = 0.10` (10% tolerance around target split ratios)
- `max_per_patient = 20` (Soft cap on samples per patient)
- `min_allowed_per_patient = 1` (Minimum samples to preserve per patient)

### Splitting Process

#### Step 1: Patient Sample Capping
- Applies soft cap (`max_per_patient = 20`) to prevent individual patients from dominating
- Maintains minimum of 1 sample per patient with non-zero original counts

#### Step 2: Class-Level Balancing
- Calculates bottleneck (smallest class size) and average class size
- Applies dominance cap: `min(original_size, 1.35 × average_class_size)`
- Uses one-by-one trimming algorithm to reduce class sizes fairly

#### Step 3: Split Assignment
- Randomly shuffles patients within each class
- Assigns patients to splits based on:
  - Target split ratios with slack tolerance
  - Current split fill levels
  - Load balancing across splits

#### Step 4: Constraint Enforcement
- Ensures each class has ≥1 patient in each split (when feasible)
- Moves smallest patients between splits if needed
- Preserves one-patient-one-split rule

### Output Structure
```
datasplit/
└── data[N]/                    # Auto-incremented folder (data1, data2, etc.)
    ├── summary.txt             # Split statistics and parameters
    ├── metadata.csv           # Patient-level split assignments
    └── all_audio_split.csv    # Complete audio inventory with splits
```

### Summary Output
The algorithm generates detailed statistics:
```
=== Final Results ===
Total samples (original):  1,663
Total samples (final):     823
Patients preserved:        98 / 98
Bottleneck (pre-trim):     164
Avg class total:           253

          Class  Train  Val  Test  Total
0             0    199   24    23    246
1             1    197   22    27    246
2             2    135   16    16    167
3             3    130   17    17    164
4  Combined All    661   79    83    823
```

---

## Phase 3: Model Training & Fine-tuning

### Purpose
Implements regression-based fine-tuning of pre-trained audio models for dysarthria severity assessment with configurable backbone unfreezing strategies.

### Supported Models
- **Wav2Vec2**: `facebook/wav2vec2-base`
- **HuBERT**: `facebook/hubert-base-ls960`
- **Whisper**: `openai/whisper-base`

### Architecture Overview

#### Model Configuration
- **Task Type**: Regression (continuous severity prediction)
- **Output**: Single numeric value (0.0-3.0 severity scale)
- **Loss Function**: Mean Squared Error (MSE)
- **Post-processing**: Round predictions to nearest integer class

#### Fine-tuning Strategies
```python
UNFREEZE_LAYERS = 1    # Options:
                       # -1: Full fine-tuning (unfreeze entire backbone)
                       #  0: Feature extraction (freeze backbone)
                       #  N: Partial fine-tuning (unfreeze last N layers)
```

### Data Processing Pipeline

#### Audio Preprocessing
1. **Loading**: Uses `torchaudio.load()` for robust audio loading
2. **Resampling**: Converts to model-specific sampling rate (16kHz for most models)
3. **Channel Reduction**: Averages multi-channel audio to mono
4. **Standardization**: Applies model feature extractor with:
   - `padding="max_length"` (pads shorter clips)
   - `max_length = 30 seconds` (truncates longer clips)
   - `truncation=True` (handles variable lengths)

#### Dataset Creation
```python
# HuggingFace Dataset structure
DatasetDict({
    'train': Dataset with training samples,
    'valid': Dataset with validation samples,
    'test': Dataset with test samples
})
```

### Training Configuration

#### Hardware & Performance
- **Device**: Auto-detects CUDA GPU or falls back to CPU
- **Mixed Precision**: FP16 enabled for memory efficiency
- **Batch Size**: 8 per device with gradient accumulation (effective batch = 16)

#### Optimization Settings
```python
TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=2,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)
```

#### Training Features
- **Early Stopping**: 20 epoch patience with 0% improvement threshold
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Evaluation Strategy**: Validates after each epoch
- **Metric**: Accuracy (rounded predictions vs. true labels)

### Layer Unfreezing Logic

#### Model Architecture Detection
```python
# Automatic backbone identification
if hasattr(model, 'hubert'):
    transformer_layers = model.hubert.encoder.layers
elif hasattr(model, 'wav2vec2'):
    transformer_layers = model.wav2vec2.encoder.layers
elif hasattr(model, 'whisper'):
    transformer_layers = model.encoder.layers
```

#### Unfreezing Strategy
1. **Freeze All**: Initially freeze all model parameters
2. **Selective Unfreezing**: Based on `unfreeze_last_n_layers`:
   - `-1`: Unfreeze entire transformer backbone
   - `0`: Keep backbone frozen (only train head)
   - `N > 0`: Unfreeze last N transformer layers
3. **Head Unfreezing**: Always unfreeze classifier and projector layers

### Output Structure
```
models/
└── [datasplit]_[model]_[timestamp]_regression/
    ├── checkpoints/           # Training checkpoints
    ├── best/                 # Best model + feature extractor
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── preprocessor_config.json
    ├── test_predictions.csv  # Detailed test results
    └── [visualizations]/     # Generated by save_visualisations()
```

### Evaluation & Analysis

#### Metrics Generated
- **Test Set Accuracy**: Rounded predictions vs. ground truth
- **Per-Patient Predictions**: Individual sample-level results
- **Confusion Matrix**: Class-wise performance breakdown
- **Training Curves**: Loss and accuracy over epochs

#### Test Results Format
```csv
speaker_id,audio_name,label,path,session_code,source,split,pred,true
```

### Custom Training Classes

#### RegressionTrainer
Extends HuggingFace `Trainer` with MSE loss:
```python
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels)
        return (loss, outputs) if return_outputs else loss
```

---

## File Dependencies

### Required Files
- `Whisper-ST1.ipynb`: Main pipeline implementation
- `visualisation_code/save_visualisations.py`: Results visualization
- `requirements.txt`: Python dependencies

### Input Requirements
- Raw audio datasets in expected directory structure
- TORGO patient severity mappings
- Sufficient storage for model checkpoints (~1-2GB per model)

### Output Guarantees
- All input patients preserved in final dataset
- Balanced class distributions within specified tolerances
- Reproducible splits (fixed random seed = 42)
- Complete model artifacts for inference

---

## Usage Examples

### Basic Pipeline Execution
```python
# 1. Run metadata creation (Phase 1)
# Processes audio_datasets/ → creates dataset_all_audio.csv

# 2. Run dataset splitting (Phase 2)
datasplit = split_dataset_by_patient(meta, save_path="datasplit/data1")

# 3. Run model training (Phase 3)
run_model_on_datasplit(
    model_ckpt="facebook/hubert-base-ls960",
    datasplit="data1",
    unfreeze_last_n_layers=1
)
```

### Configuration Options
```python
# Fine-tuning strategies
UNFREEZE_LAYERS = -1  # Full fine-tuning
UNFREEZE_LAYERS = 0   # Feature extraction only
UNFREEZE_LAYERS = 2   # Unfreeze last 2 layers

# Model selection
model_ckpt = "facebook/wav2vec2-base"        # Wav2Vec2
model_ckpt = "facebook/hubert-base-ls960"    # HuBERT
model_ckpt = "openai/whisper-base"           # Whisper

# Continue from checkpoint
model_ckpt = "models/data1_hubert-base-ls960_2025-08-30_18-03-30_regression/best"
```

This pipeline provides a complete, reproducible framework for dysarthria severity assessment model development with sophisticated data handling and flexible training configurations.