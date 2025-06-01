# Beyblade Battle Analyzer

A sophisticated AI-powered system for analyzing Beyblade battles using computer vision and machine learning. This project leverages YOLO object detection to track Beyblades in real-time and provides comprehensive battle analytics.

## 🎯 Features

- **Real-time Beyblade Detection**: Custom-trained YOLO model for accurate Beyblade detection
- **Battle Analysis**: Advanced algorithms to track movement patterns, spinning states, and battle outcomes
- **Arena Bounds Selection**: Interactive tool to define battle arena boundaries
- **Comprehensive Statistics**: Detailed battle summaries including winner determination and performance metrics
- **Video Processing**: Annotated video output with real-time battle visualization
- **Data Export**: CSV and JSON export capabilities for further analysis
- **Multiple Model Sizes**: Support for nano, small, and medium YOLO models

## 🛠️ Architecture

The project follows a modular pipeline architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Ingestion  │───▶│ Model Training   │───▶│ Video Analysis  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Battle Summary  │◀───│ Arena Selection  │◀───│ Frame Processing│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
beyblade_battle_analyzer/
├── main.py                     # Main entry point
├── requirements.txt             # Python dependencies
├── config/
│   └── config.yaml             # Configuration settings
├── src/beyblade_battle_analyzer/
│   ├── components/             # Core components
│   │   ├── data_ingestion.py   # Dataset download from Roboflow
│   │   ├── model_training.py   # YOLO model training
│   │   ├── beyblade_detector.py # Object detection
│   │   ├── battle_analyzer.py  # Battle logic and tracking
│   │   ├── video_processor.py  # Video processing
│   │   ├── battle_summary.py   # Battle result analysis
│   │   └── data_manager.py     # Data export and management
│   ├── pipelines/              # Processing pipelines
│   ├── config/                 # Configuration management
│   ├── entity/                 # Data classes and entities
│   └── utils/                  # Utility functions
├── artifacts/                  # Generated outputs
│   ├── training/               # Model weights and training data
│   ├── video_processor/        # Processed videos and results
│   └── input_video/           # Input video files
└── logs/                      # Application logs
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd beyblade_battle_analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration**:
   - Edit `config/config.yaml` to set your Roboflow API key
   - Adjust paths and parameters as needed

## 📋 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Roboflow
- Pandas
- NumPy

See `requirements.txt` for complete dependency list.

## 🎮 Usage

### Training Pipeline

Train a custom YOLO model on Beyblade data:

```bash
python main.py --pipeline training
```

This will:
1. Download the dataset from Roboflow
2. Train a YOLO model (nano/small/medium versions available)
3. Save trained weights to `artifacts/training/`

### Video Analysis Pipeline

Analyze Beyblade battles in video:

```bash
python main.py --pipeline video_analyzer
```

This will:
1. Launch arena bounds selector (interactive)
2. Process video with trained model
3. Generate battle analysis and statistics
4. Export results to CSV/JSON
5. Create annotated video output

### Device Configuration

Both training and inference support GPU/CPU device selection:

```bash
# Use automatic device selection (default)
python main.py --pipeline training --device auto

# Force CPU usage
python main.py --pipeline training --device cpu

# Use CUDA GPU (if available)
python main.py --pipeline training --device cuda

# Use specific GPU
python main.py --pipeline training --device cuda:0
python main.py --pipeline video_analyzer --device cuda:1
```

Device options:
- `auto`: Automatically select best available device (GPU if available, otherwise CPU)
- `cpu`: Force CPU usage
- `cuda`: Use default CUDA GPU
- `cuda:0`, `cuda:1`, etc.: Use specific GPU device

## ⚙️ Configuration

Key configuration options in `config/config.yaml`:

### Device Configuration

The system supports flexible device selection for both training and inference:

#### Configuration File
Set the default device in `config/config.yaml`:
```yaml
model_training:
  device: auto    # Options: auto, cpu, cuda, cuda:0, cuda:1, etc.

beyblade_detector:
  device: auto    # Options: auto, cpu, cuda, cuda:0, cuda:1, etc.
```

#### Command Line Override
Override device settings at runtime:
```bash
# Use automatic device selection
python main.py --pipeline training --device auto

# Force CPU usage (useful for debugging or CPU-only systems)
python main.py --pipeline training --device cpu

# Use CUDA GPU if available
python main.py --pipeline video_analyzer --device cuda

# Use specific GPU device
python main.py --pipeline training --device cuda:0
python main.py --pipeline video_analyzer --device cuda:1
```

#### Device Options
- `auto`: Automatically selects the best available device (CUDA GPU if available, otherwise CPU)
- `cpu`: Forces CPU usage regardless of GPU availability
- `cuda`: Uses the default CUDA device if available, falls back to CPU if not
- `cuda:N`: Uses specific GPU device N (e.g., `cuda:0`, `cuda:1`, etc.)

#### Device Detection
The system automatically:
- Detects CUDA availability and GPU count
- Validates device configurations
- Falls back to CPU if requested GPU is unavailable
- Logs device information and selection reasoning

### Data Ingestion
```yaml
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_url: https://universe.roboflow.com/beyblade-djcpy/beyblade-dtd3d/dataset/14
  roboflow_api_key: YOUR_API_KEY
  dataset_version: yolov11
  dataset_name: Beyblade-14
```

### Model Training
```yaml
model_training:
  root_dir: artifacts/training
  weight_model: yolo11n.pt    # nano/small/medium options
  epochs: 100
  patience: 25
  image_size: 640
  project_name: beyblade_detector-nano
  device: auto                # Device selection: auto/cpu/cuda/cuda:0/cuda:1
```

### Video Processing
```yaml
video_processor:
  input_video_path: artifacts/input_video/beyblade_trim.mov
  output_video_path: artifacts/video_processor/outputs
  visualization: true
```

### Detection Parameters
```yaml
beyblade_detector:
  model_path: artifacts/training/beyblade_detector-nano/weights/best.pt
  confidence_threshold: 0.5
  image_size: 640
  device: auto                # Device selection: auto/cpu/cuda/cuda:0/cuda:1
```

## 📊 Battle Analysis Features

### Tracking System
- **Multi-object tracking**: Track multiple Beyblades simultaneously
- **Velocity calculation**: Monitor spinning speed and movement
- **State detection**: Determine spinning vs. stopped states
- **Arena boundaries**: Filter detections within defined arena

### Battle States
- `STARTING`: Initial detection phase
- `ACTIVE`: Active battle in progress
- `ENDING`: Battle conclusion detected
- `FINISHED`: Final results available

### Winner Determination
- **Last standing**: Beyblade that spins longest
- **Performance scoring**: Based on stability, movement quality, and survival time
- **Movement patterns**: Analysis of circular vs. erratic movement

### Statistics Generated
- Battle duration
- Individual Beyblade performance metrics
- Movement patterns and velocity analysis
- Winner analysis with detailed reasoning
- Frame-by-frame tracking data

## 📈 Output Data

### CSV Exports
- `battle_summaries_*.csv`: Battle results and statistics
- `detection_data_*.csv`: Frame-by-frame detection data

### JSON Export
- Complete dataset with all analysis results
- Hierarchical data structure for advanced analysis

### Video Output
- Annotated video with:
  - Beyblade detection boxes
  - Arena boundaries
  - Battle state indicators
  - Real-time statistics overlay

## 🎯 Arena Bounds Selection

Interactive tool for defining battle arena:

1. **Mouse Controls**:
   - Click and drag to select rectangular area
   - ESC: Cancel selection
   - ENTER: Confirm selection
   - R: Reset selection

2. **Visual Feedback**:
   - Real-time boundary preview
   - Arena dimensions display
   - Corner markers for precision

## 🔧 Model Training Details

### Supported Models
- **YOLO11n**: Nano version (fastest inference)
- **YOLO11s**: Small version (balanced speed/accuracy)
- **YOLO11m**: Medium version (highest accuracy)

### Training Parameters
- **Epochs**: 100 (configurable)
- **Patience**: 25 (early stopping)
- **Image Size**: 640x640
- **Augmentation**: Automatic YOLO augmentations
- **Learning Rate**: Cosine learning rate scheduling

### Output Artifacts
- `best.pt`: Best model weights
- `results.csv`: Training metrics
- `confusion_matrix.png`: Model performance visualization
- Training/validation plots

## 📝 Logging

Comprehensive logging system:
- Application logs in `logs/beyblade_battle_analyzer.log`
- Pipeline progress tracking
- Error handling and debugging information
- Performance metrics
