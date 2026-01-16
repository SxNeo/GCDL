# LP
lithology prediction
Usage
1. Data Preparation
Organize your data in the following structure:
data/raw/
├── seismic_poststack.sgy      # Post-stack seismic data (SEG-Y format)
├── impedance_model.sgy        # Acoustic impedance model (optional, for physics constraint)
├── well1_labels.xlsx          # Well 1 lithology labels
├── well2_labels.xlsx          # Well 2 lithology labels
└── well3_labels.xlsx          # Well 3 lithology labels
Label files should contain two columns:

Column 1: Time (ms)
Column 2: Lithology label (0/1: Sandstone, 2: Mudstone)

2. Data Preprocessing
Configure the parameters in data_preprocessing.py:
pythonclass Config:
    INPUT_PATH = "./data/raw"
    OUTPUT_PATH = "./data/processed"
    WINDOW_RADIUS = 14  # Results in 29-point samples
    NORMALIZE_MODE = "trace"  # Options: "none", "trace", "rms_global", "max_global"
Run preprocessing:
bashpython data_preprocessing.py
This will generate:

Training samples from well locations
Prediction samples for the entire seismic volume

3. Model Training
Configure training parameters in train.py:
pythonclass Config:
    # Model parameters
    FILTERS = 64
    NUM_ENCODER_LAYERS = 2
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # Physics constraint (Stage 2)
    ENABLE_PHYSICS_FINETUNE = True
    LAMBDA_SEISMIC = 1.0
Run training:
bashpython train.py
4. Output Files
After training, the following files are generated:
output/gcdl_results_trace_norm/
├── metrics.json                          # Evaluation metrics
├── training_history_stage1.png           # Training curves
├── confusion_matrix_train_stage1.png     # Training confusion matrix
├── confusion_matrix_test_stage1.png      # Test confusion matrix
├── roc_curve_stage1.png                  # ROC curve
├── predictions.csv                       # Prediction results
├── predictions.npy                       # Prediction labels
├── probabilities.npy                     # Prediction probabilities
└── model/
    ├── final_weights.h5                  # Model weights
    ├── config.json                       # Model configuration
    └── encoder.pkl                       # Label encoder
