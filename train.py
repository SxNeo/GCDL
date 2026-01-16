"""
Geophysically Constrained Deep Learning (GCDL) for Lithology Prediction

This module implements the GCDL method which combines:
- Dual-branch Temporal Convolutional Network (TCN)
  - Local branch: Causal convolution for short-range dependencies
  - Global branch: Dilated convolution for long-range dependencies
- Transformer encoder for global contextual relationships
- Geophysical constraint loss function based on seismic forward modeling

Training Strategy:
- Stage 1: Standard cross-entropy training
- Stage 2: Physics-constrained fine-tuning (optional)

Author: GCDL Research Team
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import time
import random
from collections import defaultdict
from typing import Tuple, Dict, List, Optional

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Input,
    Conv1D, BatchNormalization, ReLU, SpatialDropout1D, Lambda,
    GlobalAveragePooling1D, Layer
)
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score,
    recall_score, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False
    print("Warning: segyio not available. Physics constraint features will be limited.")


# ======================= Configuration =======================
class Config:
    """Configuration parameters for model training."""
    
    # Paths (modify according to your setup)
    INPUT_PATH = "./data/raw"
    PROCESSED_PATH = "./data/processed"
    OUTPUT_PATH = "./output"
    
    # Data parameters
    NORMALIZE_MODE = "trace"  # "none", "trace", "rms_global", "max_global"
    WINDOW_RADIUS = 14
    SAMPLE_LENGTH = 2 * WINDOW_RADIUS + 1  # 29
    
    # Model parameters
    N_CLASSES = 2
    CLASSES = {0: 'Sandstone', 1: 'Mudstone'}
    FILTERS = 64
    NUM_ENCODER_LAYERS = 2
    
    # Training parameters - Stage 1
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 4
    
    # Training parameters - Stage 2 (Physics fine-tuning)
    ENABLE_PHYSICS_FINETUNE = True
    FINETUNE_EPOCHS = 5
    FINETUNE_LR = 0.0001
    LAMBDA_SEISMIC = 1.0
    ALPHA_ENVELOPE = 0.3
    TRACES_PER_BATCH = 2
    
    # Random seed
    RANDOM_SEED = 50
    
    # Spatial parameters (for physics constraint)
    N_INLINES = 151
    N_XLINES = 201
    FIRST_INLINE_ID = 2327
    FIRST_XLINE_ID = 3678
    INCREMENT = 2


# Set plotting style
plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})


# ======================= Random Seed =======================
def set_all_seeds(seed: int = 50) -> None:
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass
    
    print(f"Random seed set to: {seed}")


# ======================= Label Mapping =======================
def map_labels_to_binary(y: np.ndarray) -> np.ndarray:
    """
    Map multi-class labels to binary lithology labels.
    
    Args:
        y: Original labels (0, 1 -> Sandstone; 2 -> Mudstone)
    
    Returns:
        Binary labels (0: Sandstone, 1: Mudstone)
    """
    y_binary = np.copy(y)
    y_binary[y <= 1] = 0  # Sandstone
    y_binary[y == 2] = 1  # Mudstone
    return y_binary


# ======================= TCN Modules =======================
def residual_block(inputs: tf.Tensor, 
                   filters: int, 
                   kernel_size: int, 
                   dilation_rate: int, 
                   dropout_rate: float) -> tf.Tensor:
    """
    TCN residual block with causal convolution.
    
    Args:
        inputs: Input tensor
        filters: Number of convolutional filters
        kernel_size: Kernel size
        dilation_rate: Dilation rate (1 for pure causal, >1 for dilated)
        dropout_rate: Dropout rate
    
    Returns:
        Output tensor
    """
    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(dropout_rate)(x)
    
    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(dropout_rate)(x)
    
    # Residual connection
    if inputs.shape[-1] != filters:
        residual = Conv1D(filters, kernel_size=1, padding='same')(inputs)
    else:
        residual = inputs
    
    return Add()([x, residual])


# ======================= Transformer Modules =======================
class PositionalEncoding(Layer):
    """Sinusoidal positional encoding layer."""
    
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Compute positional encoding
        angle_rads = self._get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def _get_angles(self, pos, i, d_model):
        return pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    
    def get_config(self):
        return {**super().get_config(), 'max_len': self.max_len, 'd_model': self.d_model}


def transformer_encoder_block(x: tf.Tensor, 
                             d_model: int, 
                             num_heads: int, 
                             dff: int, 
                             dropout_rate: float = 0.1) -> tf.Tensor:
    """
    Single Transformer encoder block.
    
    Args:
        x: Input tensor
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward network dimension
        dropout_rate: Dropout rate
    
    Returns:
        Output tensor
    """
    # Multi-head self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate
    )(x, x)
    
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed-forward network
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def transformer_encoder(x: tf.Tensor, 
                       num_layers: int, 
                       d_model: int, 
                       num_heads: int, 
                       dff: int, 
                       dropout_rate: float = 0.1, 
                       max_len: int = 1000) -> tf.Tensor:
    """
    Complete Transformer encoder.
    
    Args:
        x: Input tensor
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward network dimension
        dropout_rate: Dropout rate
        max_len: Maximum sequence length for positional encoding
    
    Returns:
        Output tensor
    """
    if x.shape[-1] != d_model:
        x = Dense(d_model)(x)
    
    x = PositionalEncoding(max_len, d_model)(x)
    x = Dropout(dropout_rate)(x)
    
    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model, num_heads, dff, dropout_rate)
    
    return x


# ======================= Model Construction =======================
def build_gcdl_model(input_shape: Tuple[int, int], 
                     n_classes: int, 
                     filters: int = 64, 
                     num_encoder_layers: int = 2) -> Model:
    """
    Build the GCDL (Geophysically Constrained Deep Learning) model.
    
    Architecture:
    - Dual-branch TCN:
        - Local branch: Causal convolution (dilation=1) for short-range dependencies
        - Global branch: Dilated convolution for long-range dependencies
    - Transformer encoders for global contextual modeling
    - Weighted feature fusion
    - Classification head
    
    Args:
        input_shape: Input shape (sequence_length, features)
        n_classes: Number of output classes
        filters: Number of convolutional filters
        num_encoder_layers: Number of Transformer encoder layers
    
    Returns:
        Compiled Keras model
    """
    print(f"\nBuilding GCDL model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Filters: {filters}")
    
    inputs = Input(shape=input_shape)
    
    # ==================== Dual-branch TCN ====================
    
    # Global branch: Dilated convolution for long-range dependencies
    # Dilation factors: 2, 4, 8 for exponentially increasing receptive field
    global_branch = inputs
    for i in range(3):
        global_branch = residual_block(
            global_branch, 
            filters=filters, 
            kernel_size=3,
            dilation_rate=2**(i+1),  # 2, 4, 8
            dropout_rate=0.3
        )
    
    # Local branch: Pure causal convolution for short-range dependencies
    # Dilation factor = 1 for all layers (no dilation)
    local_branch = inputs
    for i in range(3):
        local_branch = residual_block(
            local_branch, 
            filters=filters, 
            kernel_size=3,
            dilation_rate=1,  # Pure causal convolution
            dropout_rate=0.3
        )
    
    # ==================== Transformer Encoders ====================
    
    # Global Transformer encoder (12 attention heads)
    encoder1_output = transformer_encoder(
        global_branch, 
        num_layers=num_encoder_layers, 
        d_model=filters,
        num_heads=12, 
        dff=filters * 4, 
        dropout_rate=0.1, 
        max_len=1000
    )
    
    # Local Transformer encoder (6 attention heads)
    encoder2_output = transformer_encoder(
        local_branch, 
        num_layers=num_encoder_layers, 
        d_model=filters,
        num_heads=6, 
        dff=filters * 4, 
        dropout_rate=0.1, 
        max_len=1000
    )
    
    # ==================== Feature Fusion ====================
    
    # Weighted fusion: lambda * global + (1-lambda) * local
    lambda_param = 0.5
    combined = Add()([
        Lambda(lambda x: x * lambda_param)(encoder1_output),
        Lambda(lambda x: x * (1 - lambda_param))(encoder2_output)
    ])
    
    # ==================== Classification Head ====================
    
    x = GlobalAveragePooling1D()(combined)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='GCDL')
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        metrics=['accuracy']
    )
    
    print("Model built successfully")
    
    return model


# ======================= Physics Constraint Components =======================
def build_forward_model(wavelet_length: int) -> Model:
    """
    Build the seismic forward modeling component.
    
    Convolves reflection coefficients with seismic wavelet to generate
    synthetic seismic records.
    
    Args:
        wavelet_length: Length of the seismic wavelet
    
    Returns:
        Forward model
    """
    rc_input = Input(shape=(None, 1), name='reflection_coefficients')
    synthetic = Conv1D(
        filters=1, 
        kernel_size=wavelet_length,
        padding='same', 
        use_bias=False, 
        name='wavelet_conv'
    )(rc_input)
    return Model(inputs=rc_input, outputs=synthetic, name='Forward_Model')


class PhysicsFineTuner:
    """
    Physics-constrained fine-tuning trainer.
    
    Implements the geophysical constraint by:
    1. Converting lithology probabilities to acoustic impedance
    2. Computing reflection coefficients
    3. Generating synthetic seismic via convolution with wavelet
    4. Computing loss based on fit between synthetic and observed data
    """
    
    def __init__(self, 
                 model: Model, 
                 wavelet: np.ndarray, 
                 Z_sandstone: float, 
                 Z_mudstone: float,
                 lambda_seismic: float = 1.0, 
                 alpha_envelope: float = 0.3):
        """
        Initialize the physics fine-tuner.
        
        Args:
            model: Pre-trained classification model
            wavelet: Normalized seismic wavelet
            Z_sandstone: Reference acoustic impedance for sandstone
            Z_mudstone: Reference acoustic impedance for mudstone
            lambda_seismic: Weight for seismic constraint loss
            alpha_envelope: Weight for envelope loss component
        """
        self.model = model
        self.wavelet = tf.constant(wavelet.reshape(-1, 1, 1), dtype=tf.float32)
        self.Z_sandstone = tf.constant(Z_sandstone, dtype=tf.float32)
        self.Z_mudstone = tf.constant(Z_mudstone, dtype=tf.float32)
        self.lambda_seismic = lambda_seismic
        self.alpha_envelope = alpha_envelope
        
        # Build forward model
        self.fwd_model = build_forward_model(len(wavelet))
        self.fwd_model.get_layer('wavelet_conv').set_weights([wavelet.reshape(-1, 1, 1)])
        self.fwd_model.trainable = False
        
        self.optimizer = Adam(learning_rate=Config.FINETUNE_LR)
        self.history = {'loss': [], 'cls_loss': [], 'seismic_loss': [], 'accuracy': []}
    
    def lithology_to_impedance(self, probs: tf.Tensor) -> tf.Tensor:
        """Convert lithology probabilities to acoustic impedance."""
        return probs[:, 0:1] * self.Z_sandstone + probs[:, 1:2] * self.Z_mudstone
    
    def compute_reflection_coefficients(self, impedance: tf.Tensor) -> tf.Tensor:
        """Compute reflection coefficients from impedance sequence."""
        Z_upper = impedance[:-1]
        Z_lower = impedance[1:]
        return (Z_lower - Z_upper) / (Z_lower + Z_upper + 1e-8)
    
    def train_step(self, 
                   X_batch: tf.Tensor, 
                   y_batch: tf.Tensor, 
                   trace_sample_indices: List, 
                   observed_seismic_traces: List) -> Tuple[tf.Tensor, ...]:
        """
        Single training step with physics constraint.
        
        Args:
            X_batch: Input batch
            y_batch: Label batch (one-hot encoded)
            trace_sample_indices: Indices mapping samples to traces
            observed_seismic_traces: Observed seismic data for each trace
        
        Returns:
            Tuple of (total_loss, cls_loss, seismic_loss, accuracy)
        """
        with tf.GradientTape() as tape:
            # Classification prediction
            probs = self.model(X_batch, training=True)
            
            # Classification loss
            cls_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_batch, probs)
            )
            
            # Physics constraint loss
            seismic_losses = []
            
            for i in range(len(trace_sample_indices)):
                indices = trace_sample_indices[i]
                trace_probs = tf.gather(probs, indices)
                
                # Probability -> Impedance -> Reflection coefficients
                impedance = self.lithology_to_impedance(trace_probs)
                rc = self.compute_reflection_coefficients(impedance)
                
                # Forward modeling: convolve with wavelet
                rc_expanded = tf.expand_dims(rc, 0)
                synthetic = self.fwd_model(rc_expanded, training=False)
                synthetic = tf.squeeze(synthetic, [0, -1])
                
                observed = observed_seismic_traces[i]
                
                # Align lengths
                min_len = tf.minimum(tf.shape(synthetic)[0], tf.shape(observed)[0])
                synthetic_aligned = synthetic[:min_len]
                observed_aligned = observed[:min_len]
                
                # Normalize for comparison
                syn_norm = synthetic_aligned / (tf.reduce_max(tf.abs(synthetic_aligned)) + 1e-8)
                obs_norm = observed_aligned / (tf.reduce_max(tf.abs(observed_aligned)) + 1e-8)
                
                # Waveform loss + envelope loss
                waveform_loss = tf.reduce_mean(tf.square(syn_norm - obs_norm))
                envelope_loss = tf.reduce_mean(tf.square(tf.abs(syn_norm) - tf.abs(obs_norm)))
                
                seismic_losses.append(waveform_loss + self.alpha_envelope * envelope_loss)
            
            seismic_loss = tf.reduce_mean(seismic_losses) if seismic_losses else tf.constant(0.0)
            
            # Total loss
            total_loss = cls_loss + self.lambda_seismic * seismic_loss
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute accuracy
        predictions = tf.argmax(probs, axis=1)
        y_true = tf.argmax(y_batch, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))
        
        return total_loss, cls_loss, seismic_loss, accuracy
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train_encoded: np.ndarray, 
            trace_groups: Dict, 
            sample_times: np.ndarray,
            seismic_3d: np.ndarray, 
            time_samples_seis: np.ndarray, 
            inlines_seis: np.ndarray, 
            xlines_seis: np.ndarray,
            epochs: int = 5, 
            traces_per_batch: int = 2) -> Dict:
        """
        Fine-tune the model with physics constraints.
        
        Args:
            X_train: Training features
            y_train_encoded: One-hot encoded labels
            trace_groups: Dictionary mapping trace IDs to sample indices
            sample_times: Time values for each sample
            seismic_3d: 3D seismic volume
            time_samples_seis: Time samples for seismic data
            inlines_seis: Inline coordinates
            xlines_seis: Xline coordinates
            epochs: Number of fine-tuning epochs
            traces_per_batch: Number of traces per batch
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Stage 2: Physics-Constrained Fine-tuning")
        print(f"Learning rate: {Config.FINETUNE_LR}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        trace_list = list(trace_groups.keys())
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_cls = []
            epoch_seis = []
            epoch_acc = []
            
            random.shuffle(trace_list)
            
            for batch_start in range(0, len(trace_list), traces_per_batch):
                batch_traces = trace_list[batch_start:batch_start + traces_per_batch]
                
                # Collect batch indices
                batch_indices = []
                trace_sample_indices = []
                current_offset = 0
                
                for trace_id in batch_traces:
                    indices = trace_groups[trace_id]
                    batch_indices.extend(indices)
                    trace_sample_indices.append(
                        list(range(current_offset, current_offset + len(indices)))
                    )
                    current_offset += len(indices)
                
                X_batch = X_train[batch_indices].astype(np.float32)
                y_batch = y_train_encoded[batch_indices].astype(np.float32)
                
                # Get observed seismic data
                observed_list = []
                for trace_id in batch_traces:
                    indices = trace_groups[trace_id]
                    times = sample_times[indices]
                    
                    inline, xline = map(int, trace_id.split('_'))
                    inline_idx = np.argmin(np.abs(inlines_seis - inline))
                    xline_idx = np.argmin(np.abs(xlines_seis - xline))
                    
                    trace_data = seismic_3d[:, xline_idx, inline_idx]
                    
                    time_min, time_max = times.min(), times.max()
                    time_mask = (time_samples_seis >= time_min) & (time_samples_seis <= time_max)
                    observed = trace_data[time_mask]
                    
                    observed_list.append(tf.constant(observed.astype(np.float32)))
                
                trace_indices_tf = [tf.constant(idx, dtype=tf.int32) for idx in trace_sample_indices]
                
                # Training step
                total_loss, cls_loss, seis_loss, acc = self.train_step(
                    tf.constant(X_batch), 
                    tf.constant(y_batch),
                    trace_indices_tf, 
                    observed_list
                )
                
                epoch_losses.append(float(total_loss))
                epoch_cls.append(float(cls_loss))
                epoch_seis.append(float(seis_loss))
                epoch_acc.append(float(acc))
            
            # Record epoch history
            self.history['loss'].append(np.mean(epoch_losses))
            self.history['cls_loss'].append(np.mean(epoch_cls))
            self.history['seismic_loss'].append(np.mean(epoch_seis))
            self.history['accuracy'].append(np.mean(epoch_acc))
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {np.mean(epoch_losses):.4f} - "
                  f"cls_loss: {np.mean(epoch_cls):.4f} - "
                  f"seismic_loss: {np.mean(epoch_seis):.4f} - "
                  f"accuracy: {np.mean(epoch_acc):.4f}")
        
        return self.history


# ======================= Evaluation Functions =======================
def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC calculation)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Binary classification specific metrics
    if len(cm) == 2:
        TN, FP, FN, TP = cm.ravel()
        metrics['TN'] = int(TN)
        metrics['FP'] = int(FP)
        metrics['FN'] = int(FN)
        metrics['TP'] = int(TP)
        
        metrics['specificity'] = TN / (TN + FP) if (TN + FP) > 0 else 0
        metrics['sensitivity'] = TP / (TP + FN) if (TP + FN) > 0 else 0
        metrics['npv'] = TN / (TN + FN) if (TN + FN) > 0 else 0
        metrics['ppv'] = TP / (TP + FP) if (TP + FP) > 0 else 0
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in Config.CLASSES.items():
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
            metrics[f'support_{class_name}'] = int(np.sum(y_true == i))
    
    # AUC metrics
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
            elif y_prob.ndim == 1:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            metrics['auc'] = None
            metrics['average_precision'] = None
    
    return metrics


def print_evaluation_report(metrics: Dict, stage_name: str = "") -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*70}")
    print(f"{stage_name} Evaluation Report")
    print(f"{'='*70}")
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision (macro):  {metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall (macro):     {metrics.get('recall_macro', 0):.4f}")
    print(f"  F1-Score (macro):   {metrics.get('f1_macro', 0):.4f}")
    
    if 'specificity' in metrics:
        print(f"\nBinary Classification Metrics:")
        print(f"  Sensitivity:        {metrics.get('sensitivity', 0):.4f}")
        print(f"  Specificity:        {metrics.get('specificity', 0):.4f}")
        print(f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  MCC:                {metrics.get('mcc', 0):.4f}")
        print(f"  Cohen's Kappa:      {metrics.get('kappa', 0):.4f}")
    
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"  AUC:                {metrics.get('auc', 0):.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print(f"  {'-'*52}")
    for i, class_name in Config.CLASSES.items():
        prec = metrics.get(f'precision_{class_name}', 0)
        rec = metrics.get(f'recall_{class_name}', 0)
        f1 = metrics.get(f'f1_{class_name}', 0)
        sup = metrics.get(f'support_{class_name}', 0)
        print(f"  {class_name:<12} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {sup:<10}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  {'':>12} {'Pred Sand':>10} {'Pred Mud':>10}")
        print(f"  {'True Sand':<12} {cm[0][0]:>10} {cm[0][1]:>10}")
        print(f"  {'True Mud':<12} {cm[1][0]:>10} {cm[1][1]:>10}")


# ======================= Visualization Functions =======================
def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         classes_dict: Dict, 
                         save_path: str, 
                         title: str = "Confusion Matrix") -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [classes_dict[i] for i in sorted(classes_dict.keys())]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 16})
    plt.ylabel('Real label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray, 
                  save_path: str, 
                  title: str = "ROC Curve") -> float:
    """Plot and save ROC curve."""
    if y_prob.ndim == 2:
        y_score = y_prob[:, 1]
    else:
        y_score = y_prob
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved: {save_path}")
    return roc_auc


def plot_training_history(history, save_path: str, title: str = "Training History") -> None:
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], 'b-', label='Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], 'g-', label='Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved: {save_path}")


# ======================= Data Loading Functions =======================
def get_normalization_suffix() -> str:
    """Get file suffix based on normalization mode."""
    suffix_map = {
        "none": "_raw",
        "trace": "_trace_norm",
        "rms_global": "_rms_global",
        "max_global": "_max_global"
    }
    return suffix_map.get(Config.NORMALIZE_MODE, "_raw")


def load_training_samples() -> Tuple:
    """
    Load training samples from preprocessed files.
    
    Returns:
        Tuple of (X, y_binary, trace_ids, sample_times, trace_groups, y_original)
    """
    print(f"\n{'='*60}")
    print(f"Loading training samples")
    print(f"{'='*60}")
    
    suffix = get_normalization_suffix()
    training_path = os.path.join(Config.PROCESSED_PATH, "training_samples")
    sample_file = os.path.join(training_path, f"training_samples{suffix}.txt")
    
    if not os.path.exists(sample_file):
        print(f"File not found: {sample_file}")
        return None, None, None, None, None, None
    
    print(f"Reading: {sample_file}")
    
    # Read samples
    all_samples = []
    with open(sample_file, 'r') as f:
        current_sample = []
        for line in f:
            line = line.strip()
            if line:
                current_sample.append([float(x) for x in line.split()])
            else:
                if current_sample:
                    all_samples.append(np.array(current_sample))
                    current_sample = []
        if current_sample:
            all_samples.append(np.array(current_sample))
    
    print(f"Loaded {len(all_samples)} samples")
    
    # Extract data
    X_list = []
    y_list = []
    trace_ids = []
    sample_times = []
    trace_groups = defaultdict(list)
    
    for sample in all_samples:
        center_idx = len(sample) // 2
        label = sample[center_idx, -1]
        center_time = sample[center_idx, 0]
        inline = sample[center_idx, 2]
        xline = sample[center_idx, 3]
        
        if not np.isnan(label):
            X_list.append(sample[:, 1].reshape(-1, 1))
            y_list.append(int(label))
            sample_times.append(center_time)
            
            trace_id = f"{int(inline)}_{int(xline)}"
            trace_ids.append(trace_id)
            trace_groups[trace_id].append(len(X_list) - 1)
    
    X = np.array(X_list)
    y_original = np.array(y_list)
    sample_times = np.array(sample_times)
    
    # Map to binary labels
    y_binary = map_labels_to_binary(y_original)
    
    print(f"Data loaded successfully")
    print(f"  X shape: {X.shape}")
    print(f"  Number of traces: {len(trace_groups)}")
    
    # Print class distribution
    print(f"\nOriginal label distribution:")
    for label in [0, 1, 2]:
        count = np.sum(y_original == label)
        if count > 0:
            print(f"  Label {label}: {count} ({count/len(y_original)*100:.1f}%)")
    
    print(f"\nBinary label distribution:")
    for label in [0, 1]:
        count = np.sum(y_binary == label)
        print(f"  {Config.CLASSES[label]}: {count} ({count/len(y_binary)*100:.1f}%)")
    
    return X, y_binary, trace_ids, sample_times, dict(trace_groups), y_original


def load_prediction_samples() -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Load prediction samples from preprocessed files.
    
    Returns:
        Tuple of (X_pred, info_df)
    """
    print(f"\n{'='*60}")
    print(f"Loading prediction samples")
    print(f"{'='*60}")
    
    suffix = get_normalization_suffix()
    prediction_path = os.path.join(Config.PROCESSED_PATH, "prediction_samples")
    sample_file = os.path.join(prediction_path, f"prediction_samples{suffix}.txt")
    info_file = os.path.join(prediction_path, f"prediction_info{suffix}.csv")
    
    if not os.path.exists(sample_file):
        print(f"File not found: {sample_file}")
        return None, None
    
    print(f"Reading: {sample_file}")
    
    all_samples = []
    with open(sample_file, 'r') as f:
        current_sample = []
        for line in f:
            line = line.strip()
            if line:
                current_sample.append([float(x) for x in line.split()])
            else:
                if current_sample:
                    all_samples.append(np.array(current_sample))
                    current_sample = []
        if current_sample:
            all_samples.append(np.array(current_sample))
    
    print(f"Loaded {len(all_samples)} prediction samples")
    
    X_pred = np.array([s[:, 1].reshape(-1, 1) for s in all_samples])
    
    info_df = None
    if os.path.exists(info_file):
        info_df = pd.read_csv(info_file)
        print(f"Info file loaded: {info_df.shape}")
    
    print(f"Prediction samples shape: {X_pred.shape}")
    
    return X_pred, info_df


# ======================= Main Training Function =======================
def train_and_predict() -> None:
    """Main training and prediction pipeline."""
    print("\n" + "="*80)
    print("GCDL: Geophysically Constrained Deep Learning for Lithology Prediction")
    print("="*80)
    
    # Set random seeds
    set_all_seeds(Config.RANDOM_SEED)
    
    # Create output directory
    suffix = get_normalization_suffix()
    output_dir = os.path.join(Config.OUTPUT_PATH, f"gcdl_results{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Load Data ====================
    result = load_training_samples()
    if result[0] is None:
        print("Failed to load training samples")
        return
    
    X, y, trace_ids, sample_times, trace_groups, y_original = result
    
    # ==================== Train-Test Split ====================
    print(f"\n{'='*60}")
    print(f"Train-Test Split (stratified)")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(y_train)} samples")
    print(f"  Sandstone: {np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"  Mudstone: {np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"Test set: {len(y_test)} samples")
    print(f"  Sandstone: {np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"  Mudstone: {np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")
    
    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y_train.reshape(-1, 1))
    y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
    
    # ==================== Stage 1: Standard Training ====================
    print(f"\n{'='*60}")
    print(f"Stage 1: Standard Cross-Entropy Training")
    print(f"{'='*60}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_gcdl_model(input_shape, Config.N_CLASSES, Config.FILTERS, Config.NUM_ENCODER_LAYERS)
    
    print(f"\nModel Summary:")
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=Config.EARLY_STOPPING_PATIENCE, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
    ]
    
    print(f"\nStarting training...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train_encoded,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    stage1_time = time.time() - start_time
    print(f"\nStage 1 training completed in {stage1_time:.2f} seconds")
    
    # ==================== Stage 1 Evaluation ====================
    print(f"\n{'='*60}")
    print(f"Stage 1 Evaluation")
    print(f"{'='*60}")
    
    # Get predictions
    y_train_prob_s1 = model.predict(X_train, verbose=0)
    y_test_prob_s1 = model.predict(X_test, verbose=0)
    y_train_pred_s1 = np.argmax(y_train_prob_s1, axis=1)
    y_test_pred_s1 = np.argmax(y_test_prob_s1, axis=1)
    
    # Calculate metrics
    train_metrics_s1 = calculate_metrics(y_train, y_train_pred_s1, y_train_prob_s1)
    test_metrics_s1 = calculate_metrics(y_test, y_test_pred_s1, y_test_prob_s1)
    
    # Print reports
    print_evaluation_report(train_metrics_s1, "Stage 1 - Training Set")
    print_evaluation_report(test_metrics_s1, "Stage 1 - Test Set")
    
    # Classification report
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}")
    print("\nTraining Set:")
    print(classification_report(y_train, y_train_pred_s1, target_names=['Sandstone', 'Mudstone']))
    print("\nTest Set:")
    print(classification_report(y_test, y_test_pred_s1, target_names=['Sandstone', 'Mudstone']))
    
    # Save plots
    plot_training_history(history, os.path.join(output_dir, "training_history_stage1.png"))
    plot_confusion_matrix(y_train, y_train_pred_s1, Config.CLASSES,
                         os.path.join(output_dir, "confusion_matrix_train_stage1.png"),
                         "Stage 1 - Training Set")
    plot_confusion_matrix(y_test, y_test_pred_s1, Config.CLASSES,
                         os.path.join(output_dir, "confusion_matrix_test_stage1.png"),
                         "Stage 1 - Test Set")
    plot_roc_curve(y_test, y_test_prob_s1,
                  os.path.join(output_dir, "roc_curve_stage1.png"),
                  "Stage 1 - ROC Curve")
    
    # Save Stage 1 model
    model.save_weights(os.path.join(output_dir, "model_stage1.weights.h5"))
    
    # Initialize Stage 2 variables
    stage2_time = 0
    test_metrics_s2 = None
    
    # ==================== Stage 2: Physics Fine-tuning (Optional) ====================
    if Config.ENABLE_PHYSICS_FINETUNE and SEGYIO_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"Stage 2: Physics-Constrained Fine-tuning")
        print(f"{'='*60}")
        
        # Load physics data (wavelet, impedance volume, seismic volume)
        # This section requires actual data files
        print("Note: Physics fine-tuning requires seismic and impedance data files.")
        print("Skipping Stage 2 due to missing data files.")
        # If you have the required data files, implement the loading and fine-tuning here
    
    # ==================== Save Results ====================
    print(f"\n{'='*60}")
    print(f"Saving Results")
    print(f"{'='*60}")
    
    model_save_dir = os.path.join(output_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save model weights
    model.save_weights(os.path.join(model_save_dir, "final_weights.h5"))
    
    # Build metrics dictionary
    all_metrics = {
        'method': 'GCDL',
        'normalize_mode': Config.NORMALIZE_MODE,
        'enable_physics_finetune': Config.ENABLE_PHYSICS_FINETUNE,
        'stage1_epochs': Config.EPOCHS,
        'stage1_time': stage1_time,
    }
    
    # Add Stage 1 metrics
    for key, value in train_metrics_s1.items():
        all_metrics[f'stage1_train_{key}'] = value
    for key, value in test_metrics_s1.items():
        all_metrics[f'stage1_test_{key}'] = value
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        # Convert numpy types to Python types
        metrics_clean = {}
        for k, v in all_metrics.items():
            if isinstance(v, np.ndarray):
                metrics_clean[k] = v.tolist()
            elif isinstance(v, (np.int32, np.int64)):
                metrics_clean[k] = int(v)
            elif isinstance(v, (np.float32, np.float64)):
                metrics_clean[k] = float(v)
            else:
                metrics_clean[k] = v
        json.dump(metrics_clean, f, indent=4)
    
    print(f"Metrics saved: {os.path.join(output_dir, 'metrics.json')}")
    
    # Save model config
    config = {
        'input_shape': list(input_shape),
        'n_classes': Config.N_CLASSES,
        'filters': Config.FILTERS,
        'num_encoder_layers': Config.NUM_ENCODER_LAYERS,
        'normalize_mode': Config.NORMALIZE_MODE,
    }
    
    with open(os.path.join(model_save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save encoder
    with open(os.path.join(model_save_dir, "encoder.pkl"), 'wb') as f:
        pickle.dump(encoder, f)
    
    print(f"Model saved to: {model_save_dir}")
    
    # ==================== Prediction ====================
    X_pred, info_df = load_prediction_samples()
    
    if X_pred is not None:
        print(f"\n{'='*60}")
        print(f"Generating Predictions")
        print(f"{'='*60}")
        
        print(f"Prediction samples: {X_pred.shape[0]}")
        
        batch_size_pred = 1000
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(X_pred), batch_size_pred):
            batch = X_pred[i:i + batch_size_pred]
            proba = model.predict(batch, verbose=0)
            pred = np.argmax(proba, axis=1)
            
            all_predictions.extend(pred)
            all_probabilities.extend(proba)
            
            if (i // batch_size_pred + 1) % 50 == 0:
                print(f"  Processed {i + batch_size_pred}/{len(X_pred)} samples")
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        print(f"Prediction completed")
        
        # Print prediction distribution
        print(f"\nPrediction distribution:")
        for label in [0, 1]:
            count = np.sum(all_predictions == label)
            percentage = count / len(all_predictions) * 100
            print(f"  {Config.CLASSES[label]}: {count} ({percentage:.1f}%)")
        
        # Save predictions
        if info_df is not None:
            info_df['predicted_label'] = all_predictions
            info_df['predicted_lithology'] = [Config.CLASSES[int(x)] for x in all_predictions]
            info_df['prob_Sandstone'] = all_probabilities[:, 0]
            info_df['prob_Mudstone'] = all_probabilities[:, 1]
            
            pred_file = os.path.join(output_dir, "predictions.csv")
            info_df.to_csv(pred_file, index=False)
            print(f"Predictions saved: {pred_file}")
        
        np.save(os.path.join(output_dir, "probabilities.npy"), all_probabilities)
        np.save(os.path.join(output_dir, "predictions.npy"), all_predictions)
        print(f"Probability arrays saved")
    
    # ==================== Summary ====================
    print(f"\n{'='*80}")
    print("Processing Completed!")
    print(f"{'='*80}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFinal Results:")
    print(f"  Stage 1 Test Set:")
    print(f"    Accuracy:  {test_metrics_s1['accuracy']:.4f}")
    print(f"    Precision: {test_metrics_s1['precision_macro']:.4f}")
    print(f"    Recall:    {test_metrics_s1['recall_macro']:.4f}")
    print(f"    F1-Score:  {test_metrics_s1['f1_macro']:.4f}")
    if test_metrics_s1.get('auc'):
        print(f"    AUC:       {test_metrics_s1['auc']:.4f}")


# ======================= Entry Point =======================
def main():
    """Main entry point."""
    print("="*80)
    print("GCDL: Geophysically Constrained Deep Learning for Lithology Prediction")
    print("="*80)
    print(f"\nModel Architecture:")
    print(f"  - Dual-branch TCN:")
    print(f"    - Local branch: Causal convolution (dilation=1)")
    print(f"    - Global branch: Dilated convolution (dilation=2,4,8)")
    print(f"  - Transformer encoder for global dependencies")
    print(f"  - Physics-constrained loss function (optional)")
    print(f"\nConfiguration:")
    print(f"  Normalization: {Config.NORMALIZE_MODE}")
    print(f"  Sample length: {Config.SAMPLE_LENGTH}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Physics fine-tuning: {'Enabled' if Config.ENABLE_PHYSICS_FINETUNE else 'Disabled'}")
    print("="*80)
    
    try:
        train_and_predict()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
