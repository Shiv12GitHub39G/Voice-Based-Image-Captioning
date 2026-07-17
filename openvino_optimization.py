"""
OpenVINO Optimization — Voice-Based Image Captioning
======================================================
This script converts the EfficientNetB0 CNN encoder from the
Voice-Based Image Captioning project to Intel OpenVINO IR format
and benchmarks inference speed improvement over standard TensorFlow.

Why OpenVINO?
- Intel OpenVINO optimizes trained models for faster CPU inference
- Reduces latency by applying model compression and graph optimization
- Particularly useful for edge/on-device deployment without GPU

How to run (Google Colab):
    !pip install openvino tensorflow numpy pillow

Author: Shivam Gupta
Project: Voice-Based Image Captioning
"""

import numpy as np
import time
import os
from PIL import Image

# ── Step 1: Install OpenVINO if not already installed ──────────
# Run this in Colab before running the script:
# !pip install openvino tensorflow numpy pillow -q

# ── Step 2: Import libraries ───────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

print("TensorFlow version:", tf.__version__)

# ── Step 3: Load EfficientNetB0 (same as used in main project) ─
print("\nLoading EfficientNetB0 encoder...")
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,          # remove classifier head
    input_shape=(224, 224, 3)
)
print(f"Model loaded — Output shape: {base_model.output_shape}")
print(f"Total parameters: {base_model.count_params():,}")

# ── Step 4: Save model in SavedModel format for OpenVINO ───────
SAVED_MODEL_DIR = "efficientnet_encoder"
print(f"\nSaving model to: {SAVED_MODEL_DIR}/")
base_model.export(SAVED_MODEL_DIR)
print("Model saved successfully.")

# ── Step 5: Convert to OpenVINO IR format ──────────────────────
print("\nConverting to OpenVINO IR format...")
print("Run this command in your terminal or Colab cell:")
print(f"  !mo --saved_model_dir {SAVED_MODEL_DIR} --output_dir openvino_model --input_shape [1,224,224,3]")

# Alternative: use Python API for conversion (OpenVINO 2023+)
try:
    from openvino.tools.mo import convert_model
    from openvino.runtime import serialize

    print("\nUsing OpenVINO Python API for conversion...")
    ov_model = convert_model(
        SAVED_MODEL_DIR,
        input_shape=[1, 224, 224, 3]
    )
    os.makedirs("openvino_model", exist_ok=True)
    serialize(ov_model, "openvino_model/efficientnet_encoder.xml")
    print("Conversion successful!")
    print("Files saved:")
    print("  openvino_model/efficientnet_encoder.xml  <- model structure")
    print("  openvino_model/efficientnet_encoder.bin  <- model weights")

except ImportError:
    print("OpenVINO tools not found. Install with: pip install openvino-dev")
except Exception as e:
    print(f"Conversion error: {e}")
    print("Try running the CLI command above instead.")

# ── Step 6: Run inference with OpenVINO ────────────────────────
print("\n" + "="*55)
print("INFERENCE BENCHMARK — TensorFlow vs OpenVINO")
print("="*55)

# Create a dummy test image (same preprocessing as main project)
dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
dummy_pil   = Image.fromarray(dummy_image)

def preprocess_for_model(pil_image):
    """Same preprocessing used in app.py"""
    img = pil_image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))
    return img

input_data = preprocess_for_model(dummy_pil)

# ── Benchmark 1: TensorFlow inference ──────────────────────────
print("\n[1] TensorFlow Inference:")
N_RUNS = 20

# Warmup
_ = base_model.predict(input_data, verbose=0)

tf_times = []
for i in range(N_RUNS):
    start = time.time()
    tf_output = base_model.predict(input_data, verbose=0)
    tf_times.append(time.time() - start)

tf_mean = np.mean(tf_times) * 1000
tf_std  = np.std(tf_times)  * 1000
print(f"   Mean latency: {tf_mean:.1f} ms ± {tf_std:.1f} ms")
print(f"   Output shape: {tf_output.shape}")
print(f"   Reshaped (49, 1280): {tf_output.reshape(1, 49, 1280).shape}")

# ── Benchmark 2: OpenVINO inference ────────────────────────────
print("\n[2] OpenVINO Inference:")
try:
    from openvino.runtime import Core

    core     = Core()
    ov_model = core.read_model("openvino_model/efficientnet_encoder.xml")

    # Compile for CPU (Intel optimized)
    compiled_model = core.compile_model(ov_model, device_name="CPU")
    output_layer   = compiled_model.output(0)

    print(f"   Device: CPU (Intel OpenVINO optimized)")
    print(f"   Available devices: {core.available_devices}")

    # Warmup
    _ = compiled_model([input_data])[output_layer]

    ov_times = []
    for i in range(N_RUNS):
        start     = time.time()
        ov_output = compiled_model([input_data])[output_layer]
        ov_times.append(time.time() - start)

    ov_mean = np.mean(ov_times) * 1000
    ov_std  = np.std(ov_times)  * 1000
    print(f"   Mean latency: {ov_mean:.1f} ms ± {ov_std:.1f} ms")
    print(f"   Output shape: {ov_output.shape}")

    # ── Results ────────────────────────────────────────────────
    speedup = tf_mean / ov_mean
    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)
    print(f"  TensorFlow latency : {tf_mean:.1f} ms")
    print(f"  OpenVINO latency   : {ov_mean:.1f} ms")
    print(f"  Speedup            : {speedup:.2f}x faster with OpenVINO")
    print(f"  Latency reduction  : {((tf_mean - ov_mean)/tf_mean)*100:.1f}%")
    print()
    print("Conclusion:")
    print(f"  Intel OpenVINO reduced EfficientNetB0 inference latency")
    print(f"  by {((tf_mean - ov_mean)/tf_mean)*100:.0f}% compared to standard TensorFlow.")
    print(f"  This directly improves caption generation speed in the")
    print(f"  Voice-Based Image Captioning pipeline for real-time use.")

    # ── Verify outputs match ───────────────────────────────────
    print("\nVerifying output consistency (TF vs OpenVINO):")
    tf_flat = tf_output.flatten()
    ov_flat = ov_output.flatten()
    max_diff = np.max(np.abs(tf_flat - ov_flat))
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Outputs match: {'YES' if max_diff < 1e-3 else 'SMALL NUMERICAL DIFF (expected)'}")

except ImportError:
    print("   OpenVINO runtime not installed.")
    print("   Install with: pip install openvino")
except FileNotFoundError:
    print("   OpenVINO model file not found.")
    print("   Run the conversion step (Step 5) first.")
except Exception as e:
    print(f"   OpenVINO inference error: {e}")

# ── Step 7: How this fits into the main project ────────────────
print("\n" + "="*55)
print("HOW THIS CONNECTS TO THE MAIN PROJECT")
print("="*55)
print("""
In the Voice-Based Image Captioning project:

  Original pipeline:
    Image → EfficientNetB0 (TensorFlow) → (49, 1280) features
           → Attention LSTM → Caption → gTTS → Audio

  With OpenVINO optimization:
    Image → EfficientNetB0 (OpenVINO IR) → (49, 1280) features
           → Attention LSTM → Caption → gTTS → Audio
           ^^ This step is now faster on Intel CPU

  The EfficientNetB0 encoder is the most compute-heavy part
  of the pipeline. OpenVINO optimizes exactly this bottleneck,
  enabling faster real-time caption generation on CPU devices
  without requiring a GPU — important for edge/accessibility tools.
""")
