import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile, os

print("=" * 50)
print("1. GPU CHECK")
print("=" * 50)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {len(gpus)}")
assert len(gpus) > 0, "No GPU detected!"

with tf.device('/GPU:0'):
    a = tf.random.normal([512, 512])
    b = tf.matmul(a, tf.transpose(a))
    print(f"GPU matmul OK — result shape: {b.shape}")

# -------------------------
print("\n" + "=" * 50)
print("2. SYNTHETIC TFRECORD READ/WRITE")
print("=" * 50)

# Write a few fake records mimicking your schema
tmp_dir = tempfile.mkdtemp()
tfr_path = os.path.join(tmp_dir, "test.tfrecord")

N_RECORDS = 10
LABEL_LEN = 18  # your actual label count
CUBE_SIZE  = 64 * 64 * 64

with tf.io.TFRecordWriter(tfr_path) as writer:
    for _ in range(N_RECORDS):
        cube   = np.random.rand(CUBE_SIZE).astype(np.float32)
        labels = np.random.rand(LABEL_LEN).astype(np.float32)
        feature = {
            "cube":   tf.train.Feature(bytes_list=tf.train.BytesList(value=[cube.tobytes()])),
            "labels": tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
        }
        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

print(f"Wrote {N_RECORDS} fake records to {tfr_path}")

# Read them back
feature_description = {
    "labels": tf.io.FixedLenFeature([LABEL_LEN], tf.float32),
    "cube":   tf.io.FixedLenFeature([], tf.string),
}

def parse_and_slice(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.io.decode_raw(parsed['cube'], tf.float32)
    x = tf.ensure_shape(x, [CUBE_SIZE])
    idx = tf.constant([0,1,2,3,5,6,7,8,10,17], dtype=tf.int32)
    y = tf.gather(parsed['labels'], idx)
    return x, y

ds = tf.data.TFRecordDataset(tfr_path).map(parse_and_slice)
x0, y0 = next(iter(ds))
print(f"Parsed record — x shape: {x0.shape}, y shape: {y0.shape}")
assert x0.shape == (CUBE_SIZE,) and y0.shape == (10,)

# -------------------------
print("\n" + "=" * 50)
print("3. NORMALIZATION LAYER")
print("=" * 50)

x_norm_layer = layers.Normalization(axis=-1)
x_only = ds.map(lambda x, y: x).batch(4).take(3)
x_norm_layer.adapt(x_only)
x_mean = tf.constant(tf.squeeze(x_norm_layer.mean), dtype=tf.float32)
x_std  = tf.constant(tf.maximum(tf.sqrt(tf.squeeze(x_norm_layer.variance)), 1e-8), dtype=tf.float32)
print(f"x_mean shape: {x_mean.shape}, mean of means: {x_mean.numpy().mean():.4f}")
print(f"x_std  shape: {x_std.shape},  mean of stds:  {x_std.numpy().mean():.4f}")

# -------------------------
print("\n" + "=" * 50)
print("4. PIPELINE + MODEL FORWARD PASS ON GPU")
print("=" * 50)

y_vals = list(ds.map(lambda x, y: y))
y_stack = tf.stack(y_vals)
y_min   = tf.reduce_min(y_stack, axis=0)
y_max   = tf.reduce_max(y_stack, axis=0)
y_range = tf.maximum(y_max - y_min, 1e-8)

def scale_and_reshape(x, y):
    x_scaled   = (x - x_mean) / x_std
    x_reshaped = tf.reshape(x_scaled, [64, 64, 64, 1])
    y_scaled   = (y - y_min) / y_range
    return x_reshaped, y_scaled

batch_ds = ds.map(scale_and_reshape).batch(4)
x_batch, y_batch = next(iter(batch_ds))
print(f"Batch shapes — x: {x_batch.shape}, y: {y_batch.shape}")

# Build your exact model architecture
inputs = layers.Input(shape=(64, 64, 64, 1))
x = layers.Conv3D(16, 3, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(2)(x)
x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(2)(x)
x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(2)(x)
x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(2)(x)
x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

with tf.device('/GPU:0'):
    preds = model(x_batch, training=False)
print(f"Forward pass OK — output shape: {preds.shape}")
model.summary()

print("\n✅ ALL CHECKS PASSED — ready for full training run")
print("\n" + "=" * 50)
print("5. MODEL COMPILE + TRAINING STEPS")
print("=" * 50)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.002, weight_decay=0.002, beta_1=0.9),
    loss='mse',
    metrics=['mae']
)

# Generate enough synthetic data for a few steps
N_SYNTH = 32
synth_ds = (
    tf.data.Dataset.from_tensors((
        tf.random.normal([N_SYNTH, 64, 64, 64, 1]),
        tf.random.uniform([N_SYNTH, 10])
    ))
    .unbatch()
    .batch(8)
    .repeat()
)

print("Running 5 training steps on synthetic batches...")
history = model.fit(synth_ds, steps_per_epoch=5, epochs=2, verbose=1)

print(f"\nFinal loss:  {history.history['loss'][-1]:.6f}")
print(f"Final MAE:   {history.history['mae'][-1]:.6f}")

# Checkpoint save/load round-trip
print("\nTesting checkpoint save/load...")
model.save('test_checkpoint.keras')
loaded = keras.models.load_model('test_checkpoint.keras')
with tf.device('/GPU:0'):
    dummy = tf.random.normal([2, 64, 64, 64, 1])
    out_original = model(dummy, training=False)
    out_loaded   = loaded(dummy, training=False)
    max_diff = tf.reduce_max(tf.abs(out_original - out_loaded)).numpy()
print(f"Max weight diff after reload: {max_diff:.2e}  (should be ~0)")
assert max_diff < 1e-5, "Checkpoint mismatch!"

print("\n✅ ALL CHECKS PASSED — full training run is ready")
