import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import r2_score
import os

# -------------------------
# 1. File Discovery & Explicit ID Splitting
# -------------------------
file_pattern = "gs://64-universes/tfrecords_out/*.tfrecord"
all_files = tf.io.gfile.glob(file_pattern)

if len(all_files) == 0:
    raise FileNotFoundError(f"No files found at {file_pattern}!")

train_files = []
val_files   = []
test_files  = []

for f in all_files:
    match = re.search(r'shard_(\d{2})_', f)
    if match:
        run_id = match.group(1)
        if run_id in ['02', '04']:
            test_files.append(f)
        elif run_id in ['06', '09']:
            val_files.append(f)
        else:
            train_files.append(f)
    else:
        train_files.append(f)

print(f"Files allocated -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

RECORDS_PER_SHARD = 147
n_train = len(train_files) * RECORDS_PER_SHARD
n_val   = len(val_files)   * RECORDS_PER_SHARD
n_test  = len(test_files)  * RECORDS_PER_SHARD
print(f"Total records -> Train: {n_train} | Val: {n_val} | Test: {n_test}")

# Inspect the first file to get label length dynamically
raw = next(iter(tf.data.TFRecordDataset(train_files[0])))
ex  = tf.train.Example()
ex.ParseFromString(raw.numpy())
label_length = len(ex.features.feature["labels"].float_list.value)
print(f"Label length detected: {label_length}")

feature_description = {
    "labels": tf.io.FixedLenFeature([label_length], tf.float32),
    "cube":   tf.io.FixedLenFeature([], tf.string),
}

# -------------------------
# 2. Base Parsing Function
# -------------------------
def parse_and_slice(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.io.decode_raw(parsed['cube'], tf.float32)
    x = tf.ensure_shape(x, [262144])  # 64 * 64 * 64
    idx = tf.constant([0,1,2,3,5,6,7,8,10,17], dtype=tf.int32)
    y = tf.gather(parsed['labels'], idx)
    return x, y

raw_train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE).map(parse_and_slice, num_parallel_calls=tf.data.AUTOTUNE)
raw_val_ds   = tf.data.TFRecordDataset(val_files,   num_parallel_reads=tf.data.AUTOTUNE).map(parse_and_slice, num_parallel_calls=tf.data.AUTOTUNE)
raw_test_ds  = tf.data.TFRecordDataset(test_files,  num_parallel_reads=tf.data.AUTOTUNE).map(parse_and_slice, num_parallel_calls=tf.data.AUTOTUNE)

# -------------------------
# 3. Compute Scaling Statistics (Train ONLY)
# -------------------------
print("Calculating Min-Max statistics for Y (Streaming)...")
y_ds    = raw_train_ds.map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE)
y_min   = y_ds.reduce(tf.fill([10], float('inf')),  tf.minimum)
y_max   = y_ds.reduce(tf.fill([10], float('-inf')), tf.maximum)
y_range = tf.maximum(y_max - y_min, 1e-8)

print("Calculating Mean-Variance statistics for X (Streaming subset)...")
x_norm_layer = layers.Normalization(axis=-1)
x_only_ds    = raw_train_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
x_norm_layer.adapt(x_only_ds.take(250))  # 125 * 64 = 8000 cubes

x_mean  = tf.constant(tf.squeeze(x_norm_layer.mean))
x_std   = tf.constant(tf.maximum(tf.sqrt(tf.squeeze(x_norm_layer.variance)), 1e-8))
y_min   = tf.constant(y_min)
y_range = tf.constant(y_range)


# -------------------------
# 4. Create the Scaled, Ready-to-Train Datasets
# -------------------------
def scale_and_reshape(x, y):
    x_scaled   = (x - x_mean) / x_std
    x_reshaped = tf.reshape(x_scaled, [64, 64, 64, 1])
    y_scaled   = (y - y_min) / y_range
    return x_reshaped, y_scaled

BATCH_SIZE       = 128
steps_per_epoch  = n_train // BATCH_SIZE
validation_steps = n_val   // BATCH_SIZE

train_ds = raw_train_ds.map(scale_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(8192).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

val_ds = raw_val_ds.map(scale_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

test_ds = raw_test_ds.map(scale_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------
# 5. Model
# -------------------------
inputs = layers.Input(shape=(64, 64, 64, 1))

x = layers.Conv3D(16,  (3,3,3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D((2,2,2))(x)   # 32x32x32

x = layers.Conv3D(32,  (3,3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D((2,2,2))(x)   # 16x16x16

x = layers.Conv3D(64,  (3,3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D((2,2,2))(x)   # 8x8x8

x = layers.Conv3D(128, (3,3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D((2,2,2))(x)   # 4x4x4

x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(10)(x)
model   = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# -------------------------
# 6. Callbacks & Compilation
# -------------------------
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_loss', verbose=1, save_best_only=True
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.002, weight_decay=0.002, beta_1=0.9),
    loss='mse',
    metrics=['mae']
)

# -------------------------
# 7. Training
# -------------------------
history = model.fit(
    train_ds,
    epochs=250,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[reduce_lr_cb, checkpoint, early_stopping]
)

# -------------------------
# 8. Evaluation
# -------------------------
model.load_weights('best_model.keras')

test_loss, test_mae = model.evaluate(test_ds, verbose=0)
print(f"Final Test Loss: {test_loss:.6f}")
print(f"Final Test MAE:  {test_mae:.6f}")

# -------------------------
# 9. Export Results to NPZ
# -------------------------
def predict_dataset(ds, max_steps=None):
    preds_list, true_list = [], []
    if max_steps is not None:
        ds = ds.take(max_steps)
    for x_batch, y_batch in ds:
        preds_list.append(model.predict_on_batch(x_batch))
        true_list.append(y_batch.numpy())
    return np.concatenate(preds_list), np.concatenate(true_list)

print("Running inference on train / val / test for export...")
train_pred_s, train_true_s = predict_dataset(train_ds, max_steps=steps_per_epoch)
val_pred_s,   val_true_s   = predict_dataset(val_ds,   max_steps=validation_steps)
test_pred_s,  test_true_s  = predict_dataset(test_ds)

def inverse_scale(y):
    return (y * y_range.numpy()) + y_min.numpy()

artifact_dir = "run_artifacts"
os.makedirs(artifact_dir, exist_ok=True)
output_path  = os.path.join(artifact_dir, "results64.npz")

np.savez(
    output_path,
    train_pred=inverse_scale(train_pred_s),
    train_true=inverse_scale(train_true_s),
    val_pred=inverse_scale(val_pred_s),
    val_true=inverse_scale(val_true_s),
    test_pred=inverse_scale(test_pred_s),
    test_true=inverse_scale(test_true_s),
    train_pred_scaled=train_pred_s,
    val_pred_scaled=val_pred_s,
    test_pred_scaled=test_pred_s,
    loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"]),
    mae=np.array(history.history["mae"]),
    val_mae=np.array(history.history["val_mae"]),
    y_min=y_min.numpy(),
    y_max=y_max.numpy(),
    y_range=y_range.numpy(),
    x_mean=x_mean.numpy(),
    x_std=x_std.numpy(),
)

print(f"Saved full experiment -> {output_path}")
