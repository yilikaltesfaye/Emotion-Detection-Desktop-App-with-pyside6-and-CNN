import tensorflow as tf

model = tf.keras.models.load_model("models_store/emotion_cnn.keras")  # change path if needed
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
