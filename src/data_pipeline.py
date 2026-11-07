import pandas as pd
import tensorflow as tf

# Configuration
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    # Read image file
    image = tf.io.read_file(path)

    # Decode JPEG, PNG and other supported image formats
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Set image shape
    image.set_shape([None, None, 3])

    # Resize image
    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values from 0-255 to 0-1
    image = image / 255.0

    return image, label


def create_dataset(split, augment=False):
    # Read split information
    df = pd.read_csv("data/processed/splits.csv")
    df = df[df["split"] == split]

    paths = df["file"].values
    labels = df["label"].values

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle training data
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=42
        )

    # Load and preprocess images
    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    # Apply augmentation only to training data
    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05)
        ])

        dataset = dataset.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels
            ),
            num_parallel_calls=AUTOTUNE
        )

    # Batch and prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


if __name__ == "__main__":
    train_ds = create_dataset("train", augment=True)
    val_ds = create_dataset("validation")
    test_ds = create_dataset("test")

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel minimum:", tf.reduce_min(images).numpy())
        print("Pixel maximum:", tf.reduce_max(images).numpy())

    print("\nTraining batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())