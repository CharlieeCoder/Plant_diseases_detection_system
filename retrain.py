import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

def create_model(num_classes):
    """Create a CNN model with batch normalization and regularization"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First block
        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # Second block
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # Third block
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_data_generators(train_dir, val_dir=None):
    """Create train and validation data generators with augmentation
    
    Args:
        train_dir: Directory containing training data
        val_dir: Optional directory containing validation data. If None, 
                validation split will be used on training data.
    """
    # Training data generator with augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 if val_dir is None else None
    )
    
    # Validation data generator (only rescaling)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training' if val_dir is None else None,
        shuffle=True
    )
    
    # Create validation generator
    if val_dir is not None:
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    else:
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    
    return train_generator, validation_generator

def main():
    # Get data directories
    train_dir = 'train'
    val_dir = 'val' if os.path.exists('val') else None
    
    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators(train_dir, val_dir)
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"\nFound {num_classes} classes:")
    for class_name, idx in train_generator.class_indices.items():
        print(f"{idx}: {class_name}")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model(num_classes)
    
    # Compile model with additional metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks with more monitoring
    callbacks = [
        # Early stopping when accuracy is very high
        tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            min_delta=0.001,    # Stop if improvement is less than 0.1%
            patience=2,         # Only wait 2 epochs for improvement
            baseline=0.90,      # Stop if accuracy reaches 90%
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model_checkpoint.keras',
            monitor='accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history with all metrics
    metrics = ['accuracy', 'loss', 'auc']
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

    # Save training history to JSON
    with open('training_hist.json', 'w') as f:
        json.dump(history.history, f)
    
    # Save the model
    print("\nSaving model...")
    model.save('trained_plant_disease_model.keras')
    
    # Save training history and class indices with more information
    with open('model_info.json', 'w') as f:
        json.dump({
            'class_indices': train_generator.class_indices,
            'training_params': {
                'img_size': IMG_SIZE,
                'batch_size': BATCH_SIZE,
                'initial_epochs': EPOCHS,
                'final_learning_rate': float(tf.keras.backend.get_value(model.optimizer.learning_rate))
            },
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }, f, indent=2)
    
    print("\nTraining complete!")
    print("Files saved:")
    print("- trained_plant_disease_model.keras (Main model)")
    print("- model_checkpoint.keras (Best model during training)")
    print("- model_info.json (Training history and class indices)")
    print("- training_history.png (Training plots)")
    print("- training_log.csv (Detailed training logs)")

if __name__ == "__main__":
    main() 