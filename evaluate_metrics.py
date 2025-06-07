import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from PIL import Image
import pandas as pd

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

def evaluate_model(model, data_dir, class_mapping):
    """Evaluate model on a directory of images"""
    # Get reverse mapping
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Initialize lists to store predictions and true labels
    all_images = []
    true_labels = []
    image_paths = []
    
    # Process each class directory
    print("\nProcessing images...")
    for class_name in class_mapping.keys():
        # Convert model class name to directory name
        dir_name = class_name.replace(" - ", "___")
        if "Two-spotted_spider_mite" in dir_name:
            dir_name = "Tomato___Spider_mites Two-spotted_spider_mite"
            
        class_dir = os.path.join(data_dir, dir_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found for class {class_name}: {class_dir}")
            continue
            
        # Process all images in the class directory
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_array = load_and_preprocess_image(img_path)
                    all_images.append(img_array)
                    true_labels.append(class_mapping[class_name])
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    if not all_images:
        raise ValueError("No valid images found in the specified directories!")
    
    # Convert to numpy arrays
    X = np.array(all_images)
    y_true = np.array(true_labels)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    class_names = [idx_to_class[i] for i in range(len(class_mapping))]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'Image': image_paths,
        'True_Label': [idx_to_class[label] for label in y_true],
        'Predicted_Label': [idx_to_class[label] for label in y_pred],
        'Confidence': np.max(y_pred_probs, axis=1)
    })
    
    return report, cm, results_df

def main():
    # Load class mapping from model_info.json
    print("Loading model information...")
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
        class_mapping = model_info['class_indices']
    
    print(f"Number of classes: {len(class_mapping)}")
    print("Classes:", list(class_mapping.keys()))
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model('model_checkpoint.keras')
    
    # Evaluate on validation set
    val_dir = 'valid' if os.path.exists('valid') else 'test'
    print(f"\nEvaluating model using {val_dir} directory...")
    
    report, cm, results_df = evaluate_model(model, val_dir, class_mapping)
    
    # Print and save results
    print("\nClassification Report:")
    print("---------------------")
    print(report)
    
    # Save detailed results
    print("\nSaving results...")
    with open('evaluation_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("--------------------\n")
        f.write(report)
        f.write("\n\nConfusion Matrix\n")
        f.write("---------------\n")
        f.write("Classes:\n")
        for idx, name in class_mapping.items():
            f.write(f"{name}: {idx}\n")
        f.write("\nMatrix:\n")
        f.write(str(cm))
    
    # Save detailed results to CSV
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'")
    
    # Print some statistics
    print("\nPrediction Statistics:")
    print("---------------------")
    print(f"Total images evaluated: {len(results_df)}")
    print(f"Correct predictions: {sum(results_df['True_Label'] == results_df['Predicted_Label'])}")
    print(f"Average confidence: {results_df['Confidence'].mean():.4f}")

if __name__ == "__main__":
    main() 