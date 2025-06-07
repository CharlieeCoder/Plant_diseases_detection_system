import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image
import io
from waitress import serve
import logging
import sys
import matplotlib.pyplot as plt
import time
import json

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create debug directory if it doesn't exist
DEBUG_DIR = 'debug_images'
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# Disable unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='view')

# Updated class names to match exactly with training data folders
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

def save_debug_image(image_array, filename_prefix):
    """Save image array for debugging"""
    try:
        # Convert normalized array back to 0-255 range
        debug_image = (image_array * 255).astype(np.uint8)
        
        # Save as PNG
        debug_path = os.path.join(DEBUG_DIR, f"{filename_prefix}_{int(time.time())}.png")
        Image.fromarray(debug_image).save(debug_path)
        logger.info(f"Saved debug image to {debug_path}")
        
        # Also save the raw array values
        np.save(os.path.join(DEBUG_DIR, f"{filename_prefix}_{int(time.time())}.npy"), image_array)
    except Exception as e:
        logger.error(f"Error saving debug image: {str(e)}")

def load_class_indices():
    """Load class indices from training info"""
    try:
        with open('model_info.json', 'r') as f:
            info = json.load(f)
            return info['class_indices']
    except Exception as e:
        logger.error(f"Error loading class indices: {str(e)}")
        return None

def verify_model_with_test_images():
    """Test model with sample images from test directory"""
    try:
        test_dir = 'test'
        if not os.path.exists(test_dir):
            logger.warning("Test directory not found")
            return
        
        logger.info("\nRunning model verification with test images...")
        test_results = []
        
        # Test with each image in test directory
        for img_name in os.listdir(test_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(test_dir, img_name)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path)
                    processed_img = preprocess_image(img)
                    
                    # Make prediction
                    pred = model.predict(processed_img, verbose=0)
                    pred_class = np.argmax(pred[0])
                    confidence = float(np.max(pred[0]))
                    
                    # Store results
                    test_results.append({
                        'image': img_name,
                        'predicted_class': CLASS_NAMES[pred_class],
                        'confidence': confidence,
                        'prediction_distribution': pred[0].tolist()
                    })
                    
                    logger.info(f"\nTest image: {img_name}")
                    logger.info(f"Predicted class: {CLASS_NAMES[pred_class]}")
                    logger.info(f"Confidence: {confidence:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing test image {img_name}: {str(e)}")
        
        # Save test results
        with open(os.path.join(DEBUG_DIR, 'model_verification.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def analyze_predictions(predictions):
    """Analyze prediction distribution"""
    pred_array = predictions[0]
    
    # Basic statistics
    mean_pred = np.mean(pred_array)
    std_pred = np.std(pred_array)
    max_pred = np.max(pred_array)
    min_pred = np.min(pred_array)
    
    # Entropy calculation
    entropy = -np.sum(pred_array * np.log2(pred_array + 1e-10))
    
    # Count predictions above threshold
    threshold = 0.1
    above_threshold = np.sum(pred_array > threshold)
    
    logger.info("\nPrediction Analysis:")
    logger.info(f"Mean: {mean_pred:.4f}")
    logger.info(f"Std Dev: {std_pred:.4f}")
    logger.info(f"Max: {max_pred:.4f}")
    logger.info(f"Min: {min_pred:.4f}")
    logger.info(f"Entropy: {entropy:.4f}")
    logger.info(f"Classes above {threshold}: {above_threshold}")
    
    return {
        'mean': float(mean_pred),
        'std_dev': float(std_pred),
        'entropy': float(entropy),
        'classes_above_threshold': int(above_threshold)
    }

# Load and verify model at startup
try:
    logger.info("\nLoading model...")
    # Try loading the checkpoint model first
    model_path = 'model_checkpoint.keras'
    if not os.path.exists(model_path):
        model_path = 'trained_plant_disease_model.keras'
    logger.info(f"Loading model from: {model_path}")
    
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    
    # Print model architecture and summary
    model.summary(print_fn=logger.info)
    
    # Load class indices from training
    class_indices = load_class_indices()
    if class_indices:
        logger.info("\nLoaded class indices from training:")
        for name, idx in class_indices.items():
            logger.info(f"{idx}: {name}")
    
    # Verify model with test images
    verify_model_with_test_images()
    
    # Add prediction debugging
    def debug_prediction(processed_image):
        predictions = model.predict(processed_image, verbose=0)
        pred_array = predictions[0]
        
        # Log all predictions
        logger.info("\nAll predictions:")
        sorted_preds = sorted(enumerate(pred_array), key=lambda x: x[1], reverse=True)
        for idx, prob in sorted_preds[:5]:  # Show top 5 predictions
            logger.info(f"{CLASS_NAMES[idx]}: {prob*100:.2f}%")
            
        # Check for prediction issues
        if np.all(pred_array == pred_array[0]):
            logger.warning("WARNING: All predictions are identical!")
        if np.max(pred_array) > 0.99:
            logger.warning("WARNING: Extremely high confidence (>99%)")
            
        return predictions

except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def validate_image(image):
    """Validate image before processing"""
    if image.size[0] < 50 or image.size[1] < 50:
        raise ValueError("Image is too small. Minimum size is 50x50 pixels.")
    if image.size[0] > 4096 or image.size[1] > 4096:
        raise ValueError("Image is too large. Maximum size is 4096x4096 pixels.")
    return True

def preprocess_image(image):
    """Convert uploaded image to model input format"""
    try:
        logger.info(f"Original image size: {image.size}, Mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((128, 128))
        logger.info(f"Resized image size: {image.size}")
        
        # Convert to array and normalize
        image_array = np.array(image)
        logger.info(f"Image array shape: {image_array.shape}")
        logger.info(f"Array values before normalization - Min: {image_array.min()}, Max: {image_array.max()}")
        
        # Save original preprocessed image
        save_debug_image(image_array, "original")
        
        # Normalize
        image_array = image_array.astype('float32') / 255.0
        logger.info(f"Array values after normalization - Min: {image_array.min():.3f}, Max: {image_array.max():.3f}")
        
        # Save normalized image
        save_debug_image(image_array, "normalized")
        
        # Expand dimensions
        processed_image = np.expand_dims(image_array, axis=0)
        
        # Check for invalid values
        if np.any(np.isnan(processed_image)) or np.any(np.isinf(processed_image)):
            raise ValueError("Invalid values in processed image")
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    model_info = {
        'accuracy': '96.48%',
        'validation_accuracy': '95.16%',
        'auc': '0.9989',
        'loss': '0.1079',
        'total_classes': len(CLASS_NAMES),
        'image_size': '128x128',
        'training_epochs': '10',
        'last_trained': 'Current Session'
    }
    
    return render_template('index.html', 
                         class_names=CLASS_NAMES,
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400

        logger.info(f"\n\nProcessing new request: {file.filename} ({file.content_type})")
        
        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image details
        logger.info(f"Original image size: {image.size}")
        logger.info(f"Image mode: {image.mode}")
        
        processed_image = preprocess_image(image)
        
        # Make predictions with debugging
        logger.info("\nMaking prediction...")
        predictions = debug_prediction(processed_image)
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Log prediction details
        logger.info(f"\nPredicted class index: {predicted_class}")
        logger.info(f"Predicted class name: {CLASS_NAMES[predicted_class]}")
        logger.info(f"Confidence: {confidence*100:.2f}%")
        
        result = {
            'result': f"{CLASS_NAMES[predicted_class]} ({confidence*100:.2f}%)",
            'confidence': confidence,
            'class_name': CLASS_NAMES[predicted_class],
            'all_predictions': {name: float(pred) for name, pred in zip(CLASS_NAMES, predictions[0])},
            'analysis': analyze_predictions(predictions)
        }
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Development server with debugging
    app.run(host='0.0.0.0', port=5000, debug=True)