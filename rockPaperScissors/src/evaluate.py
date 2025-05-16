import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from data_processing import load_processed_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_attention_maps(model, sample_images, num_samples=4):
    """Robust attention visualization that handles any attention weights shape"""
    try:
        # Get all multi-head attention layers
        attention_layers = [layer for layer in model.layers 
                          if 'multi_head_attention' in layer.name]
        
        if not attention_layers:
            print("No attention layers found in model")
            return
        
        print(f"Found {len(attention_layers)} attention layers")
        
        # Create model that outputs attention weights
        attention_outputs = [layer.output[1] for layer in attention_layers]
        attention_model = tf.keras.Model(
            inputs=model.input,
            outputs=attention_outputs
        )
        
        # Get attention weights
        all_attentions = attention_model.predict(sample_images[:num_samples])
        
        # If single layer, wrap in list
        if len(attention_layers) == 1:
            all_attentions = [all_attentions]
        
        for i in range(num_samples):
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, len(attention_layers)+1, 1)
            plt.imshow(sample_images[i].numpy().squeeze())
            plt.title('Input Image')
            plt.axis('off')
            
            # Plot each attention layer
            for layer_idx, attentions in enumerate(all_attentions, 1):
                plt.subplot(1, len(attention_layers)+1, layer_idx+1)
                
                # Get attention for this sample and layer
                sample_attn = attentions[i] if len(attentions) > i else attentions
                
                # Handle different attention weight shapes
                if sample_attn.ndim == 4:  # (heads, queries, keys)
                    avg_attn = np.mean(sample_attn, axis=0)  # avg across heads
                    if avg_attn.ndim == 2:  # (queries, keys)
                        sns.heatmap(avg_attn, cmap='viridis')
                    else:
                        # Take first query if still too many dimensions
                        sns.heatmap(avg_attn[0], cmap='viridis')
                elif sample_attn.ndim == 3:  # (queries, keys)
                    sns.heatmap(sample_attn, cmap='viridis')
                else:  # scalar or unexpected shape
                    print(f"Unexpected attention shape {sample_attn.shape}, skipping")
                    continue
                
                plt.title(f'Layer {layer_idx} Attention')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error visualizing attention: {str(e)}")
        import traceback
        traceback.print_exc()

        

def evaluate_model():
    # Load model with custom objects
    try:
        model = tf.keras.models.load_model(
            'rps_vit_model.keras'
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test dataset
    test_ds = load_processed_dataset('processed_dataset/test')
    
    # Generate predictions
    y_true, y_pred = [], []
    sample_images, sample_labels = next(iter(test_ds))  # For attention visualization
    
    for images, labels in test_ds:
        y_true.extend(tf.argmax(labels, axis=-1))
        y_pred.extend(tf.argmax(model.predict(images, verbose=0), axis=-1))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, 
        target_names=['Rock', 'Paper', 'Scissors'],
        digits=4
    ))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, ['Rock', 'Paper', 'Scissors'])
    
    # Calculate and display test accuracy
    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nTest Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    evaluate_model()