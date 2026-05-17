import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("Starting latency test...")

# Test the latest trained model
model_path = "runs/cpu_long_training/apple_cpu_long/weights/best.pt"
if not os.path.exists(model_path):
    # Try alternative models
    for alt in ["apple_sensitive.pt", "apple_improved.pt", "apple_best.pt"]:
        if os.path.exists(alt):
            model_path = alt
            break

print(f"Testing model: {model_path}")

# Load model
start = time.time()
model = YOLO(model_path)
load_time = time.time() - start
print(f"Model loaded in {load_time:.3f}s")

# Get test images
image_dir = "apple_photos"
images = []
for ext in ['*.jpg', '*.png']:
    images.extend(glob.glob(os.path.join(image_dir, ext)))

if not images:
    print(f"No images found in {image_dir}")
    exit()

# Use first 20 images for testing
test_images = images[:20]
print(f"Testing with {len(test_images)} images")

# Warm up
print("Warming up...")
for img in test_images[:3]:
    model(img, verbose=False)

# Test latency
print("Testing latency...")
latencies = []
detections = []

for i, img in enumerate(test_images):
    start = time.time()
    result = model(img, verbose=False)
    latency = time.time() - start
    latencies.append(latency)
    
    if result and len(result) > 0:
        detections.append(len(result[0].boxes))
    else:
        detections.append(0)
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(test_images)}")

# Calculate stats
avg_latency = np.mean(latencies)
std_latency = np.std(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
fps = 1.0 / avg_latency if avg_latency > 0 else 0
avg_detections = np.mean(detections)

print("\n" + "="*50)
print("LATENCY TEST RESULTS")
print("="*50)
print(f"Model: {os.path.basename(model_path)}")
print(f"Average latency: {avg_latency:.4f}s ± {std_latency:.4f}s")
print(f"Min latency: {min_latency:.4f}s")
print(f"Max latency: {max_latency:.4f}s")
print(f"FPS: {fps:.2f}")
print(f"Average detections per image: {avg_detections:.2f}")
print(f"Total time for {len(test_images)} images: {sum(latencies):.2f}s")

# Create output directory
os.makedirs("latency_results", exist_ok=True)

# Plot 1: Latency trend
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(latencies)+1), latencies, 'b-', linewidth=2, alpha=0.7)
plt.axhline(y=avg_latency, color='r', linestyle='--', linewidth=2, label=f'Avg: {avg_latency:.4f}s')
plt.title('Inference Latency Trend', fontsize=14, fontweight='bold')
plt.xlabel('Image Number', fontsize=12)
plt.ylabel('Latency (seconds)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("latency_results/latency_trend.png", dpi=150)
print("\nSaved: latency_results/latency_trend.png")

# Plot 2: Latency distribution
plt.figure(figsize=(10, 6))
plt.hist(latencies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=avg_latency, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_latency:.4f}s')
plt.title('Latency Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Latency (seconds)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("latency_results/latency_distribution.png", dpi=150)
print("Saved: latency_results/latency_distribution.png")

# Plot 3: Detections vs Latency
plt.figure(figsize=(10, 6))
scatter = plt.scatter(detections, latencies, c=range(len(latencies)), 
                     cmap='viridis', alpha=0.7, s=100)
plt.colorbar(scatter, label='Image Index')
plt.title('Detections vs Latency', fontsize=14, fontweight='bold')
plt.xlabel('Number of Apples Detected', fontsize=12)
plt.ylabel('Latency (seconds)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("latency_results/detections_vs_latency.png", dpi=150)
print("Saved: latency_results/detections_vs_latency.png")

# Save report
with open("latency_results/latency_report.txt", "w") as f:
    f.write("="*50 + "\n")
    f.write("MODEL INFERENCE LATENCY REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Test Images: {len(test_images)} from {image_dir}\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Average Latency: {avg_latency:.4f} seconds\n")
    f.write(f"  Latency Std Dev: {std_latency:.4f} seconds\n")
    f.write(f"  Minimum Latency: {min_latency:.4f} seconds\n")
    f.write(f"  Maximum Latency: {max_latency:.4f} seconds\n")
    f.write(f"  FPS: {fps:.2f} frames per second\n")
    f.write(f"  Average Detections: {avg_detections:.2f} apples per image\n\n")
    
    f.write("LATENCY PERCENTILES:\n")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(latencies, p)
        f.write(f"  {p}% of images ≤ {val:.4f} seconds\n")
    
    f.write(f"\nTotal time for {len(test_images)} images: {sum(latencies):.2f} seconds\n")
    f.write(f"Estimated time for 100 images: {avg_latency*100:.2f} seconds\n")
    
    f.write("\nREAL-TIME PERFORMANCE ASSESSMENT:\n")
    if fps >= 30:
        f.write("  Excellent: Suitable for real-time applications (≥30 FPS)\n")
    elif fps >= 15:
        f.write("  Good: Suitable for near real-time applications (15-30 FPS)\n")
    elif fps >= 5:
        f.write("  Fair: Suitable for non-real-time applications (5-15 FPS)\n")
    else:
        f.write("  Poor: Not suitable for real-time applications (<5 FPS)\n")

print("\n" + "="*50)
print("TEST COMPLETE")
print("="*50)
print("Results saved to 'latency_results' folder:")
print("  1. latency_trend.png - Latency trend chart")
print("  2. latency_distribution.png - Distribution histogram")
print("  3. detections_vs_latency.png - Scatter plot")
print("  4. latency_report.txt - Detailed report")

print("\nKey findings:")
print(f"  • Average inference delay: {avg_latency*1000:.1f} ms per image")
print(f"  • Processing speed: {fps:.1f} FPS")
print(f"  • Real-time capability: {'Yes' if fps >= 15 else 'Limited' if fps >= 5 else 'No'}")

input("\nPress Enter to exit...")