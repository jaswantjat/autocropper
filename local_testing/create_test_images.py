#!/usr/bin/env python3
"""
Test Image Generator
Creates various test images for testing the Card Rectification API
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2

def create_simple_card(width=600, height=400, filename="test_card.jpg"):
    """Create a simple ID card-like test image."""
    # Create base image
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add border
    cv2.rectangle(img, (20, 20), (width-20, height-20), (50, 50, 50), 3)
    
    # Add header rectangle
    cv2.rectangle(img, (40, 40), (width-40, 100), (100, 150, 200), -1)
    
    # Add text areas (rectangles to simulate text)
    cv2.rectangle(img, (60, 130), (300, 160), (0, 0, 0), -1)  # Name field
    cv2.rectangle(img, (60, 180), (250, 210), (0, 0, 0), -1)  # ID field
    cv2.rectangle(img, (60, 230), (200, 260), (0, 0, 0), -1)  # Date field
    
    # Add photo area
    cv2.rectangle(img, (width-150, 120), (width-50, 280), (150, 100, 100), -1)
    
    # Add some geometric shapes for corner detection
    cv2.circle(img, (100, 320), 20, (255, 0, 0), -1)
    cv2.rectangle(img, (200, 300), (250, 340), (0, 255, 0), -1)
    
    # Save image
    cv2.imwrite(filename, img)
    print(f"✅ Created: {filename}")
    return filename

def create_skewed_card(width=600, height=400, filename="skewed_card.jpg"):
    """Create a skewed/rotated card for testing perspective correction."""
    # Create base card
    img = np.ones((height*2, width*2, 3), dtype=np.uint8) * 255  # White background
    
    # Create card in center
    card_img = np.ones((height, width, 3), dtype=np.uint8) * 220
    cv2.rectangle(card_img, (10, 10), (width-10, height-10), (0, 0, 0), 2)
    cv2.rectangle(card_img, (30, 30), (width-30, 80), (100, 100, 200), -1)
    cv2.rectangle(card_img, (50, 100), (300, 130), (0, 0, 0), -1)
    cv2.rectangle(card_img, (50, 150), (250, 180), (0, 0, 0), -1)
    cv2.rectangle(card_img, (width-120, 100), (width-30, 200), (200, 100, 100), -1)
    
    # Apply perspective transformation to make it skewed
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_points = np.float32([[50, 100], [width+100, 50], [width+50, height+100], [0, height+150]])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    skewed = cv2.warpPerspective(card_img, matrix, (width*2, height*2))
    
    # Blend with background
    mask = cv2.cvtColor(skewed, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    
    result = img.copy()
    result[mask > 0] = skewed[mask > 0]
    
    cv2.imwrite(filename, result)
    print(f"✅ Created: {filename}")
    return filename

def create_noisy_card(width=600, height=400, filename="noisy_card.jpg"):
    """Create a card with noise and artifacts."""
    # Create base card
    img = create_simple_card_array(width, height)
    
    # Add noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add some artifacts
    cv2.line(img, (0, height//2), (width, height//2), (100, 100, 100), 2)
    cv2.circle(img, (width//4, height//4), 30, (200, 200, 200), -1)
    
    cv2.imwrite(filename, img)
    print(f"✅ Created: {filename}")
    return filename

def create_simple_card_array(width, height):
    """Helper function to create card as numpy array."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (20, 20), (width-20, height-20), (50, 50, 50), 3)
    cv2.rectangle(img, (40, 40), (width-40, 100), (100, 150, 200), -1)
    cv2.rectangle(img, (60, 130), (300, 160), (0, 0, 0), -1)
    cv2.rectangle(img, (60, 180), (250, 210), (0, 0, 0), -1)
    cv2.rectangle(img, (width-150, 120), (width-50, 280), (150, 100, 100), -1)
    return img

def create_different_formats():
    """Create test images in different formats."""
    base_img = create_simple_card_array(500, 350)
    
    # PNG
    cv2.imwrite("test_card.png", base_img)
    print("✅ Created: test_card.png")
    
    # JPEG with different quality
    cv2.imwrite("test_card_hq.jpg", base_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("✅ Created: test_card_hq.jpg")
    
    cv2.imwrite("test_card_lq.jpg", base_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    print("✅ Created: test_card_lq.jpg")
    
    # BMP
    cv2.imwrite("test_card.bmp", base_img)
    print("✅ Created: test_card.bmp")

def create_size_variants():
    """Create test images of different sizes."""
    sizes = [
        (200, 150, "small_card.jpg"),
        (800, 600, "large_card.jpg"),
        (1200, 800, "xlarge_card.jpg")
    ]
    
    for width, height, filename in sizes:
        img = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Scale elements proportionally
        border = max(10, width // 30)
        cv2.rectangle(img, (border, border), (width-border, height-border), (50, 50, 50), max(1, border//3))
        
        header_height = height // 4
        cv2.rectangle(img, (border*2, border*2), (width-border*2, header_height), (100, 150, 200), -1)
        
        cv2.imwrite(filename, img)
        print(f"✅ Created: {filename}")

def main():
    """Create all test images."""
    print("🖼️  Creating Test Images for Card Rectification API")
    print("=" * 55)
    
    # Create output directory
    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    
    print(f"📁 Creating images in: {output_dir.absolute()}")
    print()
    
    # Create various test images
    create_simple_card()
    create_skewed_card()
    create_noisy_card()
    create_different_formats()
    create_size_variants()
    
    print()
    print("🎉 Test image creation complete!")
    print(f"📁 Images saved in: {output_dir.absolute()}")
    print()
    print("📋 Created images:")
    for img_file in sorted(output_dir.glob("*")):
        size_kb = img_file.stat().st_size / 1024
        print(f"   {img_file.name} ({size_kb:.1f} KB)")
    
    print()
    print("🧪 Usage:")
    print("   1. Use these images with the HTML test form")
    print("   2. Test with cURL: curl -X POST -F 'file=@test_card.jpg' http://localhost:5000/api/process-id --output result.png")
    print("   3. Copy to local_testing/ directory for easy access")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error creating test images: {e}")
        sys.exit(1)
