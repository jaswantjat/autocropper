#!/usr/bin/env python3
"""
Base64 Endpoint Test Script
Tests the /api/process-id-base64 endpoint with various image formats and scenarios
"""

import os
import sys
import base64
import requests
import json
import time
from io import BytesIO
from PIL import Image
import numpy as np

API_BASE = "http://localhost:5000"

def create_test_image(width=400, height=300, format='PNG'):
    """Create a synthetic test image."""
    # Create a simple test card image
    img = Image.new('RGB', (width, height), color='lightblue')
    
    # Add some text and shapes to make it look like a card
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Draw a border
        draw.rectangle([10, 10, width-10, height-10], outline='darkblue', width=3)
        
        # Add text
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((50, 50), "TEST ID CARD", fill='darkblue', font=font)
        draw.text((50, 100), "Name: John Doe", fill='black', font=font)
        draw.text((50, 130), "ID: 123456789", fill='black', font=font)
        draw.text((50, 160), "Valid: 2024-2030", fill='black', font=font)
        
        # Add some geometric shapes
        draw.ellipse([width-100, height-80, width-20, height-20], fill='red')
        draw.rectangle([20, height-50, 120, height-20], fill='green')
        
    except ImportError:
        print("PIL ImageDraw not available, using basic image")
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()

def image_to_base64(image_bytes, include_data_url=True):
    """Convert image bytes to base64 string."""
    b64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    if include_data_url:
        return f"data:image/png;base64,{b64_string}"
    return b64_string

def test_api_health():
    """Test if the API is running and healthy."""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model Status: {data.get('model_status')}")
            print(f"   Model Device: {data.get('model_info', {}).get('device', 'Unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_base64_endpoint(image_data, test_name, data_url=True):
    """Test the base64 endpoint with given image data."""
    print(f"\n🔗 Testing: {test_name}")
    
    try:
        # Prepare the request
        if data_url:
            base64_image = image_to_base64(image_data, include_data_url=True)
        else:
            base64_image = image_to_base64(image_data, include_data_url=False)
        
        payload = {"image": base64_image}
        
        print(f"   📤 Sending {len(base64_image)} characters of base64 data...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/process-id-base64",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        print(f"   ⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✅ Success! Received processed image")
                result_image = data.get('image', '')
                if result_image.startswith('data:image'):
                    print(f"   📷 Result image size: {len(result_image)} characters")
                    
                    # Optionally save the result
                    save_result = input("   💾 Save result image? (y/n): ").lower().strip()
                    if save_result == 'y':
                        save_base64_image(result_image, f"result_{test_name.replace(' ', '_').lower()}.png")
                else:
                    print(f"   ⚠️  Unexpected result format")
                return True
            else:
                print(f"   ❌ API returned success=false: {data.get('error', 'Unknown error')}")
                return False
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data.get('error', 'Unknown error')}")
                if 'details' in error_data:
                    print(f"   📝 Details: {error_data['details']}")
            except:
                print(f"   ❌ HTTP Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ⏰ Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def save_base64_image(base64_data, filename):
    """Save a base64 image to file."""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode and save
        image_bytes = base64.b64decode(base64_data)
        with open(filename, 'wb') as f:
            f.write(image_bytes)
        print(f"   💾 Saved result as {filename}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to save image: {e}")
        return False

def test_error_scenarios():
    """Test various error scenarios."""
    print("\n❌ Testing Error Scenarios...")
    
    # Test 1: Invalid JSON
    print("\n   Test 1: Invalid JSON")
    try:
        response = requests.post(
            f"{API_BASE}/api/process-id-base64",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected 400 or 503)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Missing image field
    print("\n   Test 2: Missing image field")
    try:
        response = requests.post(
            f"{API_BASE}/api/process-id-base64",
            json={"not_image": "data"},
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected 400 or 503)")
        if response.status_code in [400, 503]:
            data = response.json()
            print(f"   Error: {data.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Invalid base64
    print("\n   Test 3: Invalid base64 data")
    try:
        response = requests.post(
            f"{API_BASE}/api/process-id-base64",
            json={"image": "not-valid-base64!!!"},
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected 400 or 503)")
        if response.status_code in [400, 503]:
            data = response.json()
            print(f"   Error: {data.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Empty image data
    print("\n   Test 4: Empty image data")
    try:
        response = requests.post(
            f"{API_BASE}/api/process-id-base64",
            json={"image": ""},
            timeout=10
        )
        print(f"   Status: {response.status_code} (expected 400 or 503)")
        if response.status_code in [400, 503]:
            data = response.json()
            print(f"   Error: {data.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Main test function."""
    print("🎯 Card Rectification API - Base64 Endpoint Test")
    print("=" * 60)
    
    # Check if API is running
    if not test_api_health():
        print("\n❌ API is not accessible. Please ensure the API is running on http://localhost:5000")
        sys.exit(1)
    
    print(f"\n📋 Testing Base64 Endpoint: {API_BASE}/api/process-id-base64")
    
    # Test 1: PNG format with data URL
    png_image = create_test_image(400, 300, 'PNG')
    test_base64_endpoint(png_image, "PNG with Data URL", data_url=True)
    
    # Test 2: PNG format without data URL
    test_base64_endpoint(png_image, "PNG without Data URL", data_url=False)
    
    # Test 3: JPEG format
    jpeg_image = create_test_image(500, 350, 'JPEG')
    test_base64_endpoint(jpeg_image, "JPEG with Data URL", data_url=True)
    
    # Test 4: Different size
    large_image = create_test_image(800, 600, 'PNG')
    test_base64_endpoint(large_image, "Large PNG Image", data_url=True)
    
    # Test 5: Small image
    small_image = create_test_image(200, 150, 'PNG')
    test_base64_endpoint(small_image, "Small PNG Image", data_url=True)
    
    # Test error scenarios
    test_error_scenarios()
    
    print("\n🎉 Base64 endpoint testing complete!")
    print("\nNote: If the model is not loaded, you'll see 503 errors.")
    print("This is expected behavior when the model file is not available.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
