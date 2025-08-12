#!/usr/bin/env python3
"""
PDF Endpoints Test Script
Tests the new PDF processing endpoints with various scenarios
"""

import os
import sys
import requests
import time
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np

API_BASE = "http://localhost:5000"

def create_test_image(width=400, height=300, color='blue', text="TEST"):
    """Create a synthetic test image."""
    img = Image.new('RGB', (width, height), color=color)
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Draw border and text
        draw.rectangle([10, 10, width-10, height-10], outline='black', width=3)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((50, height//2), text, fill='black', font=font)
        
    except ImportError:
        pass
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def test_api_health():
    """Test if the API is running and healthy."""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Model Status: {data.get('model_status')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_process_and_pdf_single():
    """Test PDF endpoint with single image."""
    print("\n📄 Testing: Single Image to PDF")
    
    try:
        # Create test image
        image_data = create_test_image(500, 350, 'lightblue', 'SINGLE TEST')
        
        # Prepare request
        files = {'files': ('test_single.png', image_data, 'image/png')}
        data = {
            'page_size': 'A4',
            'quality': '90'
        }
        
        print(f"   📤 Sending single image for PDF conversion...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/process-and-pdf",
            files=files,
            data=data,
            timeout=30
        )
        end_time = time.time()
        
        print(f"   ⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ Success! Received PDF file")
            print(f"   📄 PDF size: {len(response.content)} bytes")
            
            # Save PDF
            with open("test_single_result.pdf", "wb") as f:
                f.write(response.content)
            print(f"   💾 Saved as test_single_result.pdf")
            return True
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   ❌ HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_process_and_pdf_multiple():
    """Test PDF endpoint with multiple images."""
    print("\n📄 Testing: Multiple Images to PDF")
    
    try:
        # Create multiple test images
        images = [
            ('test1.png', create_test_image(400, 300, 'lightcoral', 'PAGE 1')),
            ('test2.png', create_test_image(450, 350, 'lightgreen', 'PAGE 2')),
            ('test3.png', create_test_image(500, 400, 'lightyellow', 'PAGE 3'))
        ]
        
        # Prepare request
        files = [('files', (name, data, 'image/png')) for name, data in images]
        data = {
            'page_size': 'letter',
            'quality': '85'
        }
        
        print(f"   📤 Sending {len(images)} images for PDF conversion...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/process-and-pdf",
            files=files,
            data=data,
            timeout=60
        )
        end_time = time.time()
        
        print(f"   ⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ Success! Received multi-page PDF")
            print(f"   📄 PDF size: {len(response.content)} bytes")
            
            # Save PDF
            with open("test_multiple_result.pdf", "wb") as f:
                f.write(response.content)
            print(f"   💾 Saved as test_multiple_result.pdf")
            return True
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   ❌ HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_process_multiple_zip():
    """Test multiple processing endpoint that returns ZIP."""
    print("\n📦 Testing: Multiple Images to ZIP")
    
    try:
        # Create multiple test images
        images = [
            ('card1.png', create_test_image(400, 300, 'lightblue', 'CARD 1')),
            ('card2.png', create_test_image(450, 350, 'lightpink', 'CARD 2'))
        ]
        
        # Prepare request
        files = [('files', (name, data, 'image/png')) for name, data in images]
        data = {
            'page_size': 'A4',
            'quality': '95'
        }
        
        print(f"   📤 Sending {len(images)} images for ZIP creation...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/process-multiple",
            files=files,
            data=data,
            timeout=60
        )
        end_time = time.time()
        
        print(f"   ⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ Success! Received ZIP file")
            print(f"   📦 ZIP size: {len(response.content)} bytes")
            
            # Save and analyze ZIP
            zip_filename = "test_multiple_result.zip"
            with open(zip_filename, "wb") as f:
                f.write(response.content)
            
            # List ZIP contents
            with zipfile.ZipFile(zip_filename, 'r') as zip_file:
                file_list = zip_file.namelist()
                print(f"   📋 ZIP contents ({len(file_list)} files):")
                for file in file_list:
                    info = zip_file.getinfo(file)
                    print(f"      - {file} ({info.file_size} bytes)")
            
            print(f"   💾 Saved as {zip_filename}")
            return True
        else:
            try:
                error_data = response.json()
                print(f"   ❌ Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   ❌ HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_error_scenarios():
    """Test various error scenarios."""
    print("\n❌ Testing Error Scenarios...")
    
    # Test 1: No files
    print("\n   Test 1: No files provided")
    try:
        response = requests.post(f"{API_BASE}/api/process-and-pdf", files={}, timeout=10)
        print(f"   Status: {response.status_code} (expected 400 or 503)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Invalid file type
    print("\n   Test 2: Invalid file type")
    try:
        files = {'files': ('test.txt', b'not an image', 'text/plain')}
        response = requests.post(f"{API_BASE}/api/process-and-pdf", files=files, timeout=10)
        print(f"   Status: {response.status_code} (expected 400 or 503)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Empty file
    print("\n   Test 3: Empty file")
    try:
        files = {'files': ('empty.png', b'', 'image/png')}
        response = requests.post(f"{API_BASE}/api/process-and-pdf", files=files, timeout=10)
        print(f"   Status: {response.status_code} (expected 400 or 503)")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Main test function."""
    print("🎯 Card Rectification API - PDF Endpoints Test")
    print("=" * 60)
    
    # Check if API is running
    if not test_api_health():
        print("\n❌ API is not accessible. Please ensure the API is running on http://localhost:5000")
        sys.exit(1)
    
    print(f"\n📋 Testing PDF Endpoints")
    
    # Run tests
    tests = [
        ("Single Image to PDF", test_process_and_pdf_single),
        ("Multiple Images to PDF", test_process_and_pdf_multiple),
        ("Multiple Images to ZIP", test_process_multiple_zip),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Test error scenarios
    test_error_scenarios()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print("\n🎉 PDF endpoint testing complete!")
    print("\nNote: If the model is not loaded, you'll see 503 errors.")
    print("This is expected behavior when the model file is not available.")
    
    # Cleanup option
    cleanup = input("\nClean up test files? (y/n): ").lower().strip()
    if cleanup == 'y':
        for file in ['test_single_result.pdf', 'test_multiple_result.pdf', 'test_multiple_result.zip']:
            try:
                os.remove(file)
                print(f"Removed {file}")
            except FileNotFoundError:
                pass
        print("✓ Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
