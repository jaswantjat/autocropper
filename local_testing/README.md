# 🧪 Local Testing Guide

Complete guide for testing the Card Rectification API locally with multiple tools and methods.

## 🚀 Quick Start

### 1. Start the API Server
```bash
# Install dependencies
pip install -r requirements.txt

# Download the model (optional but recommended)
python download_model.py

# Start the server
python app.py
```

The API will be available at: `http://localhost:5000`

### 2. Verify API is Running
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_status": "ready",
  "model_info": {
    "loaded": true,
    "device": "cpu",
    "cuda_available": false
  }
}
```

## 🛠️ Testing Tools

### 1. 📝 HTML Test Form
**File**: `local_testing/test_form.html`

**Usage**:
1. Open `test_form.html` in your web browser
2. The form will automatically check API health
3. Test file uploads by selecting image files
4. Test base64 processing with image URLs
5. View API information

**Features**:
- ✅ Real-time API health checking
- 📁 File upload testing with drag & drop
- 🔗 Base64 endpoint testing
- 📊 Visual feedback and error handling
- 💾 Download processed results

### 2. 🖥️ cURL Commands
**File**: `local_testing/curl_commands.sh`

**Usage**:
```bash
# View all commands
./curl_commands.sh

# Run all tests automatically
./curl_commands.sh run
```

**Tests Included**:
- API information endpoint
- Health check
- File upload with test image
- Base64 processing
- Error scenarios (404, 405, 400)
- Performance timing

### 3. 🐍 Python Base64 Tester
**File**: `local_testing/test_base64.py`

**Usage**:
```bash
# Install dependencies
pip install requests pillow

# Run the test
python local_testing/test_base64.py
```

**Features**:
- 🖼️ Generates synthetic test images
- 📤 Tests multiple image formats (PNG, JPEG)
- 🔗 Tests both data URL and plain base64 formats
- ⚡ Performance timing
- ❌ Error scenario testing
- 💾 Option to save results

### 4. 📄 PDF Endpoints Tester
**File**: `local_testing/test_pdf_endpoints.py`

**Usage**:
```bash
# Install dependencies
pip install requests pillow

# Run the test
python local_testing/test_pdf_endpoints.py
```

**Features**:
- 📄 Tests single image to PDF conversion
- 📚 Tests multiple images to PDF conversion
- 📦 Tests multiple images to ZIP (with individual PNGs + PDF)
- ⚙️ Tests custom page sizes (A4, letter, legal)
- 🎛️ Tests quality settings
- ⚡ Performance timing
- ❌ Error scenario testing
- 💾 Saves test results for verification

## 📋 Testing Checklist

### ✅ Basic Functionality
- [ ] API starts without errors
- [ ] Health endpoint returns 200
- [ ] Model loads successfully (if model file exists)
- [ ] API info endpoint works

### ✅ File Upload Testing
- [ ] PNG files upload and process
- [ ] JPEG files upload and process
- [ ] BMP files upload and process
- [ ] Invalid file types are rejected
- [ ] Empty files are rejected
- [ ] Large files are handled appropriately

### ✅ Base64 Testing
- [ ] Plain base64 strings work
- [ ] Data URL format works
- [ ] Invalid base64 is rejected
- [ ] Empty data is rejected
- [ ] Large images process correctly

### ✅ PDF Processing Testing
- [ ] Single image to PDF works
- [ ] Multiple images to PDF works
- [ ] Custom page sizes work (A4, letter, legal)
- [ ] Quality settings work (1-100)
- [ ] ZIP with images and PDF works
- [ ] Invalid files are rejected for PDF endpoints
- [ ] Empty file lists are rejected

### ✅ Error Handling
- [ ] 404 for non-existent endpoints
- [ ] 405 for wrong HTTP methods
- [ ] 400 for malformed requests
- [ ] 503 when model not loaded
- [ ] Proper error messages returned

## 🔍 Model Verification

### Check Model Status
```bash
curl http://localhost:5000/health | jq '.model_info'
```

### Expected Responses

**Model Loaded Successfully**:
```json
{
  "loaded": true,
  "device": "cpu",
  "cuda_available": false,
  "error": null
}
```

**Model Not Available**:
```json
{
  "loaded": false,
  "device": "unknown",
  "cuda_available": false,
  "error": "Model file CRDN1000.pkl not found. Please run download_model.py first."
}
```

### Download Model
```bash
python download_model.py
```

### Verify Model Processing
1. Upload a test image using the HTML form
2. Check that the result is different from the input
3. Verify the output is a properly rectified card image

## 🐛 Troubleshooting

### API Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Try a different port
PORT=8000 python app.py
```

### Model Loading Issues
```bash
# Check if model file exists
ls -la CRDN1000.pkl

# Download model if missing
python download_model.py

# Check model loading logs
python app.py | grep -i model
```

### Connection Issues
```bash
# Test basic connectivity
curl -v http://localhost:5000/health

# Check firewall settings
# Ensure localhost connections are allowed
```

### Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f "python app.py")

# Force CPU mode if GPU memory issues
FORCE_CPU=true python app.py
```

## 📊 Performance Testing

### Response Time Testing
```bash
# Test health endpoint speed
time curl -s http://localhost:5000/health > /dev/null

# Test file processing speed
time curl -X POST -F "file=@test_image.jpg" http://localhost:5000/api/process-id --output result.png
```

### Load Testing (Optional)
```bash
# Install apache bench
brew install httpd  # macOS
# or
sudo apt-get install apache2-utils  # Ubuntu

# Test concurrent requests
ab -n 100 -c 10 http://localhost:5000/health
```

## 🔧 Configuration Testing

### Environment Variables
```bash
# Test different configurations
DEBUG=true python app.py
FORCE_CPU=true python app.py
LOG_LEVEL=DEBUG python app.py
PORT=8000 python app.py
```

### Test Different Image Sizes
```bash
# Create test images of various sizes
convert -size 100x100 xc:blue small_test.png
convert -size 1000x1000 xc:red large_test.png
convert -size 5000x5000 xc:green huge_test.png

# Test with each size
curl -X POST -F "file=@small_test.png" http://localhost:5000/api/process-id --output small_result.png
```

## 📝 Test Results Documentation

### Expected Processing Times
- Health check: < 100ms
- Small image (< 1MB): < 5 seconds
- Large image (> 5MB): < 30 seconds

### Expected File Sizes
- Input: Any size up to 16MB
- Output: Typically smaller than input (PNG compression)

### Success Indicators
- ✅ HTTP 200 responses for valid requests
- ✅ Proper image files returned
- ✅ Reasonable processing times
- ✅ No memory leaks during extended testing
- ✅ Proper error messages for invalid inputs

## 🎯 Next Steps

After local testing is successful:
1. Run the pytest suite: `python run_tests.py`
2. Test with real ID card images
3. Verify model accuracy with known test cases
4. Prepare for deployment testing
5. Set up monitoring and logging
