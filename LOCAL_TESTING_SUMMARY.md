# 🧪 Local Testing Tools Summary

Complete overview of all local testing tools and utilities for the Card Rectification API.

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
python start_local.py
```
- Checks dependencies
- Downloads model if needed
- Starts API server
- Opens test tools

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download model (optional)
python download_model.py

# Start API
python app.py
```

## 🛠️ Testing Tools Overview

| Tool | File | Purpose | Usage |
|------|------|---------|-------|
| 🌐 **HTML Test Form** | `local_testing/test_form.html` | Interactive web interface | Open in browser |
| 🖥️ **cURL Commands** | `local_testing/curl_commands.sh` | Command line testing | `./curl_commands.sh run` |
| 🐍 **Python Base64 Tester** | `local_testing/test_base64.py` | Base64 endpoint testing | `python test_base64.py` |
| 🧪 **Unit Test Suite** | `run_tests.py` | Comprehensive pytest suite | `python run_tests.py` |
| 🖼️ **Test Image Generator** | `local_testing/create_test_images.py` | Generate test images | `python create_test_images.py` |
| 🚀 **Startup Script** | `start_local.py` | Automated setup & launch | `python start_local.py` |

## 📋 Testing Scenarios Covered

### ✅ Basic Functionality
- [x] API health check
- [x] Model loading verification
- [x] File upload processing
- [x] Base64 image processing
- [x] Multiple image formats (PNG, JPG, BMP)

### ✅ Error Handling
- [x] Invalid file types
- [x] Corrupted images
- [x] Missing files
- [x] Invalid base64 data
- [x] Server errors (404, 405, 413, 500, 503)

### ✅ Performance Testing
- [x] Response time measurement
- [x] Different image sizes
- [x] Memory usage monitoring
- [x] Concurrent request handling

### ✅ Model Verification
- [x] Model loading status
- [x] GPU/CPU detection
- [x] Processing accuracy
- [x] Fallback to traditional methods

## 🎯 Test Files Created

### HTML Test Form (`test_form.html`)
**Features:**
- Real-time API health monitoring
- Drag & drop file upload
- Base64 URL testing
- Visual result display
- Error handling with clear messages

**Usage:**
1. Open `local_testing/test_form.html` in browser
2. Form auto-checks API health
3. Upload images or test base64 URLs
4. View processed results instantly

### cURL Test Suite (`curl_commands.sh`)
**Features:**
- Complete endpoint coverage
- Automated test execution
- Performance timing
- Error scenario testing
- JSON response formatting

**Commands:**
```bash
# View all commands
./local_testing/curl_commands.sh

# Run all tests
./local_testing/curl_commands.sh run
```

### Python Base64 Tester (`test_base64.py`)
**Features:**
- Synthetic image generation
- Multiple format testing
- Data URL and plain base64
- Performance measurement
- Result saving options

**Usage:**
```bash
python local_testing/test_base64.py
```

### Unit Test Suite (`run_tests.py`)
**Features:**
- 22 comprehensive tests
- Dependency mocking
- CI/CD compatible
- Coverage reporting
- Deployment testing

**Usage:**
```bash
# Basic run
python run_tests.py

# With coverage
python run_tests.py -c

# Verbose output
python run_tests.py -v
```

### Test Image Generator (`create_test_images.py`)
**Features:**
- Various card layouts
- Different sizes and formats
- Skewed/rotated cards
- Noisy images
- Quality variants

**Usage:**
```bash
python local_testing/create_test_images.py
```

## 🔍 Model Verification Steps

### 1. Check Model Status
```bash
curl http://localhost:5000/health | jq '.model_info'
```

### 2. Expected Responses

**✅ Model Loaded:**
```json
{
  "loaded": true,
  "device": "cpu",
  "cuda_available": false,
  "error": null
}
```

**⚠️ Model Not Available:**
```json
{
  "loaded": false,
  "device": "unknown",
  "cuda_available": false,
  "error": "Model file CRDN1000.pkl not found..."
}
```

### 3. Download Model
```bash
python download_model.py
```

### 4. Test Processing
```bash
# Create test image
python local_testing/create_test_images.py

# Test processing
curl -X POST -F "file=@test_images/test_card.jpg" \
     http://localhost:5000/api/process-id \
     --output result.png

# Verify result is different from input
```

## 🐛 Troubleshooting Guide

### API Won't Start
```bash
# Check port availability
lsof -i :5000

# Try different port
PORT=8000 python app.py
```

### Dependencies Missing
```bash
# Auto-install
python start_local.py

# Manual install
pip install -r requirements.txt
```

### Model Issues
```bash
# Check model file
ls -la CRDN1000.pkl

# Download if missing
python download_model.py

# Force CPU mode
FORCE_CPU=true python app.py
```

### Connection Issues
```bash
# Test connectivity
curl -v http://localhost:5000/health

# Check firewall
# Ensure localhost connections allowed
```

## 📊 Expected Performance

### Response Times
- Health check: < 100ms
- Small image (< 1MB): < 5 seconds
- Large image (> 5MB): < 30 seconds

### Success Indicators
- ✅ HTTP 200 for valid requests
- ✅ Proper image files returned
- ✅ Reasonable processing times
- ✅ Clear error messages for invalid inputs

## 📖 Documentation Files

| File | Description |
|------|-------------|
| `local_testing/README.md` | Comprehensive testing guide |
| `tests/README.md` | Unit test documentation |
| `API_DOCUMENTATION.md` | Complete API reference |
| `INTEGRATION_SUMMARY.md` | Integration overview |

## 🎉 Quick Verification

Run this complete test sequence:

```bash
# 1. Start API
python start_local.py

# 2. In another terminal, run quick tests
curl http://localhost:5000/health
./local_testing/curl_commands.sh run
python local_testing/test_base64.py
python run_tests.py

# 3. Open browser test
open local_testing/test_form.html
```

All tools are designed to work together and provide comprehensive coverage of the API functionality. Choose the testing method that best fits your workflow!
