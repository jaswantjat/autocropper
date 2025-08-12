# Card Rectification API Documentation

## Overview
Complete Flask API application for ID card rectification using deep learning. The API provides endpoints for processing ID card images through file upload or base64 encoding, with automatic perspective correction and image enhancement.

## Features Implemented

### ✅ Core Requirements
- [x] Model loading at startup with GPU/CPU fallback
- [x] Health check endpoint at `/health`
- [x] File upload endpoint at `/api/process-id`
- [x] Base64 endpoint at `/api/process-id-base64`
- [x] Comprehensive error handling
- [x] Automatic temporary file cleanup
- [x] GPU and CPU inference support

### ✅ Additional Features
- [x] Detailed logging system
- [x] Model status monitoring
- [x] Input validation and sanitization
- [x] Proper HTTP status codes
- [x] API documentation endpoint
- [x] Thread-safe temporary file management
- [x] Graceful error recovery
- [x] Memory-efficient processing

## API Endpoints

### 1. Root Information - `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Card Rectification API",
  "version": "2.0.0",
  "description": "REST API for ID card rectification using deep learning",
  "endpoints": {...},
  "supported_formats": ["png", "jpg", "jpeg", "bmp"],
  "max_file_size": "16MB"
}
```

### 2. Health Check - `GET /health`
Comprehensive health check including model status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Card Rectification API is running",
  "version": "2.0.0",
  "model_status": "ready",
  "model_info": {
    "loaded": true,
    "device": "cuda:0",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3080"
  }
}
```

### 3. File Upload Processing - `POST /api/process-id`
Process ID card from uploaded file.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
- Content-Type: `image/png`
- File download with rectified card image

**Error Response:**
```json
{
  "error": "Processing failed",
  "details": "Could not detect card edges"
}
```

### 4. Base64 Processing - `POST /api/process-id-base64`
Process ID card from base64 encoded image.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Endpoint not found
- `405` - Method not allowed
- `413` - File too large (>16MB)
- `500` - Internal server error
- `503` - Service unavailable (model not loaded)

### Error Response Format
```json
{
  "error": "Error description",
  "details": "Detailed error information",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Technical Implementation

### Model Loading
- Automatic GPU detection and fallback to CPU
- Model validation at startup
- Error handling for missing model files
- Device information reporting

### Image Processing Pipeline
1. **Input Validation**: Format, size, and content validation
2. **Edge Detection**: CNN-based with traditional fallback
3. **Corner Detection**: Geometric analysis and validation
4. **Perspective Correction**: Four-point transformation
5. **Post-processing**: Image enhancement and formatting

### Memory Management
- Automatic temporary file cleanup
- Thread-safe file tracking
- Memory-efficient image processing
- Proper resource disposal

### Logging
- Structured logging with timestamps
- Different log levels (INFO, WARNING, ERROR)
- Request tracking and performance monitoring
- Error details for debugging

## Usage Examples

### Python Client
```python
import requests
import base64

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# File upload
with open('id_card.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/process-id', files=files)
    
with open('rectified_card.png', 'wb') as f:
    f.write(response.content)

# Base64 processing
with open('id_card.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

data = {'image': f'data:image/jpeg;base64,{image_data}'}
response = requests.post('http://localhost:5000/api/process-id-base64', json=data)
result = response.json()
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/health

# File upload
curl -X POST -F "file=@id_card.jpg" http://localhost:5000/api/process-id --output rectified.png

# Base64 (with JSON file)
curl -X POST -H "Content-Type: application/json" -d @image_data.json http://localhost:5000/api/process-id-base64
```

## Testing

Run the comprehensive test suite:
```bash
python test_api.py
```

Tests include:
- API information endpoint
- Health check functionality
- File upload processing
- Base64 image processing
- Error handling scenarios

## Deployment

### Local Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5000 app:app --timeout 120 --workers 1
```

### Environment Variables
- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: False)

## Performance Notes

- **Processing Time**: 2-10 seconds depending on image size and hardware
- **Memory Usage**: ~1-2GB for model loading
- **Concurrent Requests**: Single worker recommended for memory efficiency
- **File Size Limit**: 16MB maximum
- **Supported Formats**: PNG, JPG, JPEG, BMP

## Troubleshooting

### Common Issues
1. **Model not loading**: Ensure `CRDN1000.pkl` is in the project directory
2. **CUDA errors**: Check GPU drivers and PyTorch CUDA compatibility
3. **Memory errors**: Reduce image size or use CPU-only mode
4. **Processing failures**: Check image quality and card visibility

### Debug Mode
Enable debug logging by setting `DEBUG=true` environment variable.
