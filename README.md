# Card Rectification API

A Flask REST API that deploys the [card-rectification algorithm](https://github.com/shakex/card-rectification) for ID card rectification. This API provides endpoints for both file upload and base64 image processing, specifically optimized for China 2nd-generation ID cards.

## Features

- **REST API Endpoints**: File upload and base64 image processing
- **Edge Detection**: Novel CNN-based edge detection with traditional fallback
- **Perspective Correction**: Automatic perspective transformation and rectification
- **Railway Deployment**: Ready-to-deploy configuration for Railway platform
- **Error Handling**: Comprehensive error handling and validation
- **Multiple Formats**: Supports PNG, JPG, JPEG, BMP input formats

## API Endpoints

### Health Check
```
GET /
```
Returns API status and version information.

### Health Check
```
GET /health
```
Check API and model status.

### File Upload Rectification
```
POST /api/process-id
Content-Type: multipart/form-data
```
Upload an image file containing an ID card for rectification.

**Parameters:**
- `file`: Image file (PNG, JPG, JPEG, BMP)

**Response:**
- Returns rectified card image as PNG file download

### Base64 Rectification
```
POST /api/process-id-base64
Content-Type: application/json
```
Process base64 encoded image for rectification.

**Request Body:**
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
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### PDF Processing
```
POST /api/process-and-pdf
Content-Type: multipart/form-data
```
Process one or more images and return as a single PDF.

**Parameters:**
- `files`: One or more image files
- `page_size`: Optional page size (A4, letter, legal) - default: A4
- `quality`: Optional JPEG quality 1-100 - default: 90

**Response:**
- Returns PDF file download

### Multiple Files Processing
```
POST /api/process-multiple
Content-Type: multipart/form-data
```
Process multiple images and return ZIP containing individual processed images and a combined PDF.

**Parameters:**
- `files`: Multiple image files
- `page_size`: Optional page size (A4, letter, legal) - default: A4
- `quality`: Optional JPEG quality 1-100 - default: 90

**Response:**
- Returns ZIP file download containing:
  - Individual processed images (PNG format)
  - Combined PDF with all processed images

## Example Usage

### cURL Examples

**Single Image Processing:**
```bash
curl -X POST -F "file=@card.jpg" \
     http://localhost:5000/api/process-id \
     --output rectified.png
```

**PDF Creation:**
```bash
curl -X POST \
     -F "files=@card1.jpg" \
     -F "files=@card2.jpg" \
     -F "page_size=A4" \
     -F "quality=90" \
     http://localhost:5000/api/process-and-pdf \
     --output cards.pdf
```

**Multiple Files to ZIP:**
```bash
curl -X POST \
     -F "files=@card1.jpg" \
     -F "files=@card2.jpg" \
     -F "page_size=letter" \
     http://localhost:5000/api/process-multiple \
     --output cards.zip
```

**Base64 Processing:**
```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"image":"data:image/jpeg;base64,/9j/4AAQ..."}' \
     http://localhost:5000/api/process-id-base64
```
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- Model weights file (`CRDN1000.pkl`)

### Local Development

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd card-rectification-api
```

2. **Download model weights:**
Download the `CRDN1000.pkl` file from the [original repository](https://github.com/shakex/card-rectification) and place it in the project root directory.

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 🧪 Local Testing

### Quick Start with Test Tools

1. **Easy Startup Script:**
```bash
python start_local.py
```
This script will:
- Check dependencies and install if needed
- Download model file if missing
- Start the API server
- Open test tools in your browser

2. **Manual Testing Tools:**

**🌐 HTML Test Form** - Interactive web interface:
```bash
# Open in browser
open local_testing/test_form.html
```

**🖥️ cURL Commands** - Command line testing:
```bash
# View all test commands
./local_testing/curl_commands.sh

# Run all tests automatically
./local_testing/curl_commands.sh run
```

**🐍 Python Base64 Tester** - Comprehensive base64 testing:
```bash
python local_testing/test_base64.py
```

**🧪 Unit Test Suite** - Complete pytest suite:
```bash
python run_tests.py
```

### Create Test Images

Generate various test images for testing:
```bash
python local_testing/create_test_images.py
```

### Verify Model is Working

1. **Check API Health:**
```bash
curl http://localhost:5000/health
```

2. **Expected Response (Model Loaded):**
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

3. **Test File Processing:**
```bash
curl -X POST -F "file=@test_image.jpg" \
     http://localhost:5000/api/process-id \
     --output result.png
```

### Testing Checklist

- [ ] ✅ API starts without errors
- [ ] ✅ Health endpoint returns 200
- [ ] ✅ Model loads successfully
- [ ] ✅ File upload works (PNG, JPG, BMP)
- [ ] ✅ Base64 endpoint works
- [ ] ✅ Error handling works properly
- [ ] ✅ Results are properly rectified

📖 **Full Testing Documentation:** `local_testing/README.md`

### Railway Deployment

1. **Prepare your repository:**
   - Ensure all files are committed to your Git repository
   - Make sure `CRDN1000.pkl` is included (or download it during deployment)

2. **Deploy to Railway:**
   - Connect your GitHub repository to Railway
   - Railway will automatically detect the `nixpacks.toml` configuration
   - The deployment will install dependencies and start the application

3. **Environment Variables:**
   No additional environment variables are required for basic functionality.

### Docker Deployment (Alternative)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120", "--workers", "1"]
```

## Model Weights

The application requires the pre-trained model weights file `CRDN1000.pkl`. You can:

1. **Download from original repository:**
   ```bash
   wget https://github.com/shakex/card-rectification/raw/master/CRDN1000.pkl
   ```

2. **Include in your repository** (if size permits)

3. **Download during deployment** (add to your deployment script)

## Usage Examples

### cURL Examples

**File Upload:**
```bash
curl -X POST \
  -F "file=@card_image.jpg" \
  http://localhost:5000/rectify \
  --output rectified_card.png
```

**Base64:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}' \
  http://localhost:5000/rectify/base64
```

### Python Example

```python
import requests
import base64

# File upload
with open('card_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/rectify', files=files)
    
with open('rectified_card.png', 'wb') as f:
    f.write(response.content)

# Base64
with open('card_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

data = {'image': f'data:image/jpeg;base64,{image_data}'}
response = requests.post('http://localhost:5000/rectify/base64', json=data)
result = response.json()
```

### JavaScript Example

```javascript
// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/rectify', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = URL.createObjectURL(blob);
    // Use the rectified image
});

// Base64
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
// ... draw image to canvas
const base64Data = canvas.toDataURL('image/jpeg');

fetch('/rectify/base64', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: base64Data })
})
.then(response => response.json())
.then(data => {
    // Use data.image (base64 encoded result)
});
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input (no file, invalid format, etc.)
- `413 Payload Too Large`: File size exceeds 16MB limit
- `500 Internal Server Error`: Processing failed

Example error response:
```json
{
  "error": "Processing failed: Could not detect card edges"
}
```

## Performance Notes

- **GPU Support**: The application will use GPU if CUDA is available, otherwise falls back to CPU
- **Processing Time**: Varies based on image size and hardware (typically 2-10 seconds)
- **Memory Usage**: Approximately 1-2GB RAM for model loading
- **Concurrent Requests**: Single worker configuration recommended for Railway deployment

## Limitations

- Optimized for China 2nd-generation ID cards
- May not work well with incomplete or heavily occluded cards
- Does not handle upside-down cards (rotation correction)
- Maximum file size: 16MB

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is based on the [card-rectification](https://github.com/shakex/card-rectification) algorithm, which is licensed under the Apache-2.0 License.

## Support

For issues related to:
- **API functionality**: Create an issue in this repository
- **Algorithm performance**: Refer to the [original repository](https://github.com/shakex/card-rectification)
- **Deployment**: Check Railway documentation or create an issue

## Acknowledgments

- Original algorithm by [shakex](https://github.com/shakex)
- Based on the [Recurrent Decoding Cell project](https://github.com/shakex/Recurrent-Decoding-Cell)
- Edge detection network with edge-consist-loss implementation
