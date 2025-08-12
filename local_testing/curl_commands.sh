#!/bin/bash

# Card Rectification API - cURL Test Commands
# Run these commands to test all API endpoints

API_BASE="http://localhost:5000"
echo "🎯 Card Rectification API Test Commands"
echo "========================================"
echo "API Base URL: $API_BASE"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "${BLUE}$1${NC}"
    echo "----------------------------------------"
}

# Function to print commands
print_command() {
    echo -e "${YELLOW}Command:${NC} $1"
    echo ""
}

# Function to execute and show results
run_command() {
    echo -e "${YELLOW}Running:${NC} $1"
    echo ""
    eval $1
    echo ""
    echo "----------------------------------------"
    echo ""
}

# 1. API Information
print_section "📋 1. API Information"
print_command "curl -X GET $API_BASE/"
if [ "$1" = "run" ]; then
    run_command "curl -s -X GET $API_BASE/ | jq ."
fi

# 2. Health Check
print_section "🏥 2. Health Check"
print_command "curl -X GET $API_BASE/health"
if [ "$1" = "run" ]; then
    run_command "curl -s -X GET $API_BASE/health | jq ."
fi

# 3. File Upload Test (requires test image)
print_section "📁 3. File Upload Test"
echo "First, create a test image:"
print_command "curl -o test_image.jpg https://via.placeholder.com/600x400/0000FF/FFFFFF?text=TEST+CARD"
echo ""
print_command "curl -X POST -F \"file=@test_image.jpg\" $API_BASE/api/process-id --output result.png"
if [ "$1" = "run" ]; then
    echo "Creating test image..."
    curl -s -o test_image.jpg "https://via.placeholder.com/600x400/0000FF/FFFFFF?text=TEST+CARD"
    if [ -f "test_image.jpg" ]; then
        echo -e "${GREEN}✓ Test image created${NC}"
        run_command "curl -X POST -F \"file=@test_image.jpg\" $API_BASE/api/process-id --output result.png -w \"HTTP Status: %{http_code}\\n\""
        if [ -f "result.png" ]; then
            echo -e "${GREEN}✓ Result saved as result.png${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to create test image${NC}"
    fi
fi

# 4. Base64 Test
print_section "🔗 4. Base64 Test"
echo "Create base64 test data:"
print_command "echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' > base64_data.txt"
echo ""
print_command "curl -X POST -H \"Content-Type: application/json\" -d '{\"image\":\"data:image/png;base64,'$(cat base64_data.txt)'\"}' $API_BASE/api/process-id-base64"
if [ "$1" = "run" ]; then
    echo "Creating base64 test data..."
    echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==' > base64_data.txt
    run_command "curl -s -X POST -H \"Content-Type: application/json\" -d '{\"image\":\"data:image/png;base64,'$(cat base64_data.txt)'\"}' $API_BASE/api/process-id-base64 | jq ."
fi

# 5. Error Testing
print_section "❌ 5. Error Testing"

echo "5.1 Test 404 - Non-existent endpoint:"
print_command "curl -X GET $API_BASE/nonexistent"
if [ "$1" = "run" ]; then
    run_command "curl -s -X GET $API_BASE/nonexistent | jq ."
fi

echo "5.2 Test 405 - Wrong method:"
print_command "curl -X PUT $API_BASE/health"
if [ "$1" = "run" ]; then
    run_command "curl -s -X PUT $API_BASE/health | jq ."
fi

echo "5.3 Test 400 - Missing file:"
print_command "curl -X POST $API_BASE/api/process-id"
if [ "$1" = "run" ]; then
    run_command "curl -s -X POST $API_BASE/api/process-id | jq ."
fi

echo "5.4 Test 400 - Invalid JSON:"
print_command "curl -X POST -H \"Content-Type: application/json\" -d 'invalid-json' $API_BASE/api/process-id-base64"
if [ "$1" = "run" ]; then
    run_command "curl -s -X POST -H \"Content-Type: application/json\" -d 'invalid-json' $API_BASE/api/process-id-base64 | jq ."
fi

# 6. PDF Processing Test
print_section "📄 6. PDF Processing Test"
echo "6.1 Single image to PDF:"
print_command "curl -X POST -F \"files=@test_image.jpg\" -F \"page_size=A4\" -F \"quality=90\" $API_BASE/api/process-and-pdf --output result.pdf"
if [ "$1" = "run" ]; then
    if [ -f "test_image.jpg" ]; then
        run_command "curl -X POST -F \"files=@test_image.jpg\" -F \"page_size=A4\" -F \"quality=90\" $API_BASE/api/process-and-pdf --output result.pdf -w \"HTTP Status: %{http_code}\\n\""
        if [ -f "result.pdf" ]; then
            echo -e "${GREEN}✓ PDF result saved as result.pdf${NC}"
        fi
    else
        echo -e "${RED}✗ test_image.jpg not found${NC}"
    fi
fi

echo ""
echo "6.2 Multiple images to PDF:"
print_command "curl -X POST -F \"files=@test_image.jpg\" -F \"files=@test_image.jpg\" -F \"page_size=letter\" $API_BASE/api/process-and-pdf --output multi_result.pdf"
if [ "$1" = "run" ]; then
    if [ -f "test_image.jpg" ]; then
        run_command "curl -X POST -F \"files=@test_image.jpg\" -F \"files=@test_image.jpg\" -F \"page_size=letter\" $API_BASE/api/process-and-pdf --output multi_result.pdf -w \"HTTP Status: %{http_code}\\n\""
        if [ -f "multi_result.pdf" ]; then
            echo -e "${GREEN}✓ Multi-page PDF saved as multi_result.pdf${NC}"
        fi
    else
        echo -e "${RED}✗ test_image.jpg not found${NC}"
    fi
fi

# 7. Multiple Files Processing Test
print_section "📦 7. Multiple Files Processing Test"
print_command "curl -X POST -F \"files=@test_image.jpg\" -F \"files=@test_image.jpg\" -F \"page_size=A4\" -F \"quality=85\" $API_BASE/api/process-multiple --output result.zip"
if [ "$1" = "run" ]; then
    if [ -f "test_image.jpg" ]; then
        run_command "curl -X POST -F \"files=@test_image.jpg\" -F \"files=@test_image.jpg\" -F \"page_size=A4\" -F \"quality=85\" $API_BASE/api/process-multiple --output result.zip -w \"HTTP Status: %{http_code}\\n\""
        if [ -f "result.zip" ]; then
            echo -e "${GREEN}✓ ZIP result saved as result.zip${NC}"
            echo "ZIP contents:"
            unzip -l result.zip 2>/dev/null || echo "Cannot list ZIP contents (unzip not available)"
        fi
    else
        echo -e "${RED}✗ test_image.jpg not found${NC}"
    fi
fi

# 8. Performance Test
print_section "⚡ 8. Performance Test"
print_command "time curl -X GET $API_BASE/health"
if [ "$1" = "run" ]; then
    run_command "time curl -s -X GET $API_BASE/health > /dev/null"
fi

# Usage instructions
echo ""
print_section "📖 Usage Instructions"
echo "To just see the commands:"
echo "  ./curl_commands.sh"
echo ""
echo "To run all commands:"
echo "  ./curl_commands.sh run"
echo ""
echo "To run individual commands, copy and paste them from above."
echo ""
echo "Prerequisites:"
echo "  - jq (for JSON formatting): brew install jq"
echo "  - curl (usually pre-installed)"
echo "  - API running on $API_BASE"
echo ""

# Cleanup function
cleanup() {
    echo "Cleaning up test files..."
    rm -f test_image.jpg base64_data.txt result.png result.pdf multi_result.pdf result.zip
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# If running tests, offer cleanup
if [ "$1" = "run" ]; then
    echo ""
    echo "Test run complete!"
    echo ""
    read -p "Clean up test files? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
fi
