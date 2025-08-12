#!/usr/bin/env python3
"""
Local Development Startup Script
Helps set up and run the Card Rectification API locally with proper checks and guidance
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("🎯 Card Rectification API - Local Development")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'flask', 'opencv-python', 'numpy', 'pillow', 
        'torch', 'torchvision', 'scikit-image', 'imutils'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            return install_dependencies(missing_packages)
        else:
            print("❌ Cannot start without required dependencies")
            return False
    
    return True

def install_dependencies(packages=None):
    """Install missing dependencies."""
    print("\n📥 Installing dependencies...")
    
    if packages:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    else:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
    
    try:
        subprocess.check_call(cmd)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_model_file():
    """Check if model file exists and offer to download."""
    model_path = Path("CRDN1000.pkl")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {size_mb:.1f}MB")
        return True
    else:
        print("⚠️  Model file not found (CRDN1000.pkl)")
        print("   The API will work with traditional edge detection only")
        
        download = input("Download model file for better results? (y/n): ").lower().strip()
        if download == 'y':
            return download_model()
        else:
            print("ℹ️  Continuing without model file")
            return True

def download_model():
    """Download the model file."""
    print("\n📥 Downloading model file...")
    try:
        subprocess.check_call([sys.executable, 'download_model.py'])
        print("✅ Model downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to download model")
        print("   Continuing without model file")
        return True
    except FileNotFoundError:
        print("❌ download_model.py not found")
        return True

def check_port_availability(port=5000):
    """Check if the port is available."""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            print(f"✅ Port {port} is available")
            return True
        except OSError:
            print(f"❌ Port {port} is already in use")
            return False

def start_api_server():
    """Start the API server."""
    print("\n🚀 Starting API server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("-" * 30)
    
    try:
        # Import and run the app
        from app import main
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    
    return True

def open_test_tools():
    """Open testing tools in browser."""
    test_form_path = Path("local_testing/test_form.html").absolute()
    
    if test_form_path.exists():
        print(f"\n🌐 Opening test form: {test_form_path}")
        webbrowser.open(f"file://{test_form_path}")
    else:
        print("⚠️  Test form not found at local_testing/test_form.html")

def show_testing_instructions():
    """Show testing instructions."""
    print("\n📋 Testing Instructions:")
    print("=" * 30)
    print("1. 🌐 Web Form: Open local_testing/test_form.html in browser")
    print("2. 🖥️  cURL Tests: Run ./local_testing/curl_commands.sh")
    print("3. 🐍 Python Tests: Run python local_testing/test_base64.py")
    print("4. 🧪 Unit Tests: Run python run_tests.py")
    print("\n📖 Full documentation: local_testing/README.md")

def main():
    """Main startup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    check_model_file()
    
    # Check port availability
    if not check_port_availability():
        port_choice = input("Try a different port? (y/n): ").lower().strip()
        if port_choice == 'y':
            new_port = input("Enter port number (default 8000): ").strip()
            if not new_port:
                new_port = "8000"
            os.environ['PORT'] = new_port
            print(f"ℹ️  Using port {new_port}")
        else:
            print("❌ Cannot start on occupied port")
            sys.exit(1)
    
    # Ask about opening test tools
    open_tools = input("\nOpen test form in browser? (y/n): ").lower().strip()
    if open_tools == 'y':
        # Delay opening to let server start
        import threading
        def delayed_open():
            time.sleep(3)
            open_test_tools()
        threading.Thread(target=delayed_open, daemon=True).start()
    
    # Show testing instructions
    show_testing_instructions()
    
    # Start the server
    start_api_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)
