#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import sys
import cv2
import time
import numpy as np
from fractions import Fraction

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("debug_receiver")

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    try:
        import aiohttp
        logger.info("✅ aiohttp is installed")
    except ImportError:
        missing_deps.append("aiohttp")
        logger.error("❌ aiohttp is not installed")
    
    try:
        import aiortc
        logger.info("✅ aiortc is installed")
    except ImportError:
        missing_deps.append("aiortc")
        logger.error("❌ aiortc is not installed")
    
    try:
        import av
        logger.info("✅ av is installed")
    except ImportError:
        missing_deps.append("av")
        logger.error("❌ av is not installed")
    
    # Check OpenCV
    try:
        logger.info(f"✅ OpenCV is installed (version: {cv2.__version__})")
        
        # Test if OpenCV can create a window
        try:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test")
            logger.info("✅ OpenCV can create windows")
        except Exception as e:
            logger.error(f"❌ OpenCV cannot create windows: {e}")
            logger.error("This might be due to missing display or X11 forwarding issues")
    except Exception as e:
        missing_deps.append("opencv-python")
        logger.error(f"❌ Error with OpenCV: {e}")
    
    return missing_deps

def check_files():
    """Check if all required files exist."""
    files_to_check = [
        "webrtc_receiver.py",
        "webrtc_receiver.html",
        "webrtc.js"
    ]
    
    missing_files = []
    
    for file in files_to_check:
        if os.path.exists(file):
            logger.info(f"✅ {file} exists")
        else:
            missing_files.append(file)
            logger.error(f"❌ {file} is missing")
    
    return missing_files

async def test_display():
    """Test if OpenCV display works."""
    logger.info("Testing OpenCV display...")
    
    try:
        # Create a test image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:] = (0, 0, 255)  # Red background
        
        # Add some text
        cv2.putText(img, "Display Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show the image
        cv2.namedWindow("Display Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Display Test", img)
        
        logger.info("If you can see a red window with 'Display Test' text, the display is working")
        logger.info("Press any key in the image window to continue...")
        
        # Wait for a key press with a timeout
        for i in range(100):  # 10 seconds timeout
            key = cv2.waitKey(100) & 0xFF
            if key != 255:
                break
            await asyncio.sleep(0.1)
        
        cv2.destroyAllWindows()
        logger.info("Display test completed")
        return True
    except Exception as e:
        logger.error(f"❌ Display test failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Debug WebRTC receiver")
    parser.add_argument("--skip-display-test", action="store_true", help="Skip the display test")
    args = parser.parse_args()
    
    print("\nWebRTC Receiver Debugging Tool")
    print("=============================\n")
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    
    # Check files
    print("\nChecking required files...")
    missing_files = check_files()
    
    # Test display if not skipped
    display_works = True
    if not args.skip_display_test:
        print("\nTesting display...")
        display_works = await test_display()
    
    # Print summary
    print("\nDiagnostic Summary")
    print("-----------------")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Run the following command to install them:")
        print(f"   pip install {' '.join(missing_deps)}")
    else:
        print("✅ All dependencies are installed")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
    else:
        print("✅ All required files exist")
    
    if not display_works and not args.skip_display_test:
        print("❌ Display test failed")
        print("   This might be due to missing display or X11 forwarding issues")
        print("   If running in a headless environment, use --skip-display-test")
    elif not args.skip_display_test:
        print("✅ Display test passed")
    
    # Provide recommendations
    print("\nRecommendations:")
    if missing_deps or missing_files or (not display_works and not args.skip_display_test):
        print("1. Install missing dependencies")
        print("2. Make sure all required files exist")
        print("3. If running in a headless environment, use --skip-display-test")
        print("4. Try running the receiver with verbose logging:")
        print("   python3 webrtc_receiver.py --verbose")
    else:
        print("All checks passed! The receiver should work correctly.")
        print("If you're still having issues, try the following:")
        print("1. Make sure the sender is running and accessible")
        print("2. Check network connectivity between sender and receiver")
        print("3. Try running with verbose logging: python3 webrtc_receiver.py --verbose")

if __name__ == "__main__":
    asyncio.run(main())