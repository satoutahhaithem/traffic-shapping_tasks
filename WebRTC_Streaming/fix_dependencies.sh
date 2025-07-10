#!/bin/bash
echo "--- Uninstalling potentially broken libraries (numpy, opencv) ---"
pip uninstall -y numpy opencv-python
echo ""
echo "--- Reinstalling libraries ---"
pip install numpy opencv-python
echo ""
echo "--- Dependency fix process complete. ---"
echo "You can now try running the receiver script again."