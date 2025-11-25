@echo off
echo ======================================================
echo Installing GPU packages for Emotion Recognition System
echo ======================================================

echo.
echo 🔧 Step 1: Backing up current environment...
pip freeze > backup_requirements.txt
echo Current packages backed up to backup_requirements.txt

echo.
echo 🗑️ Step 2: Uninstalling CPU-only packages...
pip uninstall torch torchvision torchaudio -y
pip uninstall onnxruntime -y

echo.
echo 📦 Step 3: Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 📦 Step 4: Installing ONNX Runtime GPU...
pip install onnxruntime-gpu==1.21.0

echo.
echo 🎯 Step 5: Installing optional GPU acceleration (CuPy)...
echo This is optional but recommended for maximum EEG processing performance
choice /c YN /m "Do you want to install CuPy for additional GPU acceleration"
if errorlevel 2 goto skip_cupy
pip install cupy-cuda12x
:skip_cupy

echo.
echo ✅ Step 6: Verifying installation...
echo Running GPU compatibility test...
python test_gpu_complete.py

echo.
echo 🎉 Installation completed!
echo Your emotion recognition system is now ready for GPU acceleration!
pause
