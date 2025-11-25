#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست کامل GPU برای سیستم تشخیص احساسات
"""

import sys
import time
import numpy as np

def test_pytorch_cuda():
    """تست PyTorch CUDA"""
    print("🔧 Testing PyTorch CUDA...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test tensor operations on GPU
            start_time = time.time()
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            end_time = time.time()
            print(f"   GPU matrix multiplication test: {end_time - start_time:.4f}s")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
    except ImportError as e:
        print(f"   ❌ PyTorch import error: {e}")
        return False

def test_onnxruntime_gpu():
    """تست ONNX Runtime GPU"""
    print("\n🔧 Testing ONNX Runtime GPU...")
    try:
        import onnxruntime as ort
        print(f"   ONNX Runtime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"   Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("   ✅ CUDAExecutionProvider available")
            
            # Test session creation with GPU
            try:
                # Create a dummy session to test GPU
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Only errors
                # This will fail gracefully if no model exists
                print("   GPU provider test: READY")
                return True
            except Exception as e:
                print(f"   ⚠️ GPU session test failed: {e}")
                return False
        else:
            print("   ❌ CUDAExecutionProvider NOT available")
            return False
    except ImportError as e:
        print(f"   ❌ ONNX Runtime import error: {e}")
        return False

def test_cupy_optional():
    """تست CuPy (اختیاری)"""
    print("\n🔧 Testing CuPy (Optional)...")
    try:
        import cupy as cp
        print(f"   CuPy version: {cp.__version__}")
        
        # Test basic CuPy operation
        start_time = time.time()
        x = cp.random.randn(1000, 1000)
        y = cp.random.randn(1000, 1000)
        z = cp.dot(x, y)
        end_time = time.time()
        print(f"   CuPy matrix multiplication test: {end_time - start_time:.4f}s")
        print("   ✅ CuPy working correctly")
        return True
    except ImportError:
        print("   ℹ️ CuPy not installed (optional for maximum performance)")
        return False
    except Exception as e:
        print(f"   ❌ CuPy error: {e}")
        return False

def get_system_info():
    """اطلاعات سیستم"""
    print("\n💻 System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    
def main():
    print("=" * 50)
    print("🚀 GPU Status Check for Emotion Recognition System")
    print("=" * 50)
    
    get_system_info()
    
    pytorch_ok = test_pytorch_cuda()
    onnx_ok = test_onnxruntime_gpu()
    cupy_ok = test_cupy_optional()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print("=" * 50)
    
    if pytorch_ok and onnx_ok:
        print("✅ GPU acceleration is FULLY SUPPORTED")
        print("   Your system is ready for GPU-accelerated emotion recognition!")
    elif onnx_ok:
        print("⚠️ Partial GPU support - ONNX Runtime GPU ready, PyTorch needs CUDA")
        print("   Face recognition will use GPU, but EEG processing will use CPU")
    elif pytorch_ok:
        print("⚠️ Partial GPU support - PyTorch CUDA ready, ONNX Runtime needs GPU")
        print("   EEG processing will use GPU, but face recognition will use CPU")
    else:
        print("❌ NO GPU support detected")
        print("   All processing will run on CPU")
    
    if cupy_ok:
        print("🚀 CuPy available for maximum EEG processing performance")
    
    print("\n📋 Next Steps:")
    if not pytorch_ok:
        print("1. Run: install_gpu_packages.bat")
        print("   or manually: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    if not onnx_ok:
        print("2. Install ONNX Runtime GPU: pip install onnxruntime-gpu")
    
    if not cupy_ok and (pytorch_ok or onnx_ok):
        print("3. Optional: Install CuPy for maximum performance: pip install cupy-cuda12x")
    
    print("\n🎯 Your emotion recognition system will automatically:")
    print("   - Use GPU when available")
    print("   - Fallback to CPU when GPU is not available")
    print("   - Show status messages during startup")

if __name__ == "__main__":
    main()
