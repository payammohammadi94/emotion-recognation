# راهنمای نصب و تنظیم GPU برای سیستم تشخیص احساسات

## وضعیت فعلی سیستم شما

بر اساس تست انجام شده:
- ❌ PyTorch: فقط CPU (نسخه 2.6.0+cpu)
- ❌ ONNX Runtime: فقط CPU 
- ⚠️ GPU acceleration غیرفعال است

## نیازمندی‌های سیستم

### GPU پشتیبانی شده:
- NVIDIA GPU با معماری Kepler یا جدیدتر
- حداقل 2GB VRAM
- CUDA Compute Capability 3.5+

### نرم‌افزارهای مورد نیاز:
1. **NVIDIA GPU Driver** (آخرین نسخه)
2. **CUDA Toolkit 12.1** یا جدیدتر
3. **cuDNN 8.x** (اختیاری برای بهینه‌سازی بیشتر)

## مراحل نصب

### روش ۱: نصب خودکار (توصیه شده)
```batch
# اجرای فایل batch در Command Prompt
install_gpu_packages.bat
```

### روش ۲: نصب دستی
```bash
# حذف نسخه‌های CPU
pip uninstall torch torchvision torchaudio onnxruntime -y

# نصب PyTorch با CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# نصب ONNX Runtime GPU
pip install onnxruntime-gpu==1.21.0

# اختیاری: CuPy برای حداکثر کارایی
pip install cupy-cuda12x
```

## تست عملکرد

پس از نصب، اجرا کنید:
```bash
python test_gpu_complete.py
```

## بهینه‌سازی‌های انجام شده در کد

### 1. FaceRecognation/run_app.py
- ✅ تشخیص خودکار GPU/CPU
- ✅ Fallback به CPU در صورت عدم دسترسی GPU
- ✅ تنظیمات بهینه CUDA
- ✅ مدیریت memory GPU
- ✅ پیام‌های وضعیت

### 2. eegEmotionRecognation/fineeg.py
- ✅ استفاده از PyTorch CUDA
- ✅ Fallback برای CuPy operations
- ✅ بهینه‌سازی feature extraction
- ✅ نمایش اطلاعات GPU

### 3. کدهای جدید اضافه شده:
- **GPU detection و fallback logic**
- **Memory management برای GPU**
- **Error handling بهتر**
- **Status messages برای debugging**

## انتظارات عملکرد

### با GPU:
- **Face Recognition**: 3-5x سریعتر
- **EEG Processing**: 2-3x سریعتر (با CuPy)
- **Real-time processing**: روان‌تر
- **Memory usage**: کنترل شده

### بدون GPU:
- همه چیز روی CPU اجرا می‌شود
- کارایی کمتر اما همچنان قابل استفاده

## عیب‌یابی رایج

### GPU تشخیص داده نمی‌شود:
```bash
# بررسی driver
nvidia-smi

# بررسی CUDA
nvcc --version

# تست PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### خطاهای Memory:
- کاهش batch size
- کاهش `gpu_mem_limit` در کد
- بستن برنامه‌های دیگر که GPU استفاده می‌کنند

### Performance پایین:
- اطمینان از استفاده از آخرین driver
- بررسی thermal throttling
- استفاده از CuPy برای EEG processing

## توصیه‌های اضافی

1. **نظارت بر GPU**: استفاده از `nvidia-smi` برای monitoring
2. **Memory management**: تنظیم محدودیت memory در کد
3. **Batch processing**: استفاده از batch برای processing چندین frame
4. **Model optimization**: استفاده از TensorRT برای بهینه‌سازی بیشتر (اختیاری)

## فایل‌های ایجاد شده:
- `install_gpu_packages.bat`: نصب خودکار
- `test_gpu_complete.py`: تست کامل GPU
- `requirements_gpu.txt`: dependency list برای GPU
- `GPU_Setup_Guide.md`: این راهنما

