import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

import subprocess
result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                      capture_output=True, text=True)
print("GPU Info:", result.stdout)