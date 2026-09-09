import numpy as np
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

import torch

def main():
    print("=======测试脚本启动=======")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU名称:{torch.cuda.get_device_name(0)}")
        print("测试代码运行成功，可以开始训练实验CCFB在向你招手")

if __name__ == "__main__":
    main()