# STATE (обновляется агентом после каждого шага)
- Железо: NVIDIA GeForce RTX 3060 12.0 GB, RAM 32 GB 3200 MHz, Ryzen 5 3600
- OS: Windows 11, PowerShell. venv: .venv (Python 3.12)
- torch/CUDA: torch 2.14.1+cu126, CUDA 12.6, torch.cuda.is_available()=True
- Файл submission: submission.csv (+ копия Name_Surname.csv для Moodle)
- Baseline из блокнота: MLP val 0.356, CNN val 0.549 (32x32, Colab CPU)
- Лучшая модель: —   val acc: —
- Сделано: CUDA ок. В contest_resnet.ipynb DATA_DIR=./data (SEED/cudnn не менялись). CSV: train 4999, val 699, test 8299.
- Следующий шаг: ResNet (224 px, ImageNet-норм., разморозить layer4, свой classifier в nn.Sequential)
- Открытые вопросы: ансамбль и дообучение на train+val разрешены?
