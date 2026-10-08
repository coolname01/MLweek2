# STATE (обновляется агентом после каждого шага)
- Железо: RTX 3060 12 GB, RAM 32 GB 3200 MHz, Ryzen 5 3600
- OS: Windows, PowerShell. venv: .venv (Python 3.12)
- torch/CUDA: (записать после проверки)
- Файл submission: submission.csv (+ копия Name_Surname.csv для Moodle)
- Baseline из блокнота: MLP val 0.356, CNN val 0.549 (32x32, Colab CPU)
- Лучшая модель: —   val acc: —
- Сделано: —
- Следующий шаг: ResNet (224 px, ImageNet-норм., разморозить layer4, свой classifier в nn.Sequential)
- Открытые вопросы: ансамбль и дообучение на train+val разрешены?
