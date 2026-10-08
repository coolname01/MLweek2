# STATE (обновляется агентом после каждого шага)
- Железо: NVIDIA GeForce RTX 3060 12.0 GB, RAM 32 GB 3200 MHz, Ryzen 5 3600
- OS: Windows 11, PowerShell. venv: .venv (Python 3.12)
- torch/CUDA: torch 2.14.1+cu126, CUDA 12.6, torch.cuda.is_available()=True
- Файл submission: submission.csv (+ копия Name_Surname.csv для Moodle)
- Baseline из блокнота: MLP val 0.356, CNN val 0.549 (32x32, Colab CPU)
- Лучшая модель: ResNet50 (layer4 + classifier)   val acc: 0.9499 (epoch 10/15, sklearn совпал)
- Сделано: ResNet50, layer4+голова. Кэш uint8 (N, 3, 256, 256) для train/val/test, аугментации батчем на GPU. SEED=42, cudnn deterministic=True, benchmark=False. Val acc 0.9499 (эпоха 12/15, sklearn совпал) — как у CPU-пайплайна 0.9499. Среднее время эпохи 18.7 с (было 55.8 с).
- Следующий шаг: сравнение с MLP/CNN (таблица + графики) и confusion matrix на val
- Открытые вопросы: ансамбль и дообучение на train+val разрешены?
