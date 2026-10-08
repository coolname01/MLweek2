# STATE (обновляется агентом после каждого шага)
- Железо: NVIDIA GeForce RTX 3060 12.0 GB, RAM 32 GB 3200 MHz, Ryzen 5 3600
- OS: Windows 11, PowerShell. venv: .venv (Python 3.12)
- torch/CUDA: torch 2.14.1+cu126, CUDA 12.6, torch.cuda.is_available()=True
- Файл submission: submission.csv (+ копия Name_Surname.csv для Moodle)
- Baseline из блокнота: MLP val 0.356, CNN val 0.549 (32x32, Colab CPU)
- Лучшая модель: ResNet50 layer4, аугментация A, seed 7. Чистый val 0.9442, искажённый 0.9099 (эпоха 9/15). Чекпоинт outputs/checkpoints/resnet50_robust.pt
- Сделано: искажённый val один раз (generator seed 1234567, outputs/val_corrupted.pt, n=699, 0/1/2+ искажений = 195/353/151). Эталон gpu_uint8 на нём: чистый 0.9499, искажённый 0.9084. Свип A/B/C × seed 42 и 7, 15 эпох. Выбор по среднему искажённому при падении чистого ≤ 0.01 от 0.949928: A 0.9471/0.9092 (оставлен), B 0.9399/0.9092 (чистый −0.0100, отсеян), C 0.9320/0.9235 (чистый −0.018, отсеян). Аугментации независимы на копии одного кадра (A/B/C); preview outputs/aug_preview.png. cudnn deterministic=True, benchmark=False, test не использовался.
- Следующий шаг: сравнение с MLP/CNN (таблица + графики) и confusion matrix на val
- Открытые вопросы: ансамбль и дообучение на train+val разрешены?
