# STATE (обновляется агентом после каждого шага)

\## Окружение

\- Железо: NVIDIA GeForce RTX 3060 12.0 GB, RAM 32 GB 3200 MHz, Ryzen 5 3600

\- OS: Windows 11, PowerShell. venv: .venv (Python 3.12)

\- torch 2.14.1+cu126, CUDA 12.6, torch.cuda.is\_available()=True

\- Submission: submission.csv (+ копия Name\_Surname.csv для Moodle)

\- Baseline из блокнота: MLP val 0.356, CNN val 0.549 (32x32, Colab CPU)



\## Правила

\- test используется только для инференса, не для обучения и подбора.

\- cudnn deterministic=True, benchmark=False. Seed и конфигурацию F не менять без решения пользователя.

\- Лимит: полный пайплайн ноутбука не более 45 мин (цель не более 40-42 мин).

\- Чекпоинты (\*.pt) не коммитить. Ноутбуки целиком не читать.

\- Выбор финала: по среднему (чистый+искажённый)/2 на val, public LB вторичный сигнал.

\- Ансамбль разрешён. TV1 (train+val) выполнен по явному запросу; val-метрики этого прогона недействительны и в выбор финала не идут.



\## Искажённый val

Собран один раз (generator seed 1234567, outputs/val\_corrupted.pt, n=699, искажений 0/1/2+ = 195/353/151). Эталон gpu\_uint8: чистый 0.9499, искажённый 0.9084.



\## Раунд 1 (свип A/B/C, seed 42 и 7, 15 эпох)

Критерий: чистый val не падает более чем на 0.01 от 0.949928, выбор по искажённому.

A 0.9471/0.9092 (оставлен); B 0.9399/0.9092 (чистый -0.0100, отсеян); C 0.9320/0.9235 (чистый -0.018, отсеян).



\## Раунд 2 (D/E/F, 15 эпох, seeds 42 и 7, layer3 lr 5e-5)

Среднее чистый/искажённый (score): D 0.9549/0.9084 (0.9317), E 0.9499/0.9378 (0.9438), F 0.9549/0.9456 (0.9503).

Лучший одиночный: F seed 7, эпоха 13: чистый 0.9599, искажённый 0.9471. TTA8: 0.9642 / 0.9528. Чекпоинт outputs/checkpoints/resnet50\_F\_seed7.pt.



\## Раунд 3 (проверка пайплайна)

\- Plain-траектория F воспроизведена, TTA8 F seed 7 = 0.9642 / 0.9528.

\- EMA (decay 0.999) хуже plain: финал 0.9070 / 0.8841 (seed 42), 0.9099 / 0.8913 (seed 7). Причина: за 1170 шагов 0.999 слишком высок.

\- ResNet101 seed 7: лучшая эпоха 14, 0.9557 / 0.9528, TTA8 0.9642 / 0.9499; 160 с на эпоху, пайплайн около 42 мин. Прироста нет, отклонён. Seed 42 не запускался.



\## Раунд 4

\- A (мультимасштаб на чекпоинте F seed 7): 256+288 TTA чистый 0.9700, искажённый 0.9514, score +0.0021 к TTA8 эталона (порог +0.004), не взят.

\- B (SWA, EMA 0.99): хуже plain. C (V2-веса): 0.9471/0.9471. D (CutMix, 20 эпох): 0.9471/0.9557. F (batch 32): 0.9514/0.9428. Все отклонены.

\- E (layer3 lr 1e-4, 25 эпох): seed 7 0.9614/0.9585 (лучшая эпоха 17), seed 42 0.9642/0.9542 (эпоха 18). TTA8: 0.9685/0.9614 и 0.9642/0.9571. Пайплайн 27.1 мин. Чекпоинты resnet50\_F\_seed7\_layer3\_1e-4.pt, resnet50\_F\_seed42\_layer3\_1e-4.pt.

\- Public E seed 7: 0.94974 (хуже F 0.95215), val-прирост E не подтвердился, различие в пределах шума.



\## Ансамбли (среднее softmax, TTA8, val чистый / искажённый / среднее; public)

\- F7: 0.9642 / 0.9528 / 0.9585; public 0.95215

\- E7: 0.9685 / 0.9614 / 0.9649; public 0.94974

\- F7+F42: 0.9671 / 0.9542 / 0.9607; public 0.95352

\- F7+F42+E7+E42: 0.9742 / 0.9628 / 0.9685; public 0.95559

Четыре модели не укладываются в 45 мин.



\## Решение (финал)

Ансамбль F seed 7 + F seed 42, конфигурация F, 15 эпох на модель, TTA8, среднее softmax.

Время пайплайна 34.5 мин (кэш 37.6 с + 15×62.49 с + 15×62.82 с + TTA 2×75.1 с). Альтернативы: F+E около 43.7 мин (слишком рискованно), E+E около 53 мин (лимит превышен).

Ожидаемый результат: val 0.9671 / 0.9542, public около 0.9535. Лидер public: 0.97.

IV3 отклонён (public 0.95129, порог 0.9535). Финал не меняется: F7+F42, final_solution.ipynb, тег final-ffens.



\## TV1 (train + чистый val, seed 7)

Запрос пользователя. Train 4999 + val 699 = 5698; val id уникальны, пересечения с train нет, метки из val.csv. Test только на TTA8. val\_corrupted.pt не использовался. Веса после эпохи 15, без выбора эпохи.

Эпоха 73.1 с (72.6–74.4), пайплайн 19.8 мин. Проверка на чистом val 0.9928 (запоминание, метрика недействительна). Совпадение с submission\_F\_seed7\_tta.csv: 8032/8299 = 0.9678. Чекпоинт resnet50\_F\_trainval\_seed7.pt, submission\_F\_trainval\_seed7\_tta.csv. cudnn deterministic=True, benchmark=False.

Public (Kaggle): 0.94905 (ниже F seed 7 0.95215 и порога 0.9500). Train+val отклонён, TV2 (seed 42) не запускался. Финал остаётся F7+F42 без val в обучении.

## Раунд 5

* DATA1 counts train: baseball\_diamond:250 basketball\_court:250 bridge:250 church:250 cloud:250 commercial\_area:250 lake:250 medium\_residential:250 overpass:250 palace:249 railway:250 railway\_station:250 rectangular\_farmland:250 roundabout:250 runway:250 sea\_ice:250 snowberg:250 tennis\_court:250 terrace:250 wetland:250. val: baseball\_diamond:35 basketball\_court:35 bridge:34 church:35 cloud:35 commercial\_area:35 lake:35 medium\_residential:35 overpass:35 palace:35 railway:35 railway\_station:35 rectangular\_farmland:35 roundabout:35 runway:35 sea\_ice:35 snowberg:35 tennis\_court:35 terrace:35 wetland:35.
* DATA1 size train: n 4999 | w 256/256/256 | h 256/256/256 | min\_side 256/256/256 | frac<256 0.0000. val: n 699 | w 256/256/256 | h 256/256/256 | min\_side 256/256/256 | frac<256 0.0000. train+val median min\_side 256, frac<256 0.0000.
* DATA1 dups: file md5 train 0 (cross-class 0), train-val 0 (cross-class 0); pixel sha1 train 0/0, train-val 0/0; dHash<=4 train 0/0, train-val 0/0. Список outputs/data1\_duplicates.txt. Test не читался.
* DATA1 F seed 7 на train без аугментации: mismatch 0.0042 (21/4999). Топ-40: outputs/data1\_top40.csv. Сетка: outputs/suspicious\_train.png.
- CX1 исключён: нарушает правила. submission_F7_CX1_tta.csv не используется.
- R320 не запускался: все картинки 256 px

## IV3 (Inception v3, seed 7)

Smoke 2 батча: итерация 0.711 с, оценка эпохи 81.5 с, VRAM 0.97 GB (ниже 100 с и 11 GB). Полный прогон: эпоха 29.7 с (29.0–30.9), VRAM 1124 MB. Лучшая эпоха 14: чистый 0.9442, искажённый 0.9185, score 0.9313. TTA8 0.9557 / 0.9428 / 0.9492. Чекпоинт outputs/checkpoints/inception_v3_seed7.pt. Val в обучение не входил, test только в TTA submission. cudnn deterministic=True, benchmark=False.

Эпохи чистый/искажённый: 01 0.8555/0.8340, 02 0.9027/0.8670, 03 0.9199/0.8813, 04 0.9299/0.8884, 05 0.9199/0.8999, 06 0.9313/0.9027, 07 0.9342/0.9070, 08 0.9328/0.9199, 09 0.9256/0.9142, 10 0.9356/0.9142, 11 0.9356/0.9170, 12 0.9385/0.9170, 13 0.9428/0.9156, 14 0.9442/0.9185, 15 0.9342/0.9127.

Ансамбли TTA8 (чистый | искажённый | среднее): IV3 0.9557 | 0.9428 | 0.9492; F7+IV3 0.9671 | 0.9585 | 0.9628; F7+F42+IV3 0.9700 | 0.9671 | 0.9685. submission_F7_IV3_tta.csv: 8299 строк, совпадение с submission_ens_F7F42_tta.csv 8106/8299 = 0.9767. Пайплайн F7+IV3: кэш 5.7 с + 15×62.49 с + 15×29.7 с + TTA 75.1 с + 86.9 с = 25.8 мин.

Решение: IV3 отклонён. Public F7+IV3 0.95129 ниже порога 0.9535. Финал остаётся ансамбль F seed 7 + F seed 42. Ноутбук final_solution.ipynb и тег final-ffens не меняются.

## Следующие шаги

1\. final\_solution.ipynb собран. Правило среднего по логу раунда 3: seed 7 эпоха 13, seed 42 эпоха 12. Файл resnet50\_F\_seed42.pt — эпоха 11, в ноутбук не подгонялось. Синтаксис ок, smoke 2 батча. Полный прогон не запускался.

2\. Полный прогон вручную: время не более 42 мин. Совпадение с outputs/submission\_ens\_F7F42\_tta.csv может быть ниже 100%: тот файл собран из F42 эпохи 11.

3\. Коммит, тег final.




\## Частые путаницы (F seed 7)

Чистый val: church->commercial\_area 4 (сумма 4), lake<->wetland 2+2 (4), commercial\_area<->palace 0+3 (3), railway<->railway\_station 2+1 (3), church<->palace 2+0 (2).

Искажённый val: church<->commercial\_area 3+1 (4), railway<->railway\_station 2+2 (4), church<->palace 1+2 (3), commercial\_area<->palace 0+3 (3), lake<->wetland 2+1 (3).

