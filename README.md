# Satellite Image Classification (20 classes)

Учебное соревнование: классификация спутниковых снимков с устойчивостью к артефактам (blur, инверсия цвета, низкое разрешение). Метрика — Accuracy на leaderboard (private = 30% теста).

## Классы
| id | класс | id | класс |
|---|---|---|---|
| 0 | baseball_diamond | 10 | railway |
| 1 | basketball_court | 11 | railway_station |
| 2 | bridge | 12 | rectangular_farmland |
| 3 | church | 13 | roundabout |
| 4 | cloud | 14 | runway |
| 5 | commercial_area | 15 | sea_ice |
| 6 | lake | 16 | snowberg |
| 7 | medium_residential | 17 | tennis_court |
| 8 | overpass | 18 | terrace |
| 9 | palace | 19 | wetland |

## Данные
На класс: ~250 train, ~35 val, ~415 test (test делится на public 70% / private 30%). Путь к данным указан в `src/config.py`. Папка `data/` не коммитится.

## Правила и требования
- Разрешены только предобученные AlexNet, Inception, VGG, ResNet.
- Accuracy на тесте должна быть выше 65%.
- Решение воспроизводимо и уникально, без копирования чужого кода.
- Загрузить решение на Moodle до 14 октября, 14:00.
- Итоговые места публикуют 16 октября с 09:00.
- Оценка: место в private leaderboard, checkpoint-оценка и финальная презентация команды.

## Формат submission
id,label
12.jpg,0
123.jpg,5

text
Одна строка на каждое изображение теста, `label` — целое 0-19. Имя файла: `Name_Surname.csv`.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # torch/torchvision ставить под свой CUDA
```

## Запуск
```bash
Откройте `Week5_baseline.ipynb` (или итоговый блокнот), задайте `CONTEST_DATA`, выполните все ячейки сверху вниз. Результат — `submission.csv`.
       --out outputs/submissions/Ivan_Sokolov.csv            # submission
```

## Подход
1. Transfer learning с ImageNet-весов (ResNet как основной кандидат).
2. Аугментации из `torchvision.transforms.v2`, имитирующие дефекты теста (blur, invert, downscale, color jitter, повороты).
3. AdamW, cosine LR, label smoothing, AMP.
4. TTA на инференсе, при необходимости ансамбль разрешённых моделей.
5. Выбор модели только по val.

## Эксперименты
Все запуски записаны в `outputs/experiments.csv`: модель, гиперпараметры, seed, val acc, дата. Текущее состояние проекта — `docs/STATE.md`.

## Воспроизводимость
Фиксированные seed, версии пакетов в `requirements.txt`, один entry point для submission (`src.predict`). Финальный чекпоинт и команда запуска записаны в `docs/STATE.md`.

## Структура
См. `AGENTS.md`, раздел 5.
