# Построение адаптивной индексации с помощью методов машинного обучения


## Реализованные структуры данных

### Классические методы
- **B-Tree** (`btree.h`)
  - Шаблонный класс с настраиваемым порядком дерева
  - Поддержка поиска, вставки, удаления и диапазонных запросов
  - Оценка потребления памяти

- **B+-Tree** (`bplus_tree.h`) 
  - Модификация B-Tree с данными только в листьях
  - Связанные листовые узлы для эффективного диапазонного поиска
  - Улучшенный последовательный доступ

### Адаптивные методы на основе ML

- **RMI** (`rmi.h` + `train_rmi.py`)
  - Двухуровневая архитектура с линейными моделями
  - Python-скрипт для обучения моделей
  - JSON-парсер для загрузки параметров

- **FITing-Tree** (`fiting_tree.h`)
  - Кусочно-линейная аппроксимация с параметром epsilon
  - B-дерево для индексации сегментов
  - Буферы для инкрементальных вставок

- **RadixSpline** (`radix_spline.h`)
  - Гибрид радикс-таблицы и сплайн-моделей
  - Быстрый поиск через радикс-индексацию
  - Линейная интерполяция между точками сплайна

## Система тестирования

### Генерация данных (`data_generator.h`)
- **Равномерное распределение** - базовый случай
- **Нормальное распределение** - реальные данные  
- **Логнормальное распределение** - финансовые данные
- **Распределение Зипфа** - веб-данные, социальные сети

### Бенчмарки (`benchmark.h`)
- **Время построения индекса**
- **Потребление памяти** 
- **Производительность точечного поиска**
- **Производительность диапазонного поиска**


### Интеграция с Python
- Обучение ML-моделей в Python (scikit-learn)
- Экспорт параметров в JSON
- Загрузка и использование в C++

### Результаты CSV + визуализация
- Автоматическая запись метрик в CSV
- Сравнительные графики производительности
- Анализ масштабируемости
