import numpy as np
import json
from typing import List, Tuple, Dict

class FITingTreeTrainer:
    def __init__(self, epsilon: int = 32):
        """
        Инициализирует тренера для FITing-Tree.
        
        Параметры:
        -----------
        epsilon: int
            Максимально допустимая ошибка для линейных моделей.
        """
        self.epsilon = epsilon
        self.segments = []
    
    def train(self, keys: np.ndarray, positions: np.ndarray) -> None:
        """
        Обучает индекс FITing-Tree на основе заданных ключей и их позиций.
        Использует жадный алгоритм для создания кусочно-линейной аппроксимации.
        
        Параметры:
        -----------
        keys: array-like
            Ключи для индексации (должны быть отсортированы).
        positions: array-like
            Позиции каждого ключа в отсортированном массиве.
        """
        self.segments = self._greedy_pla(keys, positions)
    
    def _greedy_pla(self, keys: np.ndarray, positions: np.ndarray) -> List[Dict]:
        """
        Жадный алгоритм для создания кусочно-линейной аппроксимации.
        
        Параметры:
        -----------
        keys: array-like
            Ключи для индексации (должны быть отсортированы).
        positions: array-like
            Позиции каждого ключа в отсортированном массиве.
        
        Возвращает:
        -----------
        List[Dict]
            Список сегментов, каждый представлен словарем с ключами:
            - start_key: первый ключ, охватываемый моделью
            - slope: наклон линейной модели
            - intercept: сдвиг линейной модели
            - start_position: начальная позиция в массиве данных
            - end_position: конечная позиция в массиве данных
        """
        segments = []
        n = len(keys)
        if n == 0:
            return segments
        
        start_idx = 0
        
        while start_idx < n:
            # Начинаем с первой точки текущего сегмента
            start_key = keys[start_idx]
            start_pos = positions[start_idx]
            
            # Пытаемся расширить текущий сегмент как можно дальше
            end_idx = start_idx + 1
            max_error = 0
            
            # Начальные значения для линейной регрессии (метод наименьших квадратов)
            sum_x = keys[start_idx]
            sum_y = positions[start_idx]
            sum_xx = keys[start_idx] * keys[start_idx]
            sum_xy = keys[start_idx] * positions[start_idx]
            count = 1
            
            # Расширяем сегмент, пока ошибка не превысит epsilon
            while end_idx < n:
                # Добавляем новую точку для регрессии
                count += 1
                sum_x += keys[end_idx]
                sum_y += positions[end_idx]
                sum_xx += keys[end_idx] * keys[end_idx]
                sum_xy += keys[end_idx] * positions[end_idx]
                
                # Вычисляем наклон и сдвиг методом наименьших квадратов
                denominator = count * sum_xx - sum_x * sum_x
                if denominator != 0:
                    slope = (count * sum_xy - sum_x * sum_y) / denominator
                    intercept = (sum_y - slope * sum_x) / count
                else:
                    # Вертикальная линия (редкий случай)
                    slope = 0
                    intercept = positions[start_idx]
                
                # Проверяем ошибки для всех точек от start_idx до end_idx
                max_current_error = 0
                for i in range(start_idx, end_idx + 1):
                    predicted_pos = slope * keys[i] + intercept
                    error = abs(predicted_pos - positions[i])
                    max_current_error = max(max_current_error, error)
                
                # Если ошибка превысила epsilon, останавливаемся
                if max_current_error > self.epsilon:
                    break
                
                # Иначе обновляем максимальную ошибку и продолжаем
                max_error = max_current_error
                end_idx += 1
            
            # Если не смогли расширить сегмент, берем только одну точку
            if end_idx == start_idx + 1 and max_error > self.epsilon:
                slope = 0
                intercept = positions[start_idx]
                max_error = 0
                end_idx = start_idx + 1
            
            # Создаем и сохраняем сегмент
            segment = {
                'start_key': int(keys[start_idx]),
                'slope': float(slope),
                'intercept': float(intercept),
                'max_error': int(np.ceil(max_error)),
                'start_position': int(positions[start_idx]),
                'end_position': int(positions[end_idx - 1])
            }
            segments.append(segment)
            
            # Переходим к следующему сегменту
            start_idx = end_idx
        
        return segments
    
    def save_model(self, filename: str) -> None:
        """
        Сохраняет обученную модель FITing-Tree в файл для использования в C++.
        
        Параметры:
        -----------
        filename: str
            Путь к файлу для сохранения модели.
        """
        model_params = {
            'epsilon': self.epsilon,
            'segments': self.segments
        }
        
        with open(filename, 'w') as f:
            json.dump(model_params, f, indent=2)
    
    def predict_position(self, key: int) -> Tuple[int, int]:
        """
        Предсказывает диапазон позиций для ключа.
        
        Параметры:
        -----------
        key: int
            Ключ для поиска.
            
        Возвращает:
        -----------
        Tuple[int, int]
            Кортеж (нижняя_граница, верхняя_граница) для бинарного поиска.
        """
        # Найдем подходящий сегмент (в реальной реализации здесь будет B-дерево)
        segment_idx = -1
        for i, segment in enumerate(self.segments):
            if key >= segment['start_key']:
                segment_idx = i
            else:
                break
        
        if segment_idx == -1:
            # Ключ меньше всех имеющихся ключей
            return (0, 0)
        
        segment = self.segments[segment_idx]
        
        # Предсказываем позицию с помощью линейной модели
        predicted_pos = segment['slope'] * key + segment['intercept']
        predicted_pos = int(round(predicted_pos))
        
        # Вычисляем границы для бинарного поиска на основе максимальной ошибки
        lower_bound = max(segment['start_position'], predicted_pos - segment['max_error'])
        upper_bound = min(segment['end_position'], predicted_pos + segment['max_error'])
        
        return (lower_bound, upper_bound)

# Пример использования
if __name__ == "__main__":
    # Генерируем тестовые данные
    n = 1000000
    keys = np.sort(np.random.randint(0, 1000000000, size=n))
    positions = np.arange(n)
    
    # Обучаем и сохраняем модель FITing-Tree
    trainer = FITingTreeTrainer(epsilon=32)
    trainer.train(keys, positions)
    trainer.save_model("fiting_tree_model.json")
    print(f"FITing-Tree модель обучена и сохранена в fiting_tree_model.json")
    print(f"Количество сегментов: {len(trainer.segments)}")
    
    # Тестируем предсказание
    test_key = keys[500000]
    lower_bound, upper_bound = trainer.predict_position(test_key)
    print(f"Для ключа {test_key} предсказан диапазон: [{lower_bound}, {upper_bound}]")
    print(f"Истинная позиция: {500000}")
    print(f"Размер диапазона: {upper_bound - lower_bound + 1}")
