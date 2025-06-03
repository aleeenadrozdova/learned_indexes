# # train_radix.py
# import argparse
# import numpy as np
# import pickle
# import os

# def build_spline_points(data, error_bound=32):
#     """
#     Build the spline points for the RadixSpline
    
#     Args:
#         data: The sorted array of keys
#         error_bound: The maximum allowed error
    
#     Returns:
#         A list of spline points, where each point is a tuple of (key, position)
#     """
#     spline_points = []
#     positions = np.arange(len(data))
    
#     # Add the first point
#     spline_points.append((data[0], 0))
    
#     current_pos = 0
#     current_key = data[current_pos]
    
#     while current_pos < len(data) - 1:
#         # Find the furthest point that still satisfies the error bound
#         next_pos = current_pos + 1
#         next_key = data[next_pos]
        
#         # Calculate the linear function between current_key and next_key
#         if next_key != current_key:
#             slope = (next_pos - current_pos) / (next_key - current_key)
#             intercept = current_pos - slope * current_key
            
#             # Extend the spline as far as possible within the error bound
#             while next_pos < len(data) - 1:
#                 predicted_pos = slope * next_key + intercept
#                 if abs(predicted_pos - next_pos) > error_bound:
#                     break
                
#                 next_pos += 1
#                 next_key = data[next_pos]
#         else:
#             # If the keys are the same, just move to the next different key
#             while next_pos < len(data) - 1 and data[next_pos] == current_key:
#                 next_pos += 1
#                 next_key = data[next_pos]
        
#         # Add the next spline point
#         spline_points.append((next_key, next_pos))
        
#         current_pos = next_pos
#         current_key = next_key
    
#     return spline_points

# def build_radix_table(spline_points, radix_bits=18):
#     """
#     Build the radix table for the RadixSpline
    
#     Args:
#         spline_points: The spline points (key, position)
#         radix_bits: The number of bits to use for the radix table
    
#     Returns:
#         A dictionary mapping radix keys to spline point indexes
#     """
#     radix_table = {}
    
#     for i, (key, _) in enumerate(spline_points):
#         radix_key = key >> (64 - radix_bits)
        
#         if radix_key not in radix_table:
#             radix_table[radix_key] = i
    
#     return radix_table

# def train_radix_spline(data_file, error_bound=32, radix_bits=18):
#     """
#     Train a RadixSpline model on the provided data
    
#     Args:
#         data_file: Path to the file containing the data (one number per line)
#         error_bound: The maximum allowed error
#         radix_bits: The number of bits to use for the radix table
    
#     Returns:
#         A dictionary containing the trained model
#     """
#     # Load the data
#     with open(data_file, 'r') as f:
#         data = [int(line.strip()) for line in f]
    
#     # Sort the data
#     data.sort()
    
#     # Build the spline points
#     spline_points = build_spline_points(data, error_bound)
    
#     # Build the radix table
#     radix_table = build_radix_table(spline_points, radix_bits)
    
#     return {
#         'spline_points': spline_points,
#         'radix_table': radix_table,
#         'data': data,
#         'error_bound': error_bound,
#         'radix_bits': radix_bits
#     }

# def save_radix_spline(model, output_file):
#     """
#     Save the trained RadixSpline model to a file
    
#     Args:
#         model: The trained RadixSpline model
#         output_file: The path to save the model to
#     """
#     with open(output_file, 'wb') as f:
#         pickle.dump(model, f)

# # Добавить в train_radix.py
# def save_model_for_cpp(model_data, model_file):
#     with open(model_file + ".txt", 'w') as f:
#         # Сохранение параметров
#         f.write(f"{model_data['error_bound']} {model_data['radix_bits']}\n")
        
#         # Сохранение точек сплайна
#         f.write(f"{len(model_data['spline_points'])}\n")
#         for key, position in model_data['spline_points']:
#             f.write(f"{key} {position}\n")
        
#         # Сохранение radix таблицы
#         f.write(f"{len(model_data['radix_table'])}\n")
#         for key, value in model_data['radix_table'].items():
#             f.write(f"{key} {value}\n")


# def main():
#     parser = argparse.ArgumentParser(description='Train a RadixSpline model')
#     parser.add_argument('--data', required=True, help='Path to the data file')
#     parser.add_argument('--output', required=True, help='Path to save the model to')
#     parser.add_argument('--error-bound', type=int, default=32, help='Maximum allowed error')
#     parser.add_argument('--radix-bits', type=int, default=18, help='Number of bits to use for the radix table')
    
#     args = parser.parse_args()
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
#     # Train the model
#     model = train_radix_spline(args.data, args.error_bound, args.radix_bits)
    
#     # Save the model
#     save_radix_spline(model, args.output)
    
#     print(f"Model saved to {args.output}")

# if name == 'main':
#     main()

# src/python/train_radix.py
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_radix_spline(data_file, model_file, error_bound=32, radix_bits=18):
    """
    Обучение RadixSpline модели на предоставленных данных
    """
    # Загрузка данных
    with open(data_file, 'r') as f:
        data = [int(line.strip()) for line in f]
    
    # Сортировка данных
    data.sort()
    
    # Построение точек сплайна
    spline_points = []
    current_pos = 0
    
    while current_pos < len(data):
        spline_points.append((data[current_pos], current_pos))
        
        # Найти наиболее удаленную позицию с ошибкой в пределах допустимой
        next_pos = current_pos + 1
        while next_pos < len(data):
            # Линейная интерполяция между текущей и следующей точкой
            x1, y1 = data[current_pos], current_pos
            x2, y2 = data[next_pos], next_pos
            
            # Проверка ошибок для всех точек между ними
            max_error = 0
            for i in range(current_pos + 1, next_pos):
                x = data[i]
                interpolated_y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                error = abs(interpolated_y - i)
                max_error = max(max_error, error)
            
            if max_error > error_bound:
                next_pos -= 1
                break
            
            next_pos += 1
        
        current_pos = next_pos
    
    # Построение radix-таблицы
    radix_table = {}
    for i in range(len(spline_points) - 1):
        start_key, _ = spline_points[i]
        end_key, _ = spline_points[i+1]
        
        # Вычисление radix-ключа
        radix_key = start_key >> (64 - radix_bits)
        if radix_key not in radix_table:
            radix_table[radix_key] = i
    
    # Сохранение модели
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, 'wb') as f:
        pickle.dump({
            'spline_points': spline_points,
            'radix_table': radix_table,
            'error_bound': error_bound,
            'radix_bits': radix_bits
        }, f)
    save_model_for_cpp({
            'spline_points': spline_points,
            'radix_table': radix_table,
            'error_bound': error_bound,
            'radix_bits': radix_bits
        }, model_file)

# Добавить в train_radix.py
def save_model_for_cpp(model_data, model_file):
    with open(model_file + ".txt", 'w') as f:
        # Сохранение параметров
        f.write(f"{model_data['error_bound']} {model_data['radix_bits']}\n")
        
        # Сохранение точек сплайна
        f.write(f"{len(model_data['spline_points'])}\n")
        for key, position in model_data['spline_points']:
            f.write(f"{key} {position}\n")
        
        # Сохранение radix таблицы
        f.write(f"{len(model_data['radix_table'])}\n")
        for key, value in model_data['radix_table'].items():
            f.write(f"{key} {value}\n")

def main():
    parser = argparse.ArgumentParser(description='Train a RadixSpline model')
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('model_file', type=str, help='Path to output model file')
    parser.add_argument('--error_bound', type=int, default=32, help='Maximum allowed error')
    parser.add_argument('--radix_bits', type=int, default=18, help='Number of bits for radix table')
    
    args = parser.parse_args()
    train_radix_spline(args.data_file, args.model_file, args.error_bound, args.radix_bits)

if __name__ == '__main__':
    main()
