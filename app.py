from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import logging
from math import exp
import os

# Создаем Flask приложение
app = Flask(__name__)

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка всех необходимых объектов
def load_models():
    try:
        # Проверяем существование файлов
        model_files = [
            'models/best_stacking_model.pkl',
            'models/scaler.pkl', 
            'models/model_metadata.pkl'
        ]
        
        for file in model_files:
            if not os.path.exists(file):
                logger.error(f"Файл не найден: {file}")
                return None, None, None
        
        # Загрузка модели
        with open('models/best_stacking_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Загрузка scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Загрузка метаданных
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info("Все модели успешно загружены")
        return model, scaler, metadata
        
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        return None, None, None

# Загружаем модели
model, scaler, metadata = load_models()

# Извлекаем информацию из метаданных
if metadata:
    FEATURE_ORDER = metadata.get('feature_order', [])
    STATUS_MAPPING = metadata.get('status_mapping', {})
    STATE_FREQ = metadata.get('state_freq', {})
else:
    FEATURE_ORDER = []
    STATUS_MAPPING = {}
    STATE_FREQ = {}

@app.route('/')
def home():
    """Главная страница с формой ввода"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Обработка предсказания через форму"""
    try:
        if model is None or scaler is None:
            return render_template('error.html', error="Модель не загружена")
        
        # Получаем данные из формы
        form_data = request.form
        
        # Преобразуем в числовые значения
        input_data = {
            'beds_cleaned': float(form_data.get('beds', 3)),
            'baths_cleaned': float(form_data.get('baths', 2)),
            'max_rating': float(form_data.get('max_rating', 7)),
            'min_rating': float(form_data.get('min_rating', 3)),
            'max_distance': float(form_data.get('max_distance', 2)),
            'min_distance': float(form_data.get('min_distance', 1)),
            'has_full_k12': int(form_data.get('has_full_k12', 1)),
            'has_private_school': int(form_data.get('has_private_school', 0)),
            'building_age': float(form_data.get('building_age', 30)),
            'was_remodeled': int(form_data.get('was_remodeled', 0)),
            'has_heating': int(form_data.get('has_heating', 1)),
            'has_cooling': int(form_data.get('has_cooling', 1)),
            'has_parking': int(form_data.get('has_parking', 1)),
            'stories_clean': float(form_data.get('stories', 1)),
            'log_sqft_cleaned': np.log(float(form_data.get('sqft', 1800))),
            'log_lotsize_cleaned': np.log(max(1, float(form_data.get('lotsize', 7000)))),
            'is_foreclosure': int(form_data.get('is_foreclosure', 0)),
            'is_multi_family': int(form_data.get('is_multi_family', 0)),
            'status_group_encoded': STATUS_MAPPING.get(form_data.get('status', 'active'), 0),
            'state_freq_encoded': STATE_FREQ.get(form_data.get('state', 'fl'), 0.334)
        }
        
        # Создаем DataFrame в правильном порядке признаков
        features_df = pd.DataFrame([input_data])[FEATURE_ORDER]
        
        # Масштабируем признаки
        scaled_features = scaler.transform(features_df)
        
        # Предсказание
        prediction_log = model.predict(scaled_features)[0]
        
        # Преобразуем обратно из логарифма
        prediction = np.exp(prediction_log)
        
        # Округляем до тысяч
        prediction_rounded = round(prediction / 1000) * 1000
        
        return render_template('result.html', 
                             prediction=prediction_rounded,
                             input_data=form_data)
                             
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint для предсказания"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Модель не загружена', 'success': False}), 500
        
        data = request.get_json()
        
        # Проверяем наличие данных
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
        
        # Преобразуем данные
        input_data = {
            'beds_cleaned': float(data.get('beds', 3)),
            'baths_cleaned': float(data.get('baths', 2)),
            'max_rating': float(data.get('max_rating', 7)),
            'min_rating': float(data.get('min_rating', 3)),
            'max_distance': float(data.get('max_distance', 2)),
            'min_distance': float(data.get('min_distance', 1)),
            'has_full_k12': int(data.get('has_full_k12', 1)),
            'has_private_school': int(data.get('has_private_school', 0)),
            'building_age': float(data.get('building_age', 30)),
            'was_remodeled': int(data.get('was_remodeled', 0)),
            'has_heating': int(data.get('has_heating', 1)),
            'has_cooling': int(data.get('has_cooling', 1)),
            'has_parking': int(data.get('has_parking', 1)),
            'stories_clean': float(data.get('stories', 1)),
            'log_sqft_cleaned': np.log(float(data.get('sqft', 1800))),
            'log_lotsize_cleaned': np.log(max(1, float(data.get('lotsize', 7000)))),
            'is_foreclosure': int(data.get('is_foreclosure', 0)),
            'is_multi_family': int(data.get('is_multi_family', 0)),
            'status_group_encoded': STATUS_MAPPING.get(data.get('status', 'active'), 0),
            'state_freq_encoded': STATE_FREQ.get(data.get('state', 'fl'), 0.334)
        }
        
        # Создаем DataFrame
        features_df = pd.DataFrame([input_data])[FEATURE_ORDER]
        
        # Масштабируем
        scaled_features = scaler.transform(features_df)
        
        # Предсказание
        prediction_log = model.predict(scaled_features)[0]
        prediction = np.exp(prediction_log)
        prediction_rounded = round(prediction / 1000) * 1000
        
        return jsonify({
            'prediction': prediction_rounded,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    print("🚀 Сервис предсказания стоимости недвижимости запущен!")
    print("📊 Эндпоинты:")
    print("   GET  /              - Веб-интерфейс")
    print("   POST /predict       - Предсказание через форму")
    print("   POST /api/predict   - JSON API")
    print("   GET  /health        - Проверка здоровья")
    print(f"🔮 Модель загружена: {model is not None}")
    print(f"📏 Scaler загружен: {scaler is not None}")
    print("🌐 Откройте: http://127.0.0.1:5001")
    print("-" * 50)
    app.run(debug=True, host='127.0.0.1', port=5001)