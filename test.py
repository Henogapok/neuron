import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# Сигмоида - функция активации
def activate_sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная сигмоиды
def proizv_sigmoid(x):
    return x * (1 - x)


# Инициализация весов
def initialize_weights(inp_size, hid_size, out_size):
    W1 = np.random.randn(inp_size, hid_size)
    W2 = np.random.randn(hid_size, out_size)
    return W1, W2


# Прямое распространение
def forward_prop(X, W1, W2):
    a = np.dot(X, W1)  # Прямое распространение
    acitve1 = activate_sigmoid(a)  # Функция активации
    b = np.dot(acitve1, W2)  # Прямое распространение
    acitve2 = activate_sigmoid(b)  # Функция активации
    return a, acitve1, b, acitve2


# Обратное распространение ошибки
def backward_prop(X, y, W1, W2, Z1, A1, A2, learning_rate):
    # Ошибка выходного слоя
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)

    # Ошибка скрытого слоя
    dZ1 = np.dot(dZ2, W2.T) * proizv_sigmoid(A1)
    dW1 = np.dot(X.T, dZ1)

    # Обновление весов
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2

    return W1, W2


# Обучение перцептрона
def train_perceptron(X, y, hid_size, epochs, learning_rate):
    input_size = X.shape[1]  # Количество входных признаков
    output_size = y.shape[1]  # Количество выходных классов

    # Инициализация весов
    W1, W2 = initialize_weights(input_size, hidden_size, output_size)

    # Обучение модели
    for epoch in range(epochs):
        # Прямое распространение
        Z1, A1, Z2, A2 = forward_prop(X, W1, W2)

        # Обратное распространение
        W1, W2 = backward_prop(X, y, W1, W2, Z1, A1, A2, learning_rate)

        # Периодически выводим ошибку
    final_loss = np.mean(np.square(y - A2))  # Среднеквадратичная ошибка
    print(f"Ошибка: {final_loss:.4f}")

    return W1, W2


# Прогноз на основе обученных весов
def predict(X, W1, W2):
    _, _, _, A2 = forward_prop(X, W1, W2)
    return np.argmax(A2, axis=1)


# Загрузка данных Iris
iris_data = load_iris()
X = iris_data['data']
y = iris_data['target']

# Преобразуем целевые значения в категориальные (one-hot encoding)
y = LabelBinarizer().fit_transform(y)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем перцептрон
hidden_size = 10  # Количество нейронов в скрытом слое
epochs = 1000  # Количество эпох
learning_rate = 0.01  # Скорость обучения

W1, W2 = train_perceptron(X_train, y_train, hidden_size, epochs, learning_rate)

# Прогнозируем на тестовой выборке
y_pred = predict(X_test, W1, W2)
y_test_labels = np.argmax(y_test, axis=1)

# Оцениваем точность
accuracy = np.mean(y_pred == y_test_labels)
print(f"Точность на тестовых данных: {accuracy * 100:.2f}%")
