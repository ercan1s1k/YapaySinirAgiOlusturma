import numpy as np  # NumPy kütüphanesini içe aktarıyoruz, bilimsel hesaplamalar için kullanacağız.

# Girdi ve çıktı verileri
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR problemi için giriş verileri
y = np.array([[0], [1], [1], [0]])  # XOR problemi için çıkış verileri

# Ağırlıklar ve önyargılar (bias)
weights_input_hidden = np.random.rand(2, 2)  # 2 giriş, 2 gizli katman nöronu
weights_hidden_output = np.random.rand(2, 1)  # 2 gizli nöron, 1 çıkış nöronu
bias_hidden = np.random.rand(1, 2)  # Gizli katman için önyargılar
bias_output = np.random.rand(1, 1)  # Çıkış katmanı için önyargılar

def sigmoid(x):
    """Sigmoid aktivasyon fonksiyonu."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid fonksiyonunun türevi, geri yayılım için kullanılır."""
    return x * (1 - x)

def forward_propagation(X):
    """İleri besleme işlemini gerçekleştirir."""
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden  # Gizli katman girdisi
    hidden_layer_output = sigmoid(hidden_layer_input)  # Gizli katman çıktısı
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output  # Çıkış katmanı girdisi
    output_layer_output = sigmoid(output_layer_input)  # Çıkış katmanı çıktısı
    
    return hidden_layer_output, output_layer_output

def backward_propagation(X, y, hidden_layer_output, output_layer_output):
    """Geri yayılım işlemini gerçekleştirir ve ağırlıkları günceller."""
    output_layer_error = y - output_layer_output  # Çıkış hatası
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)  # Çıkış katmanı hatası güncellemesi
    
    hidden_layer_error = output_layer_delta.dot(weights_hidden_output.T)  # Gizli katman hatası
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)  # Gizli katman hatası güncellemesi
    
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    weights_hidden_output += hidden_layer_output.T.dot(output_layer_delta)  # Çıkış katmanı ağırlıklarını güncelle
    weights_input_hidden += X.T.dot(hidden_layer_delta)  # Girdi-gizli katman ağırlıklarını güncelle
    bias_output += np.sum(output_layer_delta, axis=0, keepdims=True)  # Çıkış katmanı önyargı güncellemesi
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True)  # Gizli katman önyargı güncellemesi

def train(X, y, epochs=10000):
    """Yapay sinir ağını eğitir."""
    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(X)  # İleri besleme
        backward_propagation(X, y, hidden_layer_output, output_layer_output)  # Geri yayılım

train(X, y)  # Eğitimi başlat

def predict(X):
    """Eğitilmiş ağ ile tahmin yapar."""
    _, output_layer_output = forward_propagation(X)
    return output_layer_output

# Eğitim sonrası tahminler
predictions = predict(X)
print(predictions)  # XOR probleminin tahmin sonuçları
