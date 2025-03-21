import numpy as np

data_path = 'nonLinear_data.npy'
data = np.load(data_path, allow_pickle=True).item()
X, y = data['X'], data['labels']
print(X.shape, y.shape)

split_ratio = 0.8
random_state = 42
np.random.seed(random_state)
indices = np.random.permutation(len(X))
split_index = int(len(X) * split_ratio)
train_indices, test_indices = indices[:split_index], indices[split_index:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# print("5 dòng đầu tiên của tập huấn luyện:")
# print(X_train[:5], y_train[:5])
# print("5 dòng đầu tiên của tập kiểm tra:")
# print(X_test[:5], y_test[:5])

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1   
        self.a1 = self.relu(self.z1)        
        self.z2 = np.dot(self.a1, self.W2) + self.b2  
        return self.softmax(self.z2)          

    def backward(self, X, y):
        m = X.shape[0]  
        output = self.forward(X)
        error = output - y  
        
        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m
        
        hidden_error = np.dot(error, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if epoch % 10 == 0:
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))  # Cross-entropy loss
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

y_one_hot = np.eye(len(set(y.flatten())))[y.flatten()]

def evaluate_model(hidden_size, learning_rate, epochs=100):
    print(f"\nThử nghiệm: hidden_size={hidden_size}, learning_rate={learning_rate}")
    nn = NeuralNetwork(input_size=X.shape[1],
                       hidden_size=hidden_size,
                       output_size=y_one_hot.shape[1],
                       learning_rate=learning_rate)
    nn.train(X_train, y_one_hot[train_indices], epochs=epochs)
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

# Thử các cấu hình khác nhau
evaluate_model(hidden_size=4, learning_rate=0.1)   
evaluate_model(hidden_size=8, learning_rate=0.1) 
evaluate_model(hidden_size=4, learning_rate=0.01)
evaluate_model(hidden_size=16, learning_rate=0.05) 

# def evaluate_activation(activation):
#     print(f"\nThử nghiệm với hàm kích hoạt: {activation}")
#     nn = NeuralNetwork(input_size=X.shape[1],
#                        hidden_size=4,
#                        output_size=y_one_hot.shape[1],
#                        learning_rate=0.1,
#                        activation=activation)
#     nn.train(X_train, y_one_hot[train_indices], epochs=100)
#     y_pred = nn.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

# Chạy thử nghiệm
# evaluate_activation('relu')
# evaluate_activation('sigmoid')
# evaluate_activation('tanh')
# evaluate_activation('leaky_relu')