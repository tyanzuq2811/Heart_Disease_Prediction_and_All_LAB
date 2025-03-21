import numpy as np

def create_train_data():
    data = [['AI', 'Afternoon', 'Healthy', 'Mandatory', '3', 'High', '2', 'Yes'],
            ['Probability and Statistics', 'Afternoon', 'Tired', 'Optional', '4', 'Intermediate', '3', 'Yes'],
            ['English', 'Morning', 'Healthy', 'Optional', '2', 'Beginner', '1', 'No'],
            ['DBMS', 'Afternoon', 'Sick', 'Mandatory', '3', 'Advanced', '2', 'Yes'],
            ['Math', 'Afternoon', 'Healthy', 'Optional', '4', 'Advanced', '3', 'Yes'],
            ['Computer Networks', 'Afternoon', 'Healthy', 'Mandatory', '3', 'Intermediate', '2', 'No'],
            ['Traditional martial arts', 'Afternoon', 'Sick', 'Optional', '1', 'Beginner', '1', 'Yes']]
    return np.array(data)


# 2.2 Tính toán xác suất tiên nghiệm (Prior Probability)
def compute_prior_probability(train_data):
    y_unique = ['no', 'yes']

    prior_probability = np.zeros(len(y_unique))
    total_samples = train_data.shape[0]
    
    # Count the occurrences of each class
    for i, label in enumerate(y_unique):
        prior_probability[i] = np.sum(train_data[:, -1] == label) / total_samples
        
    return prior_probability

# 2.3 Tính toán xác suất có điều kiện (Conditional Probability)
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        # Initialize the conditional probability matrix for each feature
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        
        for j, label in enumerate(y_unique):
            for k, x_value in enumerate(x_unique):
                # Count occurrences of X=x_value for class y
                x_conditional_probability[j, k] = np.sum((train_data[:, i] == x_value) & (train_data[:, -1] == label)) / np.sum(train_data[:, -1] == label)
        
        conditional_probability.append(x_conditional_probability)
    
    return conditional_probability, list_x_name

# Hàm lấy chỉ số từ giá trị
def get_index_from_value(feature_name, list_features):
    return np.where(list_features == feature_name)[0][0]

# 2.4 Huấn luyện mô hình Naive Bayes
def train_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probability(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(train_data)
    
    return prior_probability, conditional_probability, list_x_name

# 2.5 Dự đoán kết quả chơi tennis
def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):
    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    # Tính toán xác suất Posterior cho cả hai lớp 'no' và 'yes'
    p0 = prior_probability[0] * np.prod([conditional_probability[i][0, j] for i, j in enumerate([x1, x2, x3, x4])])
    p1 = prior_probability[1] * np.prod([conditional_probability[i][1, j] for i, j in enumerate([x1, x2, x3, x4])])

    # Dự đoán lớp có xác suất cao hơn
    if p0 > p1:
        return 0  # 'no'
    else:
        return 1  # 'yes'

# 2.5 Dự đoán
X = ['AI', 'Morning', 'Healthy', 'Online']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
pred = prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)

if pred:
    print("The student should attend this class!")
else:
    print("The student should not attend this class!")
