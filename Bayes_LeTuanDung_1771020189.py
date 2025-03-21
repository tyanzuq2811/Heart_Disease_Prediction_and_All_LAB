import numpy as np

# Create training data
def create_train_data():
    data = [['Sunny','Hot', 'High', 'Weak', 'no'],
            ['Sunny','Hot', 'High', 'Strong', 'no'],
            ['Overcast','Hot', 'High', 'Weak', 'yes'],
            ['Rain','Mild', 'High', 'Weak', 'yes'],
            ['Rain','Cool', 'Normal', 'Weak', 'yes'],
            ['Rain','Cool', 'Normal', 'Strong', 'no'],
            ['Overcast','Cool', 'Normal', 'Strong', 'yes'],
            ['Overcast','Mild', 'High', 'Weak', 'no'],
            ['Sunny','Cool', 'Normal', 'Weak', 'yes'],
            ['Rain','Mild', 'Normal', 'Weak', 'yes']]
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

train_data = create_train_data()
# prior_probablity = compute_prior_probability(train_data)
# print("P(“PlayTennis”=no)", prior_probablity[0])
# print("P(“PlayTennis”=yes)", prior_probablity[1])

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

# conditional_probability, list_x_name = compute_conditional_probability(train_data)
# print("x1 =",list_x_name[0])
# print("x2 = ",list_x_name[1])
# print("x3 = ",list_x_name[2])
# print("x4 = ",list_x_name[3])

# Hàm lấy chỉ số từ giá trị
def get_index_from_value(feature_name, list_features):
    return np.where(list_features == feature_name)[0][0]

# conditional_probability, list_x_name = compute_conditional_probability(train_data)
# outlook = list_x_name[0]
# i1 = get_index_from_value("Overcast", outlook)
# i2 = get_index_from_value("Rain", outlook)
# i3 = get_index_from_value("Sunny", outlook)
# print(i1, i2, i3)

conditional_probability, list_x_name = compute_conditional_probability(train_data)
x1=get_index_from_value("Sunny",list_x_name[0])
print("P('Outlook'='Sunny'|Play Tennis'='Yes') = ",
np.round(conditional_probability[0][1, x1],2))
x1=get_index_from_value("Sunny",list_x_name[0])
print("P('Outlook'='Sunny'|Play Tennis'='No') = ",
np.round(conditional_probability[0][0, x1],2))

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
X = ['Sunny', 'Cool', 'Normal', 'Weak']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
pred = prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)

if pred:
    print("A should go!")
else:
    print("A should not go!")

