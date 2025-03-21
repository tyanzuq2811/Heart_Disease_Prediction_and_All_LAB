import numpy as np
import matplotlib.pyplot as plt

# Hàm tạo một nhiễm sắc thể ngẫu nhiên gồm 54 gene, mỗi gene có giá trị từ 0 đến 3, khởi tạo quần thể
def create_chromosome():
    return np.random.randint(0, 4, size=54)
#Một chromosome là một chuỗi hướng dẫn cho robot.

# Hàm tạo một quần thể gồm pop_size nhiễm sắc thể
def create_population(pop_size):
    return np.array([create_chromosome() for _ in range(pop_size)])
#Tạo nhiều cá thể ban đầu để thuật toán tiến hóa.

# Hàm đánh giá độ thích nghi của quần thể
def evaluate_population(population, room):
    fitness = []
    for chromosome in population:
        efficiency, _ = painter_play(chromosome, room)
        fitness.append(efficiency)
    return np.array(fitness)

# Hàm chọn lọc cha mẹ dựa trên độ thích nghi (Roulette Wheel Selection)
def select_parents(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        idx = np.random.choice(np.arange(len(population)), p=fitness/fitness.sum())
        parents[i, :] = population[idx, :]
    return parents
#Cá thể có fitness cao hơn sẽ có cơ hội được chọn làm cha mẹ nhiều hơn

# Hàm lai ghép hai cha mẹ để tạo ra thế hệ con mới
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.random.randint(1, offspring_size[1])
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
#Con cái được tạo ra bằng cách kết hợp gen từ cha mẹ.

# Hàm đột biến để duy trì sự đa dạng trong quần thể
def mutate(offspring, mutation_rate):
    for idx in range(offspring.shape[0]):
        for gene in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[idx, gene] = np.random.randint(0, 4)
    return offspring
# Xác xuất nhỏ tạo ra gen đột biến NN, Tạo đột biến để giữ sự đa dạng quần thể, tránh bị mắc kẹt trong tối ưu cục bộ.

# Hàm mô phỏng robot sơn phòng dựa trên nhiễm sắc thể
def painter_play(chromosome, room):
    rows, cols = room.shape
    x, y = np.random.randint(0, rows), np.random.randint(0, cols)
    direction = np.random.randint(0, 4)  # 0: lên, 1: phải, 2: xuống, 3: trái
    painted = np.zeros_like(room)
    steps = 0
    max_steps = 1000  
    
    while steps < max_steps:
        if room[x, y] == 0:
            painted[x, y] = 1
        
        # Xác định trạng thái hiện tại của robot
        state = (27 * int(painted[x, y]) + 9 * int(room[(x - 1) % rows, y]) + 
                 3 * int(room[x, (y - 1) % cols]) + int(room[x, (y + 1) % cols])) % 54
        action = chromosome[state]

        # Thực hiện hành động tương ứng
        if action == 1:
            direction = (direction - 1) % 4  # Quay trái
        elif action == 2:
            direction = (direction + 1) % 4  # Quay phải
        elif action == 3:
            direction = np.random.randint(0, 4)  # Quay ngẫu nhiên
        
        # Di chuyển robot theo hướng hiện tại
        if direction == 0:
            x = (x - 1) % rows
        elif direction == 1:
            y = (y + 1) % cols
        elif direction == 2:
            x = (x + 1) % rows
        elif direction == 3:
            y = (y - 1) % cols
        
        steps += 1
    
    efficiency = np.sum(painted) / np.sum(room == 0)
    return efficiency, painted
#Robot di chuyển, sơn phòng, và báo cáo hiệu suất.

# Hàm chạy thuật toán di truyền
def genetic_algorithm(room, pop_size=50, num_generations=200, mutation_rate=0.002):
    population = create_population(pop_size)
    best_fitness = []
    for generation in range(num_generations):
        fitness = evaluate_population(population, room)
        best_fitness.append(np.max(fitness))
        parents = select_parents(population, fitness, pop_size // 2)
        offspring = crossover(parents, (pop_size - parents.shape[0], population.shape[1]))
        offspring = mutate(offspring, mutation_rate)
        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring
    return population, best_fitness
#Lặp lại các bước của GA và theo dõi fitness.

# Tạo một căn phòng trống
room = np.zeros((20, 40))

# Chạy thuật toán di truyền
population, best_fitness = genetic_algorithm(room)

# Vẽ đồ thị độ thích nghi qua các thế hệ
plt.plot(best_fitness)
plt.xlabel('Thế hệ')
plt.ylabel('Độ thích nghi tốt nhất')
plt.title('Sự thay đổi độ thích nghi qua các thế hệ')
plt.show()

# Kiểm tra nhiễm sắc thể tốt nhất
best_chromosome = population[np.argmax(evaluate_population(population, room))]
efficiency, painted = painter_play(best_chromosome, room)
print(f"Độ hiệu quả của nhiễm sắc thể tốt nhất: {efficiency}")

# Hiển thị phòng sau khi được sơn
plt.imshow(painted, cmap='gray')
plt.title('Phòng đã được sơn bởi robot')
plt.show()
