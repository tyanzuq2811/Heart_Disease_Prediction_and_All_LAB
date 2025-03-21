import heapq

def dijkstra(graph, start):
    # Khởi tạo khoảng cách đến các đỉnh là vô hạn
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    
    # Hàng đợi ưu tiên để lưu các đỉnh và khoảng cách
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        print(f"Đang xử lý đỉnh {current_vertex} với khoảng cách {current_distance}")
        
        # Bỏ qua nếu khoảng cách này không còn tối ưu
        if current_distance > distances[current_vertex]:
            continue
        
        # Cập nhật khoảng cách cho các đỉnh kề
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                print(f"Cập nhật khoảng cách đến đỉnh {neighbor}: {distance}")
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

graph = {
    0: {2: 8, 3: 3},
    1: {6: 1},
    2: {0: 8, 5: 8, 4: 7},
    3: {0: 3, 7: 4},
    4: {2: 7, 6: 3},
    5: {2: 8, 7: 8},
    6: {1: 1, 7: 7, 4: 3},
    7: {3: 4, 4: 2, 5: 8, 6: 7}
}

# Gọi hàm Dijkstra
shortest_distances = dijkstra(graph, 0)

# In kết quả
print("Khoảng cách ngắn nhất:", shortest_distances)
