def get_k_nearest_neighbours(training_points, test_point, k):
    """Returns k nearest neighbours of test_point, among training_points"""
    distances = []
    
    dimensions = len(test_point) - 1  # -1 to not consider class label when calculating distance
    for point in training_points: 
        distance = euclidean_dist(point, test_point, dimensions) 
        distances.append((distance, point)) # append as tuple: (distance, point) 
        
    distances.sort(key=lambda x: x[0]) # sort the points by euclidean_distance
    
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i])
    
    return neighbours


def get_k_nearest_neighbours_np_version1(training_points, test_point, k):
    """Returns k nearest neighbours of test_point, among training_points"""
    dimensions = len(test_point) - 1  # -1 to not consider class label when calculating distance

    distances = np.array(list(zip(np.apply_along_axis(euclidean_dist, 1, training_points, test_point, dimensions), training_points)))
    distances = distances[distances[:,0].argsort()]
    
    neighbours = distances[:k,]

    return neighbours
