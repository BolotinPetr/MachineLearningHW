import numpy as np


class Vertex(object):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.data = np.column_stack((features, target))
        self.left = None
        self.right = None
        self.value = None
        self.feature = 1
        self.threshold = 1

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_feature(self, feature):
        self.feature = feature

    def split(self):
        def H(data):
            return np.std(data[:, -1])
        max_L = self.data
        max_R = self.data
        for feature in range(0, self.data.shape[1] - 1, 1):
            left = self.features[0, feature]
            G = 10000
            for threshold in np.sort(self.features[:, feature]):
                right = threshold
                if left != right:
                    current_threshold = (left + right) / 2
                    left = threshold
                else:
                    current_threshold = left
                L = self.data[self.data[:, feature] <= current_threshold]
                R = self.data[self.data[:, feature] > current_threshold]
                new_G = (float(L.shape[0]) / float(self.data.shape[0])) * H(L)\
                        + (float(R.shape[0]) / float(self.data.shape[0])) * H(R)
                if new_G < G:
                    G = new_G
                    self.set_feature(feature)
                    self.set_threshold(current_threshold)
                    max_L = L
                    max_R = R
        #print max_L.shape, self.threshold
        return max_L, max_R

    def calc_value(self):
        self.value = self.target.mean()


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, data, target):
        def create_vertex(vertex_features, vertex_target, current_depth):
            vertex = Vertex(vertex_features, vertex_target)
            vertex.calc_value()
            if current_depth == 0:
                #print vertex.value
                return vertex
            else:
                left_data, right_data = vertex.split()
                #print left_data[:, -1].shape
                vertex.set_left(create_vertex(left_data[:, :-1], left_data[:, -1], current_depth - 1))
                vertex.set_right(create_vertex(right_data[:, :-1], right_data[:, -1], current_depth - 1))
            return vertex
        self.top = create_vertex(data, target, self.max_depth)
        #print self.top.left.value

    def predict(self, X):
        a = []
        for i in range(0, X.shape[0], 1):
            a.append(self._predict_for_one(X[i]))
        return a

    def _predict_for_one(self, sample):
        def predict_sample(sample, vertex):
            def next_vertex(sample, vertex):
                if sample[vertex.feature] <= vertex.threshold:
                    return vertex.left
                else:
                    return vertex.right

            if vertex.left is None or vertex.right is None:
                return vertex.value
            else:
                return predict_sample(sample, next_vertex(sample, vertex))
        current_vertex = self.top
        return predict_sample(sample, current_vertex)
