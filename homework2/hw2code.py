import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    
    sorted_indices = np.argsort(feature_vector)
    feat_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]
    
    valid_split_mask = feat_sorted[:-1] != feat_sorted[1:]
    
    if not np.any(valid_split_mask):
        return np.array([]), np.array([]), None, None

    thresholds = (feat_sorted[:-1] + feat_sorted[1:]) / 2.0

    N = len(target_vector)
    
    target_cumsum = np.cumsum(target_sorted)
    
    r_l_sizes = np.arange(1, N)
    r_r_sizes = N - r_l_sizes
    
    s_l = target_cumsum[:-1]
    s_r = target_cumsum[-1] - s_l
    
    p1_l = s_l / r_l_sizes
    p0_l = 1 - p1_l
    
    p1_r = s_r / r_r_sizes
    p0_r = 1 - p1_r
    
    h_l = 1 - (p1_l ** 2 + p0_l ** 2)
    h_r = 1 - (p1_r ** 2 + p0_r ** 2)
    
    Q = - (r_l_sizes / N) * h_l - (r_r_sizes / N) * h_r
    
    thresholds = thresholds[valid_split_mask]
    ginis = Q[valid_split_mask]
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) or \
           np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            
            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature]
                counts = Counter(feature_vector)
                clicks = Counter(feature_vector[sub_y == 1])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count if current_count > 0 else 0
                
                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in feature_vector])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if gini is None:
                continue
                
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                
                if feature_type == "real":
                    threshold_best = threshold
                    split = sub_X[:, feature].astype(float) < threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat in categories_map if categories_map[cat] < threshold]
                    split = np.array([x in threshold_best for x in sub_X[:, feature]])

        if feature_best is None or (self._min_samples_leaf is not None and 
                                   (np.sum(split) < self._min_samples_leaf or 
                                    np.sum(~split) < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        
        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)