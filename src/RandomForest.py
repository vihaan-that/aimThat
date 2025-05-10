import csv
import math
import random
import sys
import pickle
from collections import Counter, defaultdict

# Set random seed for reproducibility
random.seed(42)

# Progress Bar Function
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Display a progress bar in the console.

    Parameters:
    - iteration: Current iteration (int)
    - total: Total iterations (int)
    - prefix: Prefix string (str)
    - suffix: Suffix string (str)
    - decimals: Positive number of decimals in percent complete (int)
    - length: Character length of bar (int)
    - fill: Bar fill character (str)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print a new line when complete
    if iteration == total:
        print()

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")
# Load the dataset without pandas
def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]
    data = data[1:]  # Exclude header
    # Convert strings to float
    for i in range(len(data)):
        data[i] = [float(x) for x in data[i]]
    return header, data

# Extract features and labels
def extract_features_labels(data, feature_indices, label_index):
    X = []
    y = []
    for row in data:
        X.append([row[i] for i in feature_indices])
        y.append(int(row[label_index]))
    return X, y

# Standardize features
def standardize_features(X):
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features

    # Calculate means
    for i in range(n_features):
        means[i] = sum([X[j][i] for j in range(n_samples)]) / n_samples

    # Calculate standard deviations
    for i in range(n_features):
        stds[i] = math.sqrt(sum([(X[j][i] - means[i]) ** 2 for j in range(n_samples)]) / n_samples)

    # Standardize
    for i in range(n_samples):
        for j in range(n_features):
            if stds[j] != 0:
                X[i][j] = (X[i][j] - means[j]) / stds[j]
            else:
                X[i][j] = 0.0
    return X

# Train/Test split
def train_test_split(X, y, test_size=0.2, stratify=False):
    data = list(zip(X, y))
    if stratify:
        # Stratified split
        label_to_data = defaultdict(list)
        for x_i, y_i in data:
            label_to_data[y_i].append((x_i, y_i))
        train_data = []
        test_data = []
        for label, items in label_to_data.items():
            split_idx = int(len(items) * (1 - test_size))
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])
    else:
        random.shuffle(data)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    return list(X_train), list(X_test), list(y_train), list(y_test)

# Decision Tree Classifier with Class Weighting
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_size=1, class_weight=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.class_weight = class_weight
        self.tree = None

    # Gini Index with Class Weighting
    def gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            labels = [row[-1] for row in group]
            class_counts = Counter(labels)
            for class_val in classes:
                p = class_counts[class_val] / size
                weight = self.class_weight[class_val] if self.class_weight else 1.0
                score += (p * p) * weight
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Split dataset
    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Get the best split
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = sys.maxsize, sys.maxsize, float('inf'), None
        features = list(range(len(dataset[0]) - 1))
        for index in features:
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        # Consider class weights when choosing the terminal class
        class_counts = Counter(outcomes)
        if self.class_weight:
            for class_val in class_counts:
                class_counts[class_val] *= self.class_weight[class_val]
        return class_counts.most_common(1)[0][0]

    # Create child splits
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # Check for no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # Check for max depth
        if self.max_depth is not None and depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # Process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)
        # Process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    # Build a decision tree
    def build_tree(self, train):
        root = self.get_split(train)
        self.split(root, 1)
        return root

    # Fit the model
    def fit(self, X, y):
        dataset = [X[i] + [y[i]] for i in range(len(X))]
        self.classes_ = list(set(y))
        if self.class_weight == 'balanced':
            # Compute class weights
            class_counts = Counter(y)
            total_samples = len(y)
            self.class_weight = {cls: total_samples / (len(self.classes_) * count) for cls, count in class_counts.items()}
        self.tree = self.build_tree(dataset)

    # Make a prediction with a decision tree
    def predict_row(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    # Predict
    def predict(self, X):
        predictions = []
        for row in X:
            prediction = self.predict_row(self.tree, row)
            predictions.append(prediction)
        return predictions

# Random Forest Classifier with Balanced Sampling
class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=None, min_size=1, sample_size=None, n_features=None, class_weight=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size  # Sample size for bootstrap sampling
        self.n_features = n_features  # Number of features to consider at each split
        self.class_weight = class_weight
        self.trees = []

    # Balanced bootstrap sample with oversampling
    def subsample(self, X, y):
        data = list(zip(X, y))
        label_to_data = defaultdict(list)
        for x_i, y_i in data:
            label_to_data[y_i].append((x_i, y_i))
        max_count = max(len(items) for items in label_to_data.values())
        sample_X, sample_y = [], []
        for label, items in label_to_data.items():
            n_samples = max_count
            items_sampled = [items[random.randrange(len(items))] for _ in range(n_samples)]
            sample_X.extend([x for x, y in items_sampled])
            sample_y.extend([y for x, y in items_sampled])
        return sample_X, sample_y

    # Random feature selection
    def random_features(self, n_features_total):
        n_features = self.n_features if self.n_features else int(math.sqrt(n_features_total))
        features = random.sample(range(n_features_total), n_features)
        return features

    # Fit the model
    def fit(self, X, y):
        n_features_total = len(X[0])
        for i in range(self.n_trees):
            sample_X, sample_y = self.subsample(X, y)
            features = self.random_features(n_features_total)
            # Reduce the dataset to selected features
            sample_X_reduced = [[row[i] for i in features] for row in sample_X]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_size=self.min_size, class_weight=self.class_weight)
            tree.fit(sample_X_reduced, sample_y)
            self.trees.append((tree, features))
            # Update progress bar
            print_progress_bar(i + 1, self.n_trees, prefix='Building Trees:', suffix='Complete', length=50)

    # Predict
    def predict(self, X):
        predictions = []
        for idx, row in enumerate(X):
            tree_predictions = []
            for tree, features in self.trees:
                row_reduced = [row[i] for i in features]
                prediction = tree.predict([row_reduced])[0]
                tree_predictions.append(prediction)
            # Majority vote
            prediction = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(prediction)
            # Update progress bar
            print_progress_bar(idx + 1, len(X), prefix='Predicting:', suffix='Complete', length=50)
        return predictions

    # Feature Importances
    def feature_importances(self, n_features_total):
        importances = [0.0] * n_features_total
        for tree, features in self.trees:
            # Assign importance to the features used in this tree
            for idx in features:
                importances[idx] += 1
        total_importance = sum(importances)
        importances = [fi / total_importance for fi in importances]
        return importances

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    unique_classes = sorted(set(y_true))
    matrix = [[0 for _ in unique_classes] for _ in unique_classes]
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    for true, pred in zip(y_true, y_pred):
        i = class_to_index[true]
        j = class_to_index[pred]
        matrix[i][j] += 1
    return matrix, unique_classes

# Classification Report
def classification_report(y_true, y_pred):
    matrix, classes = confusion_matrix(y_true, y_pred)
    report = {}
    for idx, cls in enumerate(classes):
        tp = matrix[idx][idx]
        fp = sum([matrix[i][idx] for i in range(len(classes)) if i != idx])
        fn = sum([matrix[idx][i] for i in range(len(classes)) if i != idx])
        tn = sum([matrix[i][j] for i in range(len(classes)) for j in range(len(classes)) if i != idx and j != idx])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }
    return report

# Main code
def main():
    # Load data
    file_path = "./SniperDataset.csv"  # Replace with your actual dataset path
    header, data = load_csv(file_path)

    # Features and label indices
    feature_names = [
        "DistanceFromTarget", 
        "ElevationDifference", 
        "GunTiltX", 
        "GunTiltY", 
        "XDifference", 
        "YDifference", 
        "Zdifference"
    ]
    label_name = "HitOrMiss"

    feature_indices = [header.index(name) for name in feature_names]
    label_index = header.index(label_name)

    # Extract features and labels
    X, y = extract_features_labels(data, feature_indices, label_index)

    # Standardize features
    X = standardize_features(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=True)
    
    # Method 1: Baseline Random Forest with Class Weighting
    print("\n--- Method 1: Baseline Random Forest with Class Weighting ---")
    rf_baseline = RandomForestClassifier(n_trees=10, max_depth=10, class_weight='balanced')
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    matrix, classes = confusion_matrix(y_test, y_pred_baseline)
    print("\nConfusion Matrix:")
    print(matrix)
    report = classification_report(y_test, y_pred_baseline)
    print("\nClassification Report:")
    for cls, metrics in report.items():
        print(f"Class {cls}: {metrics}")
    save_model(rf_baseline, "rf_baseline_model.pkl")
    
    # Method 2: Oversampling the Minority Class (Adjusted)
    print("\n--- Method 2: Oversampling the Minority Class ---")
    # Oversample minority class in training data
    data_train = list(zip(X_train, y_train))
    label_to_data = defaultdict(list)
    for x_i, y_i in data_train:
        label_to_data[y_i].append((x_i, y_i))
    max_count = max(len(items) for items in label_to_data.values())
    X_train_balanced, y_train_balanced = [], []
    for label, items in label_to_data.items():
        n_samples = max_count
        items_sampled = [items[random.randrange(len(items))] for _ in range(n_samples)]
        X_train_balanced.extend([x for x, y in items_sampled])
        y_train_balanced.extend([y for x, y in items_sampled])

    rf_oversampled = RandomForestClassifier(n_trees=10, max_depth=10, class_weight='balanced')
    rf_oversampled.fit(X_train_balanced, y_train_balanced)
    y_pred_oversampled = rf_oversampled.predict(X_test)
    matrix, classes = confusion_matrix(y_test, y_pred_oversampled)
    print("\nConfusion Matrix:")
    print(matrix)
    report = classification_report(y_test, y_pred_oversampled)
    print("\nClassification Report:")
    for cls, metrics in report.items():
        print(f"Class {cls}: {metrics}")
    save_model(rf_oversampled, "rf_oversampled_model.pkl")
    
    # Method 3: Easy Ensemble (Adjusted)
    print("\n--- Method 3: Easy Ensemble ---")
    def create_balanced_subsets(X, y, n_subsets=5):
        subsets = []
        data = list(zip(X, y))
        for subset_idx in range(n_subsets):
            # Oversample minority class
            label_to_data = defaultdict(list)
            for x_i, y_i in data:
                label_to_data[y_i].append((x_i, y_i))
            max_count = max(len(items) for items in label_to_data.values())
            X_balanced, y_balanced = [], []
            for label, items in label_to_data.items():
                n_samples = max_count
                items_sampled = [items[random.randrange(len(items))] for _ in range(n_samples)]
                X_balanced.extend([x for x, y in items_sampled])
                y_balanced.extend([y for x, y in items_sampled])
            subsets.append((X_balanced, y_balanced))
            # Update progress bar
            print_progress_bar(subset_idx + 1, n_subsets, prefix='Creating Balanced Subsets:', suffix='Complete', length=50)
        return subsets

    balanced_subsets = create_balanced_subsets(X_train, y_train)
    models = []
    for idx, (X_balanced, y_balanced) in enumerate(balanced_subsets):
        rf = RandomForestClassifier(n_trees=10, max_depth=10, class_weight='balanced')
        rf.fit(X_balanced, y_balanced)
        models.append(rf)
        # Update progress bar
        print_progress_bar(idx + 1, len(balanced_subsets), prefix='Training Models:', suffix='Complete', length=50)

    # Aggregate predictions using majority voting
    def majority_vote(models, X):
        predictions = []
        for i in range(len(X)):
            votes = []
            for model in models:
                pred = model.predict([X[i]])[0]
                votes.append(pred)
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)
            # Update progress bar
            print_progress_bar(i + 1, len(X), prefix='Aggregating Predictions:', suffix='Complete', length=50)
        return predictions

    y_pred_ensemble = majority_vote(models, X_test)
    matrix, classes = confusion_matrix(y_test, y_pred_ensemble)
    print("\nConfusion Matrix:")
    print(matrix)
    report = classification_report(y_test, y_pred_ensemble)
    print("\nClassification Report:")
    for cls, metrics in report.items():
        print(f"Class {cls}: {metrics}")
    save_model(models, "ensemble_model.pkl")
    # Method 4: Feature Importance (Adjusted)
    print("\n--- Method 4: Feature Importance ---")
    rf_feature_importance = RandomForestClassifier(n_trees=10, max_depth=10, class_weight='balanced')
    rf_feature_importance.fit(X_train, y_train)
    importances = rf_feature_importance.feature_importances(len(feature_names))
    sorted_indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
    print("Feature Importance:")
    for idx in sorted_indices:
        print(f"{feature_names[idx]}: {importances[idx]}")

    # Optional: Train on most important features
    important_indices = sorted_indices[:5]
    X_train_important = [[row[i] for i in important_indices] for row in X_train]
    X_test_important = [[row[i] for i in important_indices] for row in X_test]
    rf_feature_selected = RandomForestClassifier(n_trees=10, max_depth=10, class_weight='balanced')
    rf_feature_selected.fit(X_train_important, y_train)
    y_pred_feature_selected = rf_feature_selected.predict(X_test_important)
    matrix, classes = confusion_matrix(y_test, y_pred_feature_selected)
    print("\nConfusion Matrix:")
    print(matrix)
    report = classification_report(y_test, y_pred_feature_selected)
    print("\nClassification Report:")
    for cls, metrics in report.items():
        print(f"Class {cls}: {metrics}")
    save_model(rf_feature_selected, "rf_feature_selected_model.pkl") 
    
if __name__ == "__main__":
    main()
