from scipy.stats import uniform


db_name = 'data/age_gender.db'
train_val_test = {
        'train_val_ratio': 0.1,
        'val_test_ratio': 0.2,
        'shuffle': True,
        'stratify': 'race',
        'random_state': 30980,
        'target_cols': ['age', 'gender', 'race', 'age_interval'],
        }

modelling_dir = 'models/'
conf = {
        'SVC': {
             'C': uniform(0.1, 10),  # Regularization parameter
             'gamma': uniform(0.01, 1),  # Kernel coefficient
             'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
    },
    'KNN': {'n_neighbors' : list(range(5, 50))}
        }

target_mapping = {
        'gender': {'0': 'Male', '1': 'Female'},
        'race': {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian'}
        }




