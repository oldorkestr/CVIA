from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Створення та навчання моделі випадкового лісу з додатковими параметрами
clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features=3, random_state=0)
clf.fit(X_train, y_train)

# Оцінка точності моделі на тестовій вибірці
accuracy = clf.score(X_test, y_test)

# Виведення результатів
print("Точність моделі: {:.2f}%".format(accuracy*100))
