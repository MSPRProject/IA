import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from database_service import DatabaseService
from sklearn.tree import DecisionTreeClassifier, export_text


db_service = DatabaseService(password="mspr_password") 
df = db_service.get_full_dataset()
db_service.close()

print("Données extraites :")
print(df.head())

df["report_date"] = pd.to_datetime(df["report_date"])
df["year"] = df["report_date"].dt.year

X = df[["infection_count", "death_count", "population", "year"]]
y = df["pandemic_name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

print("Arbre de décision :")
print(export_text(tree_model, feature_names=list(X.columns)))
