import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dagster import asset, Definitions, materialize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# DAGSTER ASSETS
# =========================

@asset
def raw_data():
    return pd.read_csv("AB_NYC_2019.csv")


@asset
def cleaned_data(raw_data):
    df = raw_data.copy()

    df = df.dropna(subset=["price", "room_type", "neighbourhood_group"])
    df["minimum_nights"] = df["minimum_nights"].fillna(df["minimum_nights"].median())

    df["price_category"] = pd.cut(
        df["price"],
        bins=[0, 100, 300, 10000],
        labels=["Cheap", "Medium", "Expensive"]
    )

    df = df.dropna(subset=["price_category"])
    return df


@asset
def features(cleaned_data):
    df = cleaned_data.copy()

    le1, le2 = LabelEncoder(), LabelEncoder()
    df["room_type_enc"] = le1.fit_transform(df["room_type"])
    df["area_enc"] = le2.fit_transform(df["neighbourhood_group"])

    X = df[["room_type_enc", "area_enc", "minimum_nights"]]
    y = df["price_category"]

    return X, y


@asset
def train_test_data(features):
    X, y = features
    return train_test_split(X, y, test_size=0.2, random_state=42)


@asset
def decision_tree_model(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return {"model": model, "accuracy": acc}


@asset
def random_forest_model(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return {"model": model, "accuracy": acc}


@asset
def logistic_regression_model(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return {"model": model, "accuracy": acc}


defs = Definitions(
    assets=[
        raw_data,
        cleaned_data,
        features,
        train_test_data,
        decision_tree_model,
        random_forest_model,
        logistic_regression_model,
    ]
)


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    print("\n===== FIRST RUN : ALL MODELS =====\n")

    result = materialize([
        raw_data,
        cleaned_data,
        features,
        train_test_data,
        decision_tree_model,
        random_forest_model,
        logistic_regression_model,
    ])

    dt_acc1 = result.output_for_node("decision_tree_model")["accuracy"] * 100
    rf_acc1 = result.output_for_node("random_forest_model")["accuracy"] * 100
    lr_acc1 = result.output_for_node("logistic_regression_model")["accuracy"] * 100

    print(f"Decision Tree Accuracy      : {dt_acc1:.2f}%")
    print(f"Random Forest Accuracy      : {rf_acc1:.2f}%")
    print(f"Logistic Regression Accuracy: {lr_acc1:.2f}%")


    # =========================
    # EDA
    # =========================

    df = pd.read_csv("AB_NYC_2019.csv")

    plt.figure()
    sns.histplot(df["price"], bins=50)
    plt.title("Price Distribution")
    plt.show()

    plt.figure()
    sns.boxplot(x="room_type", y="price", data=df)
    plt.ylim(0, 500)
    plt.title("Room Type vs Price")
    plt.show()


    # =========================
    # MODEL COMPARISON
    # =========================

    plt.figure()
    plt.bar(
        ["Decision Tree", "Random Forest", "Logistic Regression"],
        [dt_acc1, rf_acc1, lr_acc1]
    )
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.show()


    # =========================
    # DATASET CHANGE
    # =========================

    df_mod = df.sample(frac=0.8, random_state=42)
    df_mod.to_csv("AB_NYC_2019_modified.csv", index=False)
    print("\nDataset modified (20% rows removed)\n")


    @asset
    def raw_data_modified():
        return pd.read_csv("AB_NYC_2019_modified.csv")


    print("===== SECOND RUN : ONLY BEST MODEL (DECISION TREE) =====\n")

    result_run2 = materialize([
        raw_data_modified,
        cleaned_data,
        features,
        train_test_data,
        decision_tree_model,
    ])

    dt_acc2 = result_run2.output_for_node("decision_tree_model")["accuracy"] * 100
    print(f"Decision Tree Accuracy After Data Change: {dt_acc2:.2f}%")


    plt.figure()
    plt.bar(
        ["Decision Tree (Run 1)", "Decision Tree (Run 2)"],
        [dt_acc1, dt_acc2]
    )
    plt.ylabel("Accuracy (%)")
    plt.title("Decision Tree Performance Before vs After Data Change")
    plt.show()


    # =========================
    # TIME WITHOUT DAGSTER
    # =========================

    start_no_dagster = time.time()

    df_manual = pd.read_csv("AB_NYC_2019.csv")
    df_manual = df_manual.dropna(subset=["price", "room_type", "neighbourhood_group"])
    df_manual["minimum_nights"] = df_manual["minimum_nights"].fillna(df_manual["minimum_nights"].median())

    df_manual["price_category"] = pd.cut(
        df_manual["price"], bins=[0,100,300,10000],
        labels=["Cheap","Medium","Expensive"]
    )
    df_manual = df_manual.dropna(subset=["price_category"])

    le1, le2 = LabelEncoder(), LabelEncoder()
    df_manual["room_type_enc"] = le1.fit_transform(df_manual["room_type"])
    df_manual["area_enc"] = le2.fit_transform(df_manual["neighbourhood_group"])

    X = df_manual[["room_type_enc","area_enc","minimum_nights"]]
    y = df_manual["price_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    DecisionTreeClassifier().fit(X_train, y_train)
    RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    LogisticRegression(max_iter=300).fit(X_train, y_train)

    time_no_dagster = (time.time() - start_no_dagster) / 60
    print(f"\nTime WITHOUT Dagster: {time_no_dagster:.3f} minutes")


    # =========================
    # TIME WITH DAGSTER
    # =========================

    start_dagster = time.time()

    materialize([
        raw_data,
        cleaned_data,
        features,
        train_test_data,
        decision_tree_model,
    ])

    time_dagster = (time.time() - start_dagster) / 60
    print(f"Time WITH Dagster: {time_dagster:.3f} minutes")


    plt.figure()
    plt.bar(
        ["Without Dagster", "With Dagster"],
        [time_no_dagster, time_dagster]
    )
    plt.ylabel("Time (minutes)")
    plt.title("Execution Time Comparison")
    plt.show()
