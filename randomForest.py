import argparse
import pickle
import warnings
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DROP_COLS = [
    "device_name", "device_mac", "label_full",
    "label1", "label2", "label3", "label4",
    "timestamp", "timestamp_start", "timestamp_end",
    "log_data-types", "log_interval-messages",
    "network_ips_all", "network_ips_dst", "network_ips_src",
    "network_macs_all", "network_macs_dst", "network_macs_src",
    "network_ports_all", "network_ports_dst", "network_ports_src",
    "network_protocols_all", "network_protocols_dst", "network_protocols_src",
    "label", "label2"
]

RF_MODEL_PATH = "models/rf_model.pkl"
RF_META_PATH  = "models/rf_meta.pkl"

def load_data(path):
    print(f"[i] Loading training data from {path} ...")
    data = pd.read_csv(path)

    if "label" not in data.columns:
        print("[!] CSV must have a 'label' column with values 'benign' or 'attack'.")
        sys.exit(1)

    data["label"] = data["label"].apply(
        lambda x: "attack" if str(x).startswith("attack") else x
    )

    counts = data["label"].value_counts().to_dict()
    if "benign" not in counts or "attack" not in counts:
        print(f"[!] Both 'benign' and 'attack' labels are required. Found: {counts}")
        sys.exit(1)

    print(f"[i] Dataset shape: \n{data.shape}")
    print(f"[i] Label distribution:\n {counts}\n")
    return data

def train_random_forest(data, run_eval=True):
    print("[i] Preparing features")
    X = data.drop(columns=["label"])
    y = data["label"].map({"benign": 0, "attack": 1})

    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = le.fit_transform(X[col].astype(str))

    training_medians = X.median()
    X = X.fillna(training_medians)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[i] Training Random Forest (n_estimators=100)")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    if run_eval:
        y_pred   = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, target_names=["Benign", "Attack"])
        print(f"[i] Test accuracy : {accuracy:.2%}")
        print(f"\nClassification Report:\n{report}")

    return clf, X_train.columns.tolist(), training_medians, X_train

def save_model(clf, feature_cols, training_medians, X_train):
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    meta = {
        "feature_cols": feature_cols,
        "training_medians": training_medians,
        "X_train": X_train,
    }
    with open(RF_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[i] Model saved: {RF_MODEL_PATH}")
    print(f"[i] Metadata saved: {RF_META_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Train and save a Random Forest IDS model.")
    parser.add_argument("csv_file", nargs="?", default=None,help="Path to training CSV")
    args = parser.parse_args()
    csv_path = "data.csv"
    data = load_data(csv_path)
    clf, feature_cols, training_medians, X_train = train_random_forest(data, run_eval=not args.no_eval)
    save_model(clf, feature_cols, training_medians, X_train)
    print("\n[i] Done.")

if __name__ == "__main__":
    main()
