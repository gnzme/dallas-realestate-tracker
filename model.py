import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.ensemble          import RandomForestRegressor
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import r2_score, mean_absolute_error, mean_squared_error

# ─── Configuración ───────────────────────────────────────────
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. Cargar datos ─────────────────────────────────────────
def load_data():
    files = glob.glob("data/dallas_listings_*.csv")
    if not files:
        raise FileNotFoundError("No CSV found. Run scraper.py first.")
    df = pd.read_csv(max(files))
    print(f"  Loaded: {len(df):,} rows")
    return df


# ─── 2. Feature engineering ──────────────────────────────────
def prepare_features(df):
    """
    Selecciona y prepara las variables que el modelo usará.
    Excluimos price_per_sqft y zestimate porque son derivadas
    del precio — usarlas causaría data leakage.
    """
    features = [
        "sqft",           # superficie en sqft
        "bedrooms",       # dormitorios
        "bathrooms",      # baños
        "year_built",     # año de construcción
        "zipcode",        # ubicación (categórica)
        "property_type",  # tipo de propiedad (categórica)
        "lot_size_acres"  # tamaño del terreno
    ]
    target = "price"

    df = df[features + [target]].copy()

    # Limpiar precios inválidos
    df = df[df[target] > 10_000].copy()
    p99 = df[target].quantile(0.99)
    df  = df[df[target] <= p99].copy()

    # Rellenar nulos numéricos con mediana
    for col in ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size_acres"]:
        df[col] = df[col].fillna(df[col].median())

    # Rellenar nulos categóricos
    df["property_type"] = df["property_type"].fillna("Single Family")
    df["zipcode"]       = df["zipcode"].fillna(df["zipcode"].mode()[0])

    # Convertir zipcode a string
    df["zipcode"] = df["zipcode"].astype(str)

    print(f"  Clean dataset: {len(df):,} rows × {len(features)} features")
    return df, features


# ─── 3. Encodear variables categóricas ───────────────────────
def encode_features(df, features):
    """
    Convierte texto a números para que el modelo pueda procesarlos.
    LabelEncoder: 'Single Family' → 0, 'Condo' → 1, etc.
    Guardamos los encoders para usarlos después en predicciones nuevas.
    """
    encoders = {}

    for col in ["zipcode", "property_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    return df, encoders


# ─── 4. Entrenar modelo ───────────────────────────────────────
def train_model(df, features):
    """
    Random Forest: entrena 200 árboles de decisión y promedia
    sus predicciones — mucho más robusto que un solo árbol.
    """
    X = df[features]
    y = df["price"]

    # 80% entrenamiento, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Training set : {len(X_train):,} rows")
    print(f"  Test set     : {len(X_test):,} rows")

    model = RandomForestRegressor(
        n_estimators = 200,   # 200 árboles
        max_depth    = 15,    # profundidad máxima por árbol
        random_state = 42,
        n_jobs       = -1     # usa todos los núcleos del CPU
    )

    print("\n  Training model... ", end="")
    model.fit(X_train, y_train)
    print("✓")

    return model, X_test, y_test, X_train, y_train


# ─── 5. Evaluar modelo ────────────────────────────────────────
def evaluate_model(model, X_test, y_test, features):
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "="*50)
    print("  MODEL PERFORMANCE")
    print("="*50)
    print(f"  R² Score  : {r2:.3f}  (1.0 = perfect)")
    print(f"  MAE       : ${mae:,.0f}  (avg prediction error)")
    print(f"  RMSE      : ${rmse:,.0f}")
    print("="*50)

    if r2 >= 0.80:
        print("  ✓ Excellent model — ready for production")
    elif r2 >= 0.65:
        print("  ✓ Good model — solid for a portfolio project")
    elif r2 >= 0.50:
        print("  ~ Acceptable — more data or features could help")
    else:
        print("  ✗ Needs improvement")

    # Importancia de variables
    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\n  Feature importance (what drives price the most):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<16} {bar} {imp:.3f}")

    return r2, mae, rmse, importances


# ─── 6. Guardar modelo y encoders ────────────────────────────
def save_model(model, encoders, features, stats):
    joblib.dump(model,    f"{MODEL_DIR}/price_model.joblib")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    joblib.dump(stats,    f"{MODEL_DIR}/model_stats.joblib")
    print(f"\n  ✓ Model saved to {MODEL_DIR}/")


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Dallas House Price Predictor — Training")
    print(" " + "─"*42)

    print("\n[1/5] Loading data...")
    df_raw = load_data()

    print("\n[2/5] Preparing features...")
    df, features = prepare_features(df_raw)

    print("\n[3/5] Encoding categorical variables...")
    df, encoders = encode_features(df, features)

    print("\n[4/5] Training Random Forest model...")
    model, X_test, y_test, X_train, y_train = train_model(df, features)

    print("\n[5/5] Evaluating model...")
    r2, mae, rmse, importances = evaluate_model(
        model, X_test, y_test, features
    )

    stats = {"r2": r2, "mae": mae, "rmse": rmse}
    save_model(model, encoders, features, stats)

    # Test rápido de predicción
    print("\n  Quick prediction test:")
    sample = pd.DataFrame([{
        "sqft":          2000,
        "bedrooms":      3,
        "bathrooms":     2,
        "year_built":    2010,
        "zipcode":       encoders["zipcode"].transform(["75206"])[0],
        "property_type": encoders["property_type"].transform(["singleFamily"])[0],
        "lot_size_acres": 0.15
    }])
    pred = model.predict(sample)[0]
    print(f"  3bed/2bath, 2000sqft, 2010, zip 75206")
    print(f"  Predicted price: ${pred:,.0f}")

    print("\n✓ Week 1 complete — model ready.\n")