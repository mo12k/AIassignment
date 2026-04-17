from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from joblib import load

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

ENCODER_PATH = BASE_DIR / "encoder_model.keras"
KMEANS_PATH = BASE_DIR / "autoencoder_kmeans.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

# Gender is dropped — model uses only these 3 features
FEATURES = ["Age", "Annual Income", "Spending Score"]

CLUSTER_NAMES = {
    0: "💰 Budget-Conscious Customers",
    1: "👑 High-Value Customers",
    2: "🛒 Average Customers",
}

CLUSTER_DESC = {
    0: "Low income, low spending, often from younger or budget-limited age groups. Price-sensitive shoppers who respond well to discounts and deals.",
    1: "High income, high spending, commonly from mature age groups with stronger purchasing power. Premium customers worth targeting with exclusive offers and loyalty programs.",
    2: "Middle income, moderate spending, typically spread across working-age customers. The typical mall visitor with broad appeal.",
}


# ── Load models (cached — only runs once per session) ─────────────────────────
@st.cache_resource
def load_models():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing encoder model: {ENCODER_PATH}")
    if not KMEANS_PATH.exists():
        raise FileNotFoundError(f"Missing KMeans model: {KMEANS_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler model: {SCALER_PATH}")

    encoder = tf.keras.models.load_model(ENCODER_PATH)
    kmeans = load(KMEANS_PATH)
    scaler = load(SCALER_PATH)
    return encoder, kmeans, scaler


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_cluster(age, annual_income, spending_score, encoder, kmeans, scaler):
    new_customer = np.array(
        [[age, annual_income, spending_score]],
        dtype=np.float32,
    )
    new_customer_scaled = scaler.transform(new_customer)
    latent  = encoder.predict(new_customer_scaled, verbose=0)
    cluster = int(kmeans.predict(latent)[0])
    return latent[0], cluster


def render_feature_space_3d(age, annual_income, spending_score, predicted_cluster, encoder, kmeans, scaler):
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("3D chart requires plotly. Install with: pip install plotly")
        return

    rng = np.random.default_rng(42)
    sample_size = 450

    sample_points = np.column_stack(
        [
            rng.uniform(12.0, 100.0, sample_size),
            rng.uniform(0.0, 500_000.0, sample_size),
            rng.uniform(1.0, 100.0, sample_size),
        ]
    ).astype(np.float32)

    sample_scaled = scaler.transform(sample_points)
    sample_latent = encoder.predict(sample_scaled, verbose=0)
    sample_clusters = kmeans.predict(sample_latent)

    colors = {
        0: "#1f77b4",
        1: "#d62728",
        2: "#2ca02c",
    }

    fig = go.Figure()
    for cluster_id in sorted(np.unique(sample_clusters)):
        mask = sample_clusters == cluster_id
        fig.add_trace(
            go.Scatter3d(
                x=sample_points[mask, 0],
                y=sample_points[mask, 1],
                z=sample_points[mask, 2],
                mode="markers",
                name=CLUSTER_NAMES.get(int(cluster_id), f"Segment {cluster_id}"),
                marker={
                    "size": 3,
                    "opacity": 0.35,
                    "color": colors.get(int(cluster_id), "#7f7f7f"),
                },
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[age],
            y=[annual_income],
            z=[spending_score],
            mode="markers",
            name=f"Your customer ({CLUSTER_NAMES.get(predicted_cluster, f'Segment {predicted_cluster}')})",
            marker={
                "size": 10,
                "color": colors.get(predicted_cluster, "#111111"),
                "symbol": "diamond",
                "line": {"width": 2, "color": "#111111"},
            },
        )
    )

    fig.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        scene={
            "xaxis_title": "Age",
            "yaxis_title": "Annual Income (RM)",
            "zaxis_title": "Spending Score",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segment Predictor",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ Mall Customer Segment Predictor")
st.caption("Autoencoder (3 → 2 latent dims) + K-Means clustering · Gender excluded from model")

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    encoder_model, kmeans_model, scaler_model = load_models()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=12, max_value=100, value=30, step=1)
    with col2:
        annual_income = st.number_input(
            "Annual Income (RM)",
            min_value=0.0, max_value=500_000.0,
            value=60_000.0, step=1_000.0,
        )
    with col3:
        spending_score = st.number_input(
            "Spending Score (1–100)",
            min_value=1.0, max_value=100.0,
            value=50.0, step=1.0,
        )

    submitted = st.form_submit_button("🔍 Predict Segment", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:
    latent_vector, predicted_cluster = predict_cluster(
        age=age,
        annual_income=annual_income,
        spending_score=spending_score,
        encoder=encoder_model,
        kmeans=kmeans_model,
        scaler=scaler_model,
    )

    cluster_name = CLUSTER_NAMES.get(predicted_cluster, f"Segment {predicted_cluster}")
    cluster_desc = CLUSTER_DESC.get(predicted_cluster, "")

    st.divider()
    st.subheader("Prediction Result")
    st.success(f"**Predicted Segment:** {cluster_name}")
    st.info(cluster_desc)   

    st.markdown("**Latent vector (compressed representation):**")
    st.code(np.array2string(latent_vector.round(4), precision=4))

    st.markdown("**3D customer-space visualisation:**")
    render_feature_space_3d(
        age=age,
        annual_income=annual_income,
        spending_score=spending_score,
        predicted_cluster=predicted_cluster,
        encoder=encoder_model,
        kmeans=kmeans_model,
        scaler=scaler_model,
    )

    st.markdown(
        """
        ---
        **How it works:**
        1. Your 3 inputs (Age, Annual Income, Spending Score) are standardised using the saved scaler.
        2. The encoder compresses 3 features → 2 latent dimensions.
        3. K-Means assigns the nearest cluster in latent space.

        > Gender was excluded from the model — mall customer segments are driven
        > primarily by spending behaviour, income, and age.
        """
    )
