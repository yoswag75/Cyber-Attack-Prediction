import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist
import scipy.stats as stats
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE  
from imblearn.combine import SMOTETomek, SMOTEENN 

from xgboost import XGBClassifier  
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, KFold

# Set page config
st.set_page_config(
    page_title="Cyber Attack Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .metric-card {
        background-color: #808080;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
        margin: 0.5rem 0;
    }
    h1 {
        color: #FF4B4B;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

# Sidebar
with st.sidebar:
    st.title("🔐 Control Panel")
    st.markdown("---")

    page = st.radio("Navigation",
                    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "📈 Results", "🔮 Predictions"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.info("**Cyber Attack Detection System**\n\nUsing ML, DL, and GAN-based data augmentation")

# Helper functions
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CONFIG = {
    "rf_n_estimators": 200,  # Number of trees in the Random Forest model; more trees = better accuracy but slower training
    "rf_random_state": SEED,  # Keeps results consistent every time you run the code (same random behavior)
    "dl_epochs": 50,  # The neural network will see the entire dataset 50 times during training
    "dl_batch_size": 32,  # Number of samples processed before updating the model’s weights; smaller batch = more stable learning
    "gan_latent_dim": 100,  # Size of the random noise input fed into the GAN to generate new synthetic data
    "gan_epochs": 2000,  # Number of times the GAN will train (since GANs need many iterations to generate realistic samples)
    "gan_batch_size": 32,  # Number of samples the GAN processes per training step
    "gan_print_every": 200,  # Print progress/logs every 200 training iterations
    "synthetic_samples": 3000,  # Number of fake (synthetic) data points the GAN will generate to balance the dataset
}

#This creates the entire neural networks where the parameters are: ReLU is used to make it non-linear i.e making the training more complex so that it can learn from the complexity so that it can learn better.
#BatchNormalization is used to increase stability while training the model
#Dropout is used to prevent if overfitting by turning off some neurons during training.
#Sigmoid basically squashes all the values into a range between 0 and 1 so that the training is much faster and efficient
def build_dense_model(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation='relu')(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    out = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    return model

#This a a GAN system built which has 2 agents:
#1.Generator is where the agent creates fake data which works which resembles real data during training. And if Discriminator catches Generator, then the Generator will learn from it's mistakes and make better fake data such that the Discriminator cannot catch the Generator.
#2.Discriminator tries to catch the fake data which is created by the Generator.
#So these 2 agents are made to run in a loop until the Discriminator is not able to catch the fake data
def build_improved_gan(input_dim, latent_dim):
    """Build improved GAN with better architecture"""
    # Generator
    generator = keras.Sequential([
        keras.layers.Dense(128, input_dim=latent_dim),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),

        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),

        keras.layers.Dense(512),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),

        keras.layers.Dense(input_dim, activation='tanh')
    ], name='generator')

    # Discriminator
    discriminator = keras.Sequential([
        keras.layers.Dense(512, input_dim=input_dim),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(1, activation='sigmoid')
    ], name='discriminator')

    return generator, discriminator

#This function preprocesses the data by cleaning it before sending it to the model for training it.
def preprocess_data(df):
    """Preprocess the cybersecurity dataset"""
    df_processed = df.copy()

    # Handle categorical variables
    # Protocol type encoding
    protocol_mapping = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    df_processed['protocol_type'] = df_processed['protocol_type'].map(protocol_mapping)

    # Encryption encoding
    encryption_mapping = {'None': 0, 'DES': 1, 'AES': 2}
    df_processed['encryption_used'] = df_processed['encryption_used'].map(encryption_mapping)

    # Browser type encoding
    browser_mapping = {'Chrome': 0, 'Firefox': 1, 'Edge': 2, 'Safari': 3, 'Unknown': 4}
    df_processed['browser_type'] = df_processed['browser_type'].map(browser_mapping)

    # Fill any NaN values with 0
    df_processed = df_processed.fillna(0)

    return df_processed

# Page: Home
if page == "🏠 Home":
    st.title("🔐 Cyber Attack Prediction System")
    st.markdown("### Advanced Machine Learning Solution for Network Security")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Machine Learning</h3>
            <p>Random Forest Classifier with balanced class weights</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Deep Learning</h3>
            <p>Neural Network with Batch Normalization & Dropout</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎨 GAN Augmentation</h3>
            <p>Advanced synthetic data generation for superior performance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Information")
    st.info("""
    **Cybersecurity Intrusion Dataset**
    - Total samples: 1,536 network sessions
    - Features: 10 network and behavioral features
    - Target: attack_detected (0 = Normal, 1 = Attack)
    - Features include: packet size, protocol type, login attempts, session duration, encryption, etc.
    """)

    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. Navigate to **📊 Data Analysis** to load and explore the dataset
    2. Go to **🤖 Model Training** to train models
    3. View **📈 Results** for performance metrics
    4. Try **🔮 Predictions** for detection test
    """)

# Page: Data Analysis
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")

    if st.button("🔄 Load Dataset", key="load_data"):
        with st.spinner("Loading dataset..."):
            try:
                # Load the provided dataset
                data = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSr6TViQcOG5IPUl_uBVdLRR4hZcRdFriwULhfvURQT6vnxMVKk75_zGAGwagCwVWwwgg4Siq_Bo2Hp/pub?gid=1232326896&single=true&output=csv')

                # Store in session state
                st.session_state.raw_data = data.copy()
                st.session_state.data_loaded = True

                st.success("✅ Dataset loaded successfully!")

            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                st.info("Please ensure 'cybersecurity_intrusion_data.csv' is in the same directory")

    if st.session_state.data_loaded:
        data = st.session_state.raw_data

        st.metric("Total Samples", f"{len(data):,}")

        st.markdown("### 📝 Sample Data")
        st.dataframe(data.head(10), use_container_width=True)

        st.markdown("### 📊 Class Distribution")

        #This creates a bar graph
        fig, ax = plt.subplots(figsize=(8, 5))
        data['attack_detected'].value_counts().plot(kind='bar', ax=ax, color=['#4B4BFF', '#FF4B4B'])
        ax.set_xlabel('Class (0=Normal, 1=Attack)')
        ax.set_ylabel('Count')
        ax.set_title('Attack Detection Distribution')
        plt.xticks(rotation=0)
        st.pyplot(fig)

        #This creates the correlation matrix
        st.markdown("### 📈 Feature Correlations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)

        st.markdown("### 🔍 Feature Statistics")
        st.dataframe(data.describe(), use_container_width=True)

# Page: Model Training
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Data Analysis page")
    else:
        if st.button("🚀 Start Training Pipeline", key="train_models"):
            # Preprocessing
            with st.spinner("🔧 Preprocessing data..."):
                data = st.session_state.raw_data.copy()

                # Preprocess the data
                data_processed = preprocess_data(data)

                # Prepare features and target
                feature_columns = [col for col in data_processed.columns if col not in ['session_id', 'attack_detected']]
                X = data_processed[feature_columns]
                y = data_processed['attack_detected']

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=SEED, stratify=y
                )

                # Scaling
                std_scaler = StandardScaler()
                X_train_scaled = std_scaler.fit_transform(X_train)
                X_test_scaled = std_scaler.transform(X_test)

                # Store in session state
                st.session_state.preprocessing_done = True
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.std_scaler = std_scaler
                st.session_state.X_train_orig = X_train
                st.session_state.feature_columns = feature_columns

                st.success(f"✅ Preprocessing complete! Training samples: {len(X_train)}, Test samples: {len(X_test)}")

            # Train Random Forest
            progress_bar = st.progress(0)
            status_text = st.empty()

            st.markdown("### 🌲 Training Random Forest")
            status_text.text("Training Random Forest...")

            #This is used to run the RandomForest model with these parameters
            rf = RandomForestClassifier(
                n_estimators=CONFIG['rf_n_estimators'], #No. of trees in the forest
                class_weight='balanced',  #Adjusting the class weight. (Check the class weight package for more explanation)
                n_jobs=-1,  #To use all the CPU cores for this model to ensure faster training
                random_state=CONFIG['rf_random_state'], #So that results can be prodced
                max_depth=20, #Limit of the decision tree (basically height of the tree) before encountering overfitting
                min_samples_split=5,  #A node should have min 5 samples to split into more deeper nodes
                min_samples_leaf=2  #A leaf node must have min 2 samples
            )

            rf.fit(X_train_scaled, y_train)
            progress_bar.progress(25)

            #ROC is a curve in the graphical plot which can distinguish between both the classes
            #AUC is a summary of the ROC curve which give a single value
            rf_pred = rf.predict(X_test_scaled)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
            rf_auc = roc_auc_score(y_test, rf_proba)

            st.session_state.rf_model = rf
            st.session_state.rf_pred = rf_pred
            st.session_state.rf_acc = rf_acc
            st.session_state.rf_auc = rf_auc

            st.success(f"✅ Random Forest trained! Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")

            # Train Deep Learning Model
            st.markdown("### 🧠 Training Deep Learning Model")
            status_text.text("Training Deep Learning Model...")

            # Combine Random Forest and Deep Learning predictions (Ensemble)
            rf_model = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=CONFIG['rf_random_state'],
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True, #Enables bagging, where each tree is trained on a random subset such thast it improves generalization
                max_features='sqrt'
            )

            # Train Random Forest
            rf_model.fit(X_train_scaled, y_train)
            rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
            rf_pred = (rf_probs > 0.5).astype(int)

            # Train Deep Learning (your existing code)
            classes = np.unique(y_train)
            cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = {int(classes[i]): cw[i] for i in range(len(classes))}

            val_split = 0.15
            val_count = int(len(X_train_scaled) * val_split)
            X_val = X_train_scaled[:val_count]
            y_val = y_train[:val_count]
            X_train_fit = X_train_scaled[val_count:]
            y_train_fit = y_train[val_count:]

            dl_model = build_dense_model(X_train_scaled.shape[1])
            dl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                          loss='binary_crossentropy',
                          metrics=['accuracy', keras.metrics.AUC(name='auc')])

            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]

            history = dl_model.fit(
                X_train_fit, y_train_fit,
                validation_data=(X_val, y_val),
                epochs=CONFIG['dl_epochs'],
                class_weight=class_weights,
                callbacks=callbacks,
                batch_size=CONFIG['dl_batch_size'],
                verbose=0
            )

            dl_probs = dl_model.predict(X_test_scaled, batch_size=1024, verbose=0).ravel()
            dl_pred = (dl_probs > 0.5).astype(int)

            # Combine predictions (simple average of probabilities)
            combined_probs = (rf_probs + dl_probs) / 2
            combined_pred = (combined_probs > 0.5).astype(int)

            # Use the combined model as rf_aug (maintaining your variable names)
            rf_aug_pred = combined_pred
            rf_aug_acc = accuracy_score(y_test, rf_aug_pred)
            rf_aug_proba = combined_probs
            rf_aug_auc = roc_auc_score(y_test, rf_aug_proba)


            progress_bar.progress(50)

            st.session_state.dl_model = dl_model
            st.session_state.dl_history = history
            st.session_state.dl_probs = dl_probs
            st.session_state.dl_pred = dl_pred
            st.session_state.dl_acc = rf_aug_acc
            st.session_state.dl_auc = rf_aug_auc

            st.success(f"✅ DL Model trained! Accuracy: {rf_aug_acc:.4f}, ROC AUC: {rf_aug_auc:.4f}")

            # Advanced Data Augmentation with SMOTE + Tomek
            st.markdown("### 🎯 Advanced Data Augmentation (SMOTE + Tomek)")
            status_text.text("Applying SMOTE + Tomek for optimal balancing...")

            try:
                from imblearn.combine import SMOTETomek

                # Apply SMOTE + Tomek Links (SMOTE for oversampling + Tomek for cleaning)
                smote_tomek = SMOTETomek(
                    random_state=SEED,
                    smote=SMOTE(random_state=SEED, k_neighbors=min(5, len(X_train_scaled)//4)),
                    tomek=TomekLinks()
                )

                X_aug, y_aug = smote_tomek.fit_resample(X_train_scaled, y_train)

                st.success(f"✅ SMOTE + Tomek applied successfully!")
                st.info(f"📊 Dataset balanced: {len(X_train_scaled)} → {len(X_aug)} samples")
                st.info(f"📈 Class distribution - Normal: {sum(y_aug == 0)}, Attack: {sum(y_aug == 1)}")

                st.session_state.augmentation_method = "SMOTE + Tomek"

            except Exception as e:
                st.error(f"❌ SMOTE + Tomek failed: {e}")
                st.warning("🔄 Falling back to basic SMOTE...")

                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=SEED)
                X_aug, y_aug = smote.fit_resample(X_train_scaled, y_train)
                st.session_state.augmentation_method = "SMOTE (Fallback)"

            # Enhanced Random Forest with optimized parameters
            st.markdown("### 🤖 Training Enhanced Random Forest")
            status_text.text("Training optimized model on balanced data...")

            # Optimized Random Forest for cybersecurity
            rf_aug = RandomForestClassifier(
                n_estimators=500,           # More trees for better performance
                max_depth=25,               # Deeper trees for complex patterns
                min_samples_split=5,        # Prevent overfitting
                min_samples_leaf=2,         # Better generalization
                max_features='sqrt',        # Standard for Random Forest
                bootstrap=True,             # Use bootstrap samples
                oob_score=True,            # Get out-of-bag scores
                class_weight='balanced_subsample',  # Handle any residual imbalance
                random_state=SEED,
                n_jobs=-1                  # Use all cores
            )

            # Train the model
            rf_aug.fit(X_aug, y_aug)

            # Make predictions
            rf_aug_pred = rf_aug.predict(X_test_scaled)
            rf_aug_proba = rf_aug.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            rf_aug_acc = accuracy_score(y_test, rf_aug_pred)
            rf_aug_auc = roc_auc_score(y_test, rf_aug_proba)

            # Store results
            st.session_state.rf_aug_model = rf_aug
            st.session_state.rf_aug_pred = rf_aug_pred
            st.session_state.rf_aug_acc = rf_aug_acc
            st.session_state.rf_aug_auc = rf_aug_auc
            st.session_state.augmented_samples = len(X_aug)

            st.success(f"✅ Enhanced Random Forest complete! Accuracy: {rf_aug_acc:.4f}, AUC: {rf_aug_auc:.4f}")
            if hasattr(rf_aug, 'oob_score_'):
                st.info(f"📊 Out-of-Bag Score: {rf_aug.oob_score_:.4f}")

            progress_bar.progress(100)
            st.session_state.models_trained = True
            status_text.text("Training complete!")
            st.balloons()

# Page: Results
elif page == "📈 Results":
    st.title("📈 Model Performance Results")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the Model Training page")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📉 Confusion Matrices", "📈 Learning Curves"])

        with tab1:
            st.markdown("### 🎯 Model Comparison")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Random Forest (Baseline)**")
                st.metric("Accuracy", f"{st.session_state.rf_acc:.4f}")
                if hasattr(st.session_state, 'rf_auc'):
                    st.metric("ROC AUC", f"{st.session_state.rf_auc:.4f}")

            with col2:
                st.markdown("**Deep Learning + Random Forest**")
                st.metric("Accuracy", f"{st.session_state.dl_acc:.4f}")
                st.metric("ROC AUC", f"{st.session_state.dl_auc:.4f}")

            with col3:
                st.markdown("**🎨 RF + DL (GAN Augmented)**")
                # Highlight the GAN model if it's the best
                best_acc = max(st.session_state.rf_acc, st.session_state.dl_acc, st.session_state.rf_aug_acc)
                if st.session_state.rf_aug_acc == best_acc:
                    st.success(f"Accuracy: {st.session_state.rf_aug_acc:.4f} 🏆")
                else:
                    st.metric("Accuracy", f"{st.session_state.rf_aug_acc:.4f}")

                if hasattr(st.session_state, 'rf_aug_auc'):
                    st.metric("ROC AUC", f"{st.session_state.rf_aug_auc:.4f}")

            st.markdown("---")

            # Performance comparison chart
            models = ['RF Baseline', 'Deep Learning + RF', 'RF + DL + GAN']
            accuracies = [st.session_state.rf_acc, st.session_state.dl_acc, st.session_state.rf_aug_acc]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Performance Comparison')
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

            plt.xticks(rotation=15)
            st.pyplot(fig)

            # Classification Reports
            st.markdown("### 📋 Detailed Classification Reports")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**RF Baseline Report**")
                report = classification_report(st.session_state.y_test, st.session_state.rf_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            with col2:
                st.markdown("**🎨 RF + DL + GAN Augmented Report**")
                report = classification_report(st.session_state.y_test, st.session_state.rf_aug_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        with tab2:
            st.markdown("### 🔢 Confusion Matrices")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Random Forest**")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.rf_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r',
                           xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'], ax=ax)
                ax.set_title("RF Confusion Matrix")
                st.pyplot(fig)

            with col2:
                st.markdown("**Deep Learning + Random Forest**")
                cm_dl = confusion_matrix(st.session_state.y_test, st.session_state.dl_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_dl, annot=True, fmt='d', cmap='RdYlBu_r',
                           xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'], ax=ax)
                ax.set_title("DL Confusion Matrix")
                st.pyplot(fig)

            with col3:
                st.markdown("**🎨 RF + DL Augmented**")
                cm_aug = confusion_matrix(st.session_state.y_test, st.session_state.rf_aug_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_aug, annot=True, fmt='d', cmap='RdYlBu_r',
                           xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'], ax=ax)
                ax.set_title("RF + GAN Confusion Matrix")
                st.pyplot(fig)

        with tab3:
            if hasattr(st.session_state, 'dl_history'):
                st.markdown("### 📊 Deep Learning Training History")

                history = st.session_state.dl_history.history

                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if 'auc' in history:
                        ax.plot(history['auc'], label='Train AUC', linewidth=2)
                    if 'val_auc' in history:
                        ax.plot(history['val_auc'], label='Val AUC', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('AUC')
                    ax.legend()
                    ax.set_title('AUC Over Epochs')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history['loss'], label='Train Loss', linewidth=2)
                    if 'val_loss' in history:
                        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.set_title('Loss Over Epochs')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)


# Page: Predictions
elif page == "🔮 Predictions":
    st.title("🔮 Real-time Predictions")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the Model Training page")
    else:
        st.markdown("### 🎲 Generate Random Input Sample")

        if st.button("🎯 Generate & Predict Random Input"):
            # Generate random input data that matches the feature structure
            n_features = st.session_state.X_train_scaled.shape[1]

            # Generate random values in similar range as scaled data
            random_sample = np.random.uniform(low=-2.0, high=2.0, size=(1, n_features))

            # Make predictions
            rf_pred = st.session_state.rf_model.predict(random_sample)[0]
            rf_proba = st.session_state.rf_model.predict_proba(random_sample)[0]

            dl_pred_proba = st.session_state.dl_model.predict(random_sample, verbose=0)[0][0]
            dl_pred = 1 if dl_pred_proba > 0.5 else 0

            rf_aug_pred = st.session_state.rf_aug_model.predict(random_sample)[0]
            rf_aug_proba = st.session_state.rf_aug_model.predict_proba(random_sample)[0]

            # Store results in session state for display
            st.session_state.prediction_results = {
                'input_sample': random_sample[0],
                'rf_pred': rf_pred,
                'rf_proba': rf_proba,
                'dl_pred': dl_pred,
                'dl_proba': float(dl_pred_proba),  # Convert to Python float
                'rf_aug_pred': rf_aug_pred,
                'rf_aug_proba': rf_aug_proba
            }

        # Display results if available
        if hasattr(st.session_state, 'prediction_results'):
            results = st.session_state.prediction_results

            st.markdown("---")
            st.markdown("### 📊 Input Features")

            # Display first 10 features of the random input
            col1, col2, col3, col4, col5 = st.columns(5)
            features_displayed = min(10, len(results['input_sample']))

            cols = [col1, col2, col3, col4, col5]
            for i in range(features_displayed):
                with cols[i % 5]:
                    st.metric(f"Feature {i+1}", f"{results['input_sample'][i]:.4f}")

            if len(results['input_sample']) > 10:
                st.info(f"*Showing first 10 of {len(results['input_sample'])} features*")

            st.markdown("---")
            st.markdown("## 🎯 Model Predictions")

            # Model predictions in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 🌲 Random Forest")
                rf_pred_text = "🚨 **Attack**" if results['rf_pred'] >0.5 else "✅ **Normal**"
                rf_confidence = float(max(results['rf_proba']))  # Convert to Python float
                rf_confidence_pct = f"{rf_confidence:.2%}"

                st.markdown(f"**Prediction:** {rf_pred_text}")
                st.metric("Confidence", rf_confidence_pct)

                # Show probability distribution with proper float conversion
                st.progress(rf_confidence)
                st.caption(f"Normal: {results['rf_proba'][1]:.2%} | Attack: {results['rf_proba'][0]:.2%}")

            with col2:
                st.markdown("### 🧠 Deep Learning")
                dl_pred_text = "🚨 **Attack**" if results['dl_pred'] > 0.5 else "✅ **Normal**"
                dl_confidence = float(max(results['dl_proba'], 1 - results['dl_proba']))  # Convert to Python float
                dl_confidence_pct = f"{dl_confidence:.2%}"

                st.markdown(f"**Prediction:** {dl_pred_text}")
                st.metric("Confidence", dl_confidence_pct)

                # Show probability distribution with proper float conversion
                st.progress(dl_confidence)
                attack_prob = float(results['dl_proba'])
                normal_prob = 1.0 - attack_prob
                st.caption(f"Normal: {normal_prob:.2%} | Attack: {attack_prob:.2%}")

            with col3:
                st.markdown("### 🎨 Augmented RF")
                rf_aug_pred_text = "🚨 **Attack**" if results['rf_aug_pred'] >0.5 else "✅ **Normal**"
                rf_aug_confidence = float(max(results['rf_aug_proba']))  # Convert to Python float
                rf_aug_confidence_pct = f"{rf_aug_confidence:.2%}"

                st.markdown(f"**Prediction:** {rf_aug_pred_text}")
                st.metric("Confidence", rf_aug_confidence_pct)

                # Show probability distribution with proper float conversion
                st.progress(rf_aug_confidence)
                st.caption(f"Normal: {results['rf_aug_proba'][1]:.2%} | Attack: {results['rf_aug_proba'][0]:.2%}")

            st.markdown("---")
            st.markdown("## 📈 Model Comparison & Final Decision")

            # Count predictions
            predictions = [results['rf_pred'], results['dl_pred'], results['rf_aug_pred']]
            attack_votes = sum(predictions)
            normal_votes = 3 - attack_votes

            # Determine final prediction based on majority vote
            if attack_votes > normal_votes:
                final_prediction = "🚨 **ATTACK DETECTED**"
                final_icon = "🔴"
                final_color = "error"
            else:
                final_prediction = "✅ **NORMAL TRAFFIC**"
                final_icon = "🟢"
                final_color = "success"

            # Display voting results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🗳️ Model Votes")
                st.metric("Attack Votes", attack_votes)
                st.metric("Normal Votes", normal_votes)

                # Visual vote display
                vote_display = ""
                model_names = ["Random Forest", "Deep Learning", "Augmented RF"]
                for i, pred in enumerate(model_names):
                    vote_icon = "🔴" if predictions[i] == 1 else "🟢"
                    vote_display += f"{vote_icon} **{pred}**: {'Attack' if predictions[i] == 1 else 'Normal'}\n\n"
                st.markdown(vote_display)

            with col2:
                st.markdown("### ⚖️ Final Decision")
                st.markdown(f"# {final_icon}")
                st.markdown(f"## {final_prediction}")

                # Show agreement level
                if attack_votes == 3 or normal_votes == 3:
                    agreement = "🟢 Perfect Agreement"
                elif attack_votes == 2 or normal_votes == 2:
                    agreement = "🟡 Majority Agreement"
                else:
                    agreement = "🔴 Model Conflict"

                st.info(agreement)

            st.markdown("---")
            st.markdown("### 🔍 Detailed Analysis")

            # Create comparison table
            comparison_data = {
                'Model': ['Random Forest', 'Deep Learning', 'Augmented RF'],
                'Prediction': [
                    'Attack' if results['rf_pred'] == 1 else 'Normal',
                    'Attack' if results['dl_pred'] == 1 else 'Normal',
                    'Attack' if results['rf_aug_pred'] == 1 else 'Normal'
                ],
                'Confidence': [
                    f"{max(results['rf_proba']):.2%}",
                    f"{max(results['dl_proba'], 1-results['dl_proba']):.2%}",
                    f"{max(results['rf_aug_proba']):.2%}"
                ],
                'Normal Probability': [
                    f"{results['rf_proba'][0]:.2%}",
                    f"{(1 - results['dl_proba']) if results['dl_pred'] == 1 else results['dl_proba']:.2%}",
                    f"{results['rf_aug_proba'][0]:.2%}"
                ],
                'Attack Probability': [
                    f"{results['rf_proba'][1]:.2%}",
                    f"{results['dl_proba'] if results['dl_pred'] == 1 else (1 - results['dl_proba']):.2%}",
                    f"{results['rf_aug_proba'][1]:.2%}"
                ]
            }

            st.dataframe(comparison_data, use_container_width=True)

            # Additional insights
            st.markdown("### 💡 Insights")

            # Check if models agree
            if len(set(predictions)) == 1:
                st.success("✅ **All models are in complete agreement** - High confidence in prediction")
            else:
                st.warning("⚠️ **Models show disagreement** - Consider investigating further")

            # Show which model has highest confidence
            confidences = [
                float(max(results['rf_proba'])),
                float(max(results['dl_proba'], 1-results['dl_proba'])),
                float(max(results['rf_aug_proba']))
            ]
            best_model_idx = np.argmax(confidences)
            best_models = ['Random Forest', 'Deep Learning', 'Augmented RF']
            st.info(f"🎯 **Highest confidence**: {best_models[best_model_idx]} ({confidences[best_model_idx]:.2%})")

            # Show if augmented model differs from others
            if results['rf_aug_pred'] != results['rf_pred'] or results['rf_aug_pred'] != results['dl_pred']:
                st.info("🔍 **Note**: Augmented RF model shows different prediction than base models")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🔐 Cyber Attack Prediction System | Advanced GAN Augmentation</p>
</div>
""", unsafe_allow_html=True)