# 📊 Mall Customer Segmentation Dashboard

An interactive Streamlit web application that uses Machine Learning to instantly categorize mall customers into distinct behavioral segments based on their demographic and financial profiles.

🌐 **Live App:** [customer-segmentation-sparsh-singhal.streamlit.app](https://streamlit.app)

---

## 🚀 Overview

This application leverages unsupervised machine learning (K-Means Clustering) to help retail businesses understand their customer base. By inputting basic customer profile details, users can instantly determine which market segment a customer belongs to and visualize their exact position on a cluster map.

### Key Features
* **Instant Segmentation:** Predict customer groups in real-time based on profile inputs.
* **Interactive UI:** Smooth sliders and dropdowns built entirely with Streamlit.
* **Data Visualization:** Multi-dimensional cluster mapping to show where the customer sits relative to the wider database.

---

## 🧠 Model & Dataset

The underlying machine learning model is trained on the classic **Mall Customer Segmentation Dataset**, evaluating consumers across four core attributes:
* **Gender:** Male / Female
* **Age:** Continuous demographic tracking
* **Annual Income (k\$):** Financial brackets of the target audience
* **Spending Score (1-100):** A score assigned by the mall based on customer behavior and purchasing history

---

## 🛠️ Tech Stack

* **Frontend Framework:** [Streamlit](https://streamlit.io)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org)
* **Data Manipulation:** [Pandas](https://pydata.org), [NumPy](https://numpy.org)
* **Deployment:** Streamlit Community Cloud

---

## 💻 Local Installation & Setup

Follow these steps to clone the repository and run the application locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com
cd customer-segmentation-sparsh-singhal
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have a `requirements.txt` file in your directory, then run:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The application will automatically open in your local browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit UI configuration
├── data/
│   └── Mall_Customers.csv   # Training dataset
├── models/
│   └── kmeans_model.pkl     # Trained clustering model artifact
├── app.py                   # Main Streamlit application entry point
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

## 👥 Author

* **Sparsh Singhal**
* GitHub: [@sparshsinghal2025](https://github.com)
