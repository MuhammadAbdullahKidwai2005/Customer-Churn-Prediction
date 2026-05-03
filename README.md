<!-- 🌈 Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=200&text=Customer%20Churn%20Prediction&fontAlign=50&fontAlignY=35&color=0:F72585,100:7209B7&fontColor=ffffff&fontSize=35"/>
</p>

<!-- ⌨️ Typing Animation -->
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=F72585&size=26&center=true&vCenter=true&width=600&lines=Data+Mining+Project;Machine+Learning+Model;Streamlit+Web+App;Predict+Customer+Churn+Smartly"/>
</p>

<!-- 🏷️ Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

---

## 📊 About the Project

This project uses **Machine Learning & Data Mining** techniques to predict **customer churn** using the Telco dataset.  
It includes a Jupyter Notebook for analysis and a Streamlit web app for real-time prediction.

---

## 🎬 App Demo

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

---

## 📁 Project Structure

<pre>
Customer-Churn-Prediction
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── Telco-Customer-Churn.csv
└── CCA_Report.pdf
</pre>

---

## 🧠 Machine Learning Models

🌳 Decision Tree  
🌲 Random Forest  
📉 Logistic Regression  

---

## 📊 Workflow

<p align="center">
<img src="https://quickchart.io/graphviz?graph=digraph{rankdir=LR;Data->Cleaning->EDA->Feature_Engineering->Model_Training->Evaluation->Deployment}" />
</p>

---

## 📈 Key Insights

🔴 High monthly charges → High churn  
🟢 Long-term customers → Low churn  
🟠 Fiber users → Higher churn risk  

---

## ⚙️ Run Locally

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
