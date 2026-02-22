# Sentiment

**Architecture**
Flow:

User enters text in Streamlit UI
        |
Text is preprocessed
        |
Converted into numerical features (TF-IDF)
        |
ML model predicts sentiment
        |
Output displayed on UI


**Agent Explanation**

In this project, the “agent” is not a chatbot but a decision-making pipeline:

Agent Components:

Input Handler → Accepts user text

Preprocessing Agent → Cleans text (lowercase, remove noise)

Feature Agent → Converts text → vectors (TF-IDF)

Prediction Agent → Runs trained ML model

Output Agent → Displays final sentiment



**Tool Orchestration Logic**

The system follows a structured pipeline:

User Input
   ↓
Text Cleaning (Regex, Lowercase)
   ↓
Vectorization (TF-IDF)
   ↓
Model Prediction (ML Classifier)
   ↓
Result Mapping (Label → Sentiment)
   ↓
Display Output (Streamlit UI)


**Key Tools Used:

Pandas → Data handling

Scikit-learn → Model training + TF-IDF

Pickle → Model saving/loading

Streamlit → UI layer**


