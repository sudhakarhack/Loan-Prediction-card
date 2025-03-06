import streamlit as st
import numpy as np
import joblib

# Language translations dictionary
translations = {
    "English": {
        "title": "🏦 Loan Prediction App",
        "description": "Fill in the details below to check loan eligibility.",
        "gender": "Gender",
        "gender_options": ["Male", "Female"],
        "married": "Married",
        "married_options": ["Yes", "No"],
        "education": "Education",
        "education_options": ["Graduate", "Not Graduate"],
        "self_employed": "Self Employed",
        "self_employed_options": ["Yes", "No"],
        "property_area": "Property Area",
        "property_area_options": ["Urban", "Semiurban", "Rural"],
        "applicant_income": "Applicant Income",
        "coapplicant_income": "Coapplicant Income",
        "property_value": "Property Value (in Lakhs)",
        "loan_term": "Loan Term (in months)",
        "check_button": "Check Eligibility",
        "loan_approved": "✅ Loan Sanctioned! (Approval Probability: {:.2f})",
        "loan_rejected": "❌ Loan Not Sanctioned (Approval Probability: {:.2f})"
    },
    "Telugu": {
        "title": "🏦 ఋణ అంచనా అప్లికేషన్",
        "description": "కింద ఉన్న వివరాలు పూరించండి మరియు ఋణ అర్హతను తనిఖీ చేయండి.",
        "gender": "లింగం",
        "gender_options": ["పురుషుడు", "స్త్రీ"],
        "married": "వివాహితా స్థితి",
        "married_options": ["అవును", "కాదు"],
        "education": "విద్య",
        "education_options": ["బిరుదు", "బిరుదు లేని"],
        "self_employed": "స్వయం ఉపాధి",
        "self_employed_options": ["అవును", "కాదు"],
        "property_area": "ప్రాపర్టీ ప్రాంతం",
        "property_area_options": ["నగరం", "అర్బన్", "గ్రామం"],
        "applicant_income": "దరఖాస్తుదారు ఆదాయం",
        "coapplicant_income": "సహ దరఖాస్తుదారు ఆదాయం",
        "property_value": "ప్రాపర్టీ విలువ (లక్షల్లో)",
        "loan_term": "ఋణ కాలం (నెలల్లో)",
        "check_button": "అర్హతను తనిఖీ చేయండి",
        "loan_approved": "✅ ఋణ మంజూరైంది! (అనుమతించే అవకాశం: {:.2f})",
        "loan_rejected": "❌ ఋణ తిరస్కరించబడింది (అనుమతించే అవకాశం: {:.2f})"
    },
    "Hindi": {
        "title": "🏦 लोन भविष्यवाणी ऐप",
        "description": "नीचे दिए गए विवरण भरें और लोन पात्रता की जांच करें।",
        "gender": "लिंग",
        "gender_options": ["पुरुष", "महिला"],
        "married": "वैवाहिक स्थिति",
        "married_options": ["हाँ", "नहीं"],
        "education": "शिक्षा",
        "education_options": ["स्नातक", "गैर-स्नातक"],
        "self_employed": "स्वरोजगार",
        "self_employed_options": ["हाँ", "नहीं"],
        "property_area": "संपत्ति क्षेत्र",
        "property_area_options": ["शहरी", "अर्ध-शहरी", "ग्रामीण"],
        "applicant_income": "आवेदक की आय",
        "coapplicant_income": "सह-आवेदक की आय",
        "property_value": "संपत्ति मूल्य (लाखों में)",
        "loan_term": "लोन अवधि (महीनों में)",
        "check_button": "पात्रता जांचें",
        "loan_approved": "✅ लोन स्वीकृत! (स्वीकृति संभावना: {:.2f})",
        "loan_rejected": "❌ लोन अस्वीकृत (स्वीकृति संभावना: {:.2f})"
    },
    "Tamil": {
        "title": "🏦 கடன் கணிப்பு செயலி",
        "description": "கீழே உள்ள விவரங்களை நிரப்பி, கடன் தகுதியை சரிபார்க்கவும்.",
        "gender": "பாலினம்",
        "gender_options": ["ஆண்", "பெண்"],
        "married": "திருமண நிலை",
        "married_options": ["ஆம்", "இல்லை"],
        "education": "கல்வி",
        "education_options": ["பட்டதாரி", "பட்டமில்லாதவர்"],
        "self_employed": "சுய தொழில்",
        "self_employed_options": ["ஆம்", "இல்லை"],
        "property_area": "சொத்து பகுதி",
        "property_area_options": ["நகரம்", "அரிமுக நகரம்", "கிராமம்"],
        "applicant_income": "விண்ணப்பதாரர் வருமானம்",
        "coapplicant_income": "சக விண்ணப்பதாரர் வருமானம்",
        "property_value": "சொத்து மதிப்பு (லட்சங்களில்)",
        "loan_term": "கடன் காலம் (மாதங்களில்)",
        "check_button": "தகுதியைச் சரிபார்க்கவும்",
        "loan_approved": "✅ கடன் ஒப்புதல்! (ஒப்புதல் சாத்தியம்: {:.2f})",
        "loan_rejected": "❌ கடன் நிராகரிக்கப்பட்டது (ஒப்புதல் சாத்தியம்: {:.2f})"
    }
}

# Load ML model if available
def load_model():
    try:
        return joblib.load("loan_model.pkl")
    except:
        return None

# Predict eligibility (dummy rule if model not found)
def predict_loan(inputs):
    model = load_model()
    if model:
        pred = model.predict([inputs])[0]
        prob = model.predict_proba([inputs])[0][1]
    else:
        prob = np.random.uniform(0.6, 0.9) if inputs[0] == 1 and inputs[4] > 50000 else np.random.uniform(0.2, 0.5)
        pred = "Loan Sanctioned" if prob > 0.5 else "Loan Not Sanctioned"
    return pred, prob

# UI starts
language = st.selectbox("Select Language", list(translations.keys()))
t = translations[language]

st.title(t["title"])
st.write(t["description"])

gender = st.selectbox(t["gender"], t["gender_options"])
married = st.selectbox(t["married"], t["married_options"])
education = st.selectbox(t["education"], t["education_options"])
self_employed = st.selectbox(t["self_employed"], t["self_employed_options"])
property_area = st.selectbox(t["property_area"], t["property_area_options"])
applicant_income = st.number_input(t["applicant_income"], min_value=0, value=60000)
coapplicant_income = st.number_input(t["coapplicant_income"], min_value=0, value=40000)
property_value = st.number_input(t["property_value"], min_value=0, value=80000)
loan_term = st.number_input(t["loan_term"], min_value=1, value=360)

# Mapping categorical data to numbers (model input format)
gender = 1 if gender in ["Male", "పురుషుడు", "पुरुष", "ஆண்"] else 0
married = 1 if married in ["Yes", "అవును", "हाँ", "ஆம்"] else 0
education = 1 if education in ["Graduate", "బిరుదు", "स्नातक", "பட்டதாரி"] else 0
self_employed = 1 if self_employed in ["Yes", "అవును", "हाँ", "ஆம்"] else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0,
                 "నగరం": 2, "అర్బన్": 1, "గ్రామం": 0,
                 "शहरी": 2, "अर्ध-शहरी": 1, "ग्रामीण": 0,
                 "நகரம்": 2, "அரிமுக நகரம்": 1, "கிராமம்": 0}[property_area]

inputs = [gender, married, education, self_employed, applicant_income, coapplicant_income, property_value, loan_term, property_area]

if st.button(t["check_button"]):
    pred, prob = predict_loan(inputs)
    st.success(t["loan_approved"].format(prob) if "Sanctioned" in pred else t["loan_rejected"].format(prob))
