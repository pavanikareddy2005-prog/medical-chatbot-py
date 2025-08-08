import streamlit as st
import pandas as pd
import re
from translate import Translator
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Define the medical knowledge base (using the expanded version)
medical_knowledge_base = {
    'glucose high': {
        'explanation': 'Elevated blood sugar levels, potentially indicating prediabetes or diabetes.',
        'potential_causes': 'Diet high in sugar and carbohydrates, lack of physical activity, family history, certain medications.',
        'lifestyle_recommendations': {
            'diet': 'Balanced diet with controlled carbohydrate intake, focus on whole grains, fruits, vegetables, and lean proteins. Limit sugary drinks and processed foods.',
            'exercise': 'Regular physical activity, aiming for at least 150 minutes of moderate-intensity aerobic exercise per week, plus muscle-strengthening activities.'
        },
        'potential_risks': 'Increased risk of heart disease, stroke, kidney damage, nerve damage, and eye problems if not managed.'
    },
    'small nodule in the upper lobe': {
        'explanation': 'A small spot or lesion found in the upper section of the lung. Often benign, but requires follow-up to monitor for changes.',
        'potential_causes': 'Previous infections, inflammation, benign tumors, or potentially early signs of cancer (less common for small nodules).',
        'lifestyle_recommendations': {
            'diet': 'Maintain a healthy diet to support overall lung health.',
            'exercise': 'Regular exercise can improve lung function and overall health.'
        },
        'potential_risks': 'While often harmless, there is a small risk the nodule could be cancerous or grow over time, necessitating monitoring.'
    },
    'benign tissue': {
        'explanation': 'Non-cancerous tissue. Indicates that the examined sample does not show signs of malignancy.',
        'potential_causes': 'Various non-cancerous conditions can lead to benign tissue changes.',
        'lifestyle_recommendations': {
            'diet': 'Maintain a healthy diet for overall well-being.',
            'exercise': 'Regular exercise supports general health.'
        },
        'potential_risks': 'Generally, there are no significant risks associated with benign tissue itself, although the underlying cause of the tissue change might require attention.'
    },
    'cholesterol high': {
        'explanation': 'High levels of cholesterol in the blood, which can increase the risk of heart disease.',
        'potential_causes': 'Diet high in saturated and trans fats, lack of exercise, smoking, genetics.',
        'lifestyle_recommendations': {
            'diet': 'Diet low in saturated and trans fats, rich in fruits, vegetables, whole grains, and lean protein.',
            'exercise': 'Regular aerobic exercise.'
        },
        'potential_risks': 'Increased risk of heart attack and stroke.'
    },
    'triglycerides high': {
        'explanation': 'High levels of triglycerides, a type of fat in your blood, which can increase the risk of heart disease.',
        'potential_causes': 'Obesity, physical inactivity, smoking, excessive alcohol intake, high-sugar diet, certain medical conditions.',
        'lifestyle_recommendations': {
            'diet': 'Limit sugary foods and refined grains, choose healthier fats, limit alcohol.',
            'exercise': 'Regular aerobic exercise.'
        },
        'potential_risks': 'Increased risk of heart attack and stroke, especially when combined with high LDL cholesterol or low HDL cholesterol.'
    },
    'calcified granuloma': {
        'explanation': 'A small area of inflammation that has healed and hardened with calcium. It is usually benign and often a sign of a past infection like tuberculosis or histoplasmosis.',
        'potential_causes': 'Previous infections (e.g., tuberculosis, histoplasmosis).',
        'lifestyle_recommendations': {
            'diet': 'Maintain a healthy diet.',
            'exercise': 'Regular exercise.'
        },
        'potential_risks': 'Generally considered harmless and does not pose a significant health risk, although follow-up may be recommended to confirm stability.'
    },
     'benign nevus': {
        'explanation': 'A common mole that is non-cancerous.',
        'potential_causes': 'Genetics, sun exposure.',
        'lifestyle_recommendations': {
            'diet': 'Maintain a healthy diet.',
            'exercise': 'Regular exercise.'
        },
        'potential_risks': 'Very low risk; however, changes in size, shape, color, or texture should be monitored by a dermatologist as they could indicate melanoma (skin cancer).'
    },
    'hemoglobin a1c high': {
        'explanation': 'Hemoglobin A1c is a test that measures your average blood sugar levels over the past 2-3 months. A high A1c indicates elevated average blood sugar, common in individuals with diabetes or prediabetes.',
        'potential_causes': 'Diabetes, prediabetes, inadequate management of existing diabetes.',
        'lifestyle_recommendations': {
            'diet': 'Balanced diet with controlled carbohydrate intake, focus on whole grains, fruits, vegetables, and lean proteins. Limit sugary drinks and processed foods.',
            'exercise': 'Regular physical activity, aiming for at least 150 minutes of moderate-intensity aerobic exercise per week, plus muscle-strengthening activities.'
        },
        'potential_risks': 'Increased risk of long-term diabetes complications including heart disease, stroke, kidney damage, nerve damage, and eye problems.'
    },
     'ldl cholesterol high': {
        'explanation': 'High levels of Low-Density Lipoprotein (LDL) cholesterol, often called "bad" cholesterol. High LDL contributes to plaque buildup in arteries, increasing the risk of heart disease and stroke.',
        'potential_causes': 'Diet high in saturated and trans fats, obesity, lack of exercise, smoking, genetics.',
        'lifestyle_recommendations': {
            'diet': 'Diet low in saturated and trans fats, rich in soluble fiber, fruits, vegetables, whole grains, and lean protein.',
            'exercise': 'Regular aerobic exercise.'
        },
        'potential_risks': 'Increased risk of heart attack, stroke, and peripheral artery disease.'
    }
}

# Load the medical NER model and tokenizer
@st.cache_resource
def load_ner_model():
    model_name_medical_ner_public = "samrawal/bert-base-uncased_clinical-ner"
    tokenizer_medical_ner_public = AutoTokenizer.from_pretrained(model_name_medical_ner_public)
    model_medical_ner_public = AutoModelForTokenClassification.from_pretrained(model_name_medical_ner_public)
    ner_pipeline_medical_public = pipeline("ner", model=model_medical_ner_public, tokenizer=tokenizer_medical_ner_public, aggregation_strategy="simple")
    return ner_pipeline_medical_public

ner_pipeline_medical_public = load_ner_model()

# Instantiate a translator object
@st.cache_resource
def load_translator():
    return Translator(to_lang="es")

translator = load_translator()

# Refined anonymization and cleaning functions
def anonymize_report_refined(report):
    """Refined function to anonymize sensitive information in the report."""
    if not isinstance(report, str):
        return ""
    report = re.sub(r'Patient Name: .*', 'Patient Name: [ANONYMIZED]', report)
    report = re.sub(r'DOB: .*', 'DOB: [ANONYMIZED]', report)
    report = re.sub(r'Patient ID: .*', 'Patient ID: [ANONYMIZED]', report)
    report = re.sub(r'Patient: .*', 'Patient: [ANONYMIZED]', report)
    report = re.sub(r'Accession: .*', 'Accession: [ANONYMIZED]', report)
    report = re.sub(r'Medical Record Number: .*', 'Medical Record Number: [ANONYMIZED]', report)
    report = re.sub(r'\d{3}-\d{2}-\d{4}', '[ANONYMIZED]', report) # Basic SSN pattern
    report = re.sub(r'\(\d{3}\) \d{3}-\d{4}', '[ANONYMIZED]', report) # Basic phone pattern
    return report

def clean_text_refined(text):
    """Refined function to clean text while preserving numbers and relevant symbols."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s\/\:\.\,]', '', text)
    text = text.lower()
    return text

# Function to extract medical terms using the NER pipeline
def extract_medical_terms(text, ner_pipeline):
    """Extracts medical terms using a medical NER pipeline."""
    if not isinstance(text, str) or text.strip() == "":
        return []
    entities = ner_pipeline(text)
    return [entity['word'] for entity in entities if 'word' in entity]

# Refined generate_explanation with enhanced flexible matching and basic term cleaning
def generate_explanation_refined(extracted_terms, knowledge_base):
    """Generates a comprehensive explanation with enhanced flexible matching and basic term cleaning."""
    explanations = []
    if not extracted_terms:
        return "No specific medical terms were identified in this report."

    processed_terms = []
    for term in extracted_terms:
        cleaned_term = term.replace('##', '').strip()
        if cleaned_term:
            processed_terms.append(cleaned_term)

    if not processed_terms:
         return "No specific medical terms were identified in this report after cleaning."

    normalized_knowledge_base = {key.lower(): value for key, value in knowledge_base.items()}

    matched_keys = set()

    for term in processed_terms:
        normalized_term = term.lower()

        if normalized_term in normalized_knowledge_base:
            matched_keys.add(normalized_term)
            continue

        found_match = False
        for kb_key in normalized_knowledge_base.keys():
            if re.search(r'\b' + re.escape(kb_key) + r'\b', normalized_term):
                 matched_keys.add(kb_key)
                 found_match = True
                 break

        if found_match:
            continue

        for kb_key in normalized_knowledge_base.keys():
             if re.search(r'\b' + re.escape(normalized_term) + r'\b', kb_key):
                 matched_keys.add(kb_key)
                 break

    for key in matched_keys:
         if 'explanation' in normalized_knowledge_base[key]:
             explanations.append(normalized_knowledge_base[key]['explanation'])

    if not explanations:
        return "While terms were extracted, no specific information was found in the knowledge base for these terms."
    else:
        unique_explanations = list(dict.fromkeys(explanations))
        return " ".join(unique_explanations)

# Refined generate_recommendations with enhanced flexible matching and basic term cleaning
def generate_recommendations_refined(extracted_terms, knowledge_base):
    """Generates personalized lifestyle recommendations with enhanced flexible matching and basic term cleaning."""
    diet_recommendations = set()
    exercise_recommendations = set()

    if not extracted_terms:
        return "No specific medical terms were identified to generate personalized recommendations."

    processed_terms = []
    for term in extracted_terms:
        cleaned_term = term.replace('##', '').strip()
        if cleaned_term:
            processed_terms.append(cleaned_term)

    if not processed_terms:
        return "No specific medical terms were identified to generate personalized recommendations after cleaning."

    normalized_knowledge_base = {key.lower(): value for key, value in knowledge_base.items()}

    matched_keys = set()

    for term in processed_terms:
        normalized_term = term.lower()

        if normalized_term in normalized_knowledge_base:
            matched_keys.add(normalized_term)
            continue

        found_match = False
        for kb_key in normalized_knowledge_base.keys():
            if re.search(r'\b' + re.escape(kb_key) + r'\b', normalized_term):
                 matched_keys.add(kb_key)
                 found_match = True
                 break

        if found_match:
            continue

        for kb_key in normalized_knowledge_base.keys():
             if re.search(r'\b' + re.escape(normalized_term) + r'\b', kb_key):
                 matched_keys.add(kb_key)
                 break

    for key in matched_keys:
        if 'lifestyle_recommendations' in normalized_knowledge_base[key]:
            recommendations = normalized_knowledge_base[key]['lifestyle_recommendations']
            if 'diet' in recommendations:
                diet_recommendations.add(recommendations['diet'])
            if 'exercise' in recommendations:
                exercise_recommendations.add(recommendations['exercise'])

    if not diet_recommendations and not exercise_recommendations:
        return "While terms were extracted, no specific lifestyle recommendations were found in the knowledge base for these terms."
    else:
        recommendation_text = ""
        if diet_recommendations:
            recommendation_text += "Diet Recommendations: " + " ".join(list(diet_recommendations)) + "\n"
        if exercise_recommendations:
            recommendation_text += "Exercise Recommendations: " + " ".join(list(exercise_recommendations))
        return recommendation_text.strip()

# Function to translate text (using the loaded translator)
def translate_text(text, translator_obj):
    """Translates text using a provided translator object."""
    if not isinstance(text, str) or text.strip() == "":
        return text
    try:
        return translator_obj.translate(text)
    except Exception as e:
        # print(f"Translation error: {e}") # Avoid printing in Streamlit
        return f"Translation unavailable: {text}"

# Refined chatbot response logic
def simulate_chatbot_response_refined(user_input, processed_data, knowledge_base):
    """Simulates a more sophisticated chatbot response."""
    prompt = user_input.lower()
    response = "I am a prototype chatbot. "

    if not processed_data:
        return response + "Please upload a medical report first so I can provide information."

    extracted_terms = processed_data.get("Extracted Medical Terms", [])
    generated_explanation = processed_data.get("Generated Explanation (English)", "")
    generated_recommendations = processed_data.get("Generated Recommendations (English)", "")


    if any(keyword in prompt for keyword in ["what does this mean", "explanation", "interpret", "understand"]):
        response += "Here is a simple explanation based on the report: " + generated_explanation
    elif any(keyword in prompt for keyword in ["recommendations", "advice", "what should i do", "lifestyle"]):
        response += "Here are some lifestyle recommendations based on the report: " + generated_recommendations
    elif any(keyword in prompt for keyword in ["medical terms", "terms explained", "what are the terms"]):
         if extracted_terms:
              response += "The key medical terms identified in the report are: " + ", ".join(extracted_terms) + "."
         else:
              response += "No specific medical terms were identified in this report."
    elif any(keyword in prompt for keyword in ["risks", "potential risks", "dangers"]):
        risks = set()
        normalized_knowledge_base = {key.lower(): value for key, value in knowledge_base.items()}
        processed_terms = [term.replace('##', '').strip().lower() for term in extracted_terms]

        for term in processed_terms:
             for kb_key in normalized_knowledge_base.keys():
                 if re.search(r'\b' + re.escape(kb_key) + r'\b', term) or re.search(r'\b' + re.escape(term) + r'\b', kb_key):
                      if 'potential_risks' in normalized_knowledge_base[kb_key]:
                           risks.add(normalized_knowledge_base[kb_key]['potential_risks'])
             if risks:
                  break

        if risks:
            response += "Based on the medical terms in the report, here are some potential risks to be aware of: " + " ".join(list(risks))
        else:
            response += "Based on the terms identified, I don't have specific risk information in my knowledge base for this report."

    elif any(keyword in prompt for keyword in ["diet", "food"]):
        diet_recs = set()
        normalized_knowledge_base = {key.lower(): value for key, value in knowledge_base.items()}
        processed_terms = [term.replace('##', '').strip().lower() for term in extracted_terms]

        for term in processed_terms:
             for kb_key in normalized_knowledge_base.keys():
                 if re.search(r'\b' + re.escape(kb_key) + r'\b', term) or re.search(r'\b' + re.escape(term) + r'\b', kb_key):
                      if 'lifestyle_recommendations' in normalized_knowledge_base[kb_key] and 'diet' in normalized_knowledge_base[kb_key]['lifestyle_recommendations']:
                           diet_recs.add(normalized_knowledge_base[kb_key]['lifestyle_recommendations']['diet'])
             if diet_recs:
                  break

        if diet_recs:
             response += "Regarding diet, based on the report: " + " ".join(list(diet_recs))
        else:
             response += "Based on the terms identified, I don't have specific diet recommendations in my knowledge base for this report."

    elif any(keyword in prompt for keyword in ["exercise", "physical activity"]):
        exercise_recs = set()
        normalized_knowledge_base = {key.lower(): value for key, value in knowledge_base.items()}
        processed_terms = [term.replace('##', '').strip().lower() for term in extracted_terms]

        for term in processed_terms:
             for kb_key in normalized_knowledge_base.keys():
                 if re.search(r'\b' + re.escape(kb_key) + r'\b', term) or re.search(r'\b' + re.escape(term) + r'\b', kb_key):
                      if 'lifestyle_recommendations' in normalized_knowledge_base[kb_key] and 'exercise' in normalized_knowledge_base[kb_key]['lifestyle_recommendations']:
                           exercise_recs.add(normalized_knowledge_base[kb_key]['lifestyle_recommendations']['exercise'])
             if exercise_recs:
                  break

        if exercise_recs:
             response += "Regarding exercise, based on the report: " + " ".join(list(exercise_recs))
        else:
             response += "Based on the terms identified, I don't have specific exercise recommendations in my knowledge base for this report."

    else:
        response += "You asked: '" + user_input + "'. I can provide the explanation, recommendations, or extracted terms from the report, and some specific details like risks, diet, or exercise if available in my knowledge base."

    return response


# --- Disclaimer Statements ---
DISCLAIMER_TITLE = "Important Disclaimer"
DISCLAIMER_CONTENT = """
This AI Medical Report Chatbot is a **prototype** developed for informational and demonstration purposes only.
It is **not intended to provide medical advice, diagnosis, or treatment**.
The information generated by this chatbot is based on a limited knowledge base and automated processing.
**Always consult with a qualified healthcare professional** for any health concerns, medical conditions,
or before making any decisions related to your health or treatment.
Do not disregard professional medical advice or delay seeking it because of information from this chatbot.

**Data Privacy:** The medical report you upload is processed in memory and **anonymized immediately**.
**Sensitive information is not stored** or used for training purposes.
"""

# Streamlit App Title
st.title("Medical Report Chatbot Prototype")

# Display Disclaimers Prominently
st.warning(DISCLAIMER_TITLE)
st.info(DISCLAIMER_CONTENT)

st.write("Upload a medical report (text file) and interact with the chatbot.")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

processed_data = None

if uploaded_file is not None:
    # Read the uploaded file
    medical_report = uploaded_file.getvalue().decode("utf-8")

    # Process the report using refined functions
    anonymized_report = anonymize_report_refined(medical_report)
    cleaned_report = clean_text_refined(anonymized_report)
    extracted_terms = extract_medical_terms(cleaned_report, ner_pipeline_medical_public)
    # Use the refined generate functions
    generated_explanation = generate_explanation_refined(extracted_terms, medical_knowledge_base)
    generated_recommendations = generate_recommendations_refined(extracted_terms, medical_knowledge_base)
    generated_explanation_es = translate_text(generated_explanation, translator)
    generated_recommendations_es = translate_text(generated_recommendations, translator)


    processed_data = {
        "Original Report": medical_report,
        "Anonymized Report": anonymized_report,
        "Extracted Medical Terms": extracted_terms,
        "Generated Explanation (English)": generated_explanation,
        "Generated Explanation (Spanish)": generated_explanation_es,
        "Generated Recommendations (English)": generated_recommendations,
        "Generated Recommendations (Spanish)": generated_recommendations_es,
    }

    st.subheader("Processed Report Information:")
    st.write("**Original Report:**")
    st.text(processed_data["Original Report"])
    st.write("**Anonymized Report:**")
    st.text(processed_data["Anonymized Report"])
    st.write("**Extracted Medical Terms:**")
    st.write(processed_data["Extracted Medical Terms"])
    st.write("**Generated Explanation (English):**")
    st.write(processed_data["Generated Explanation (English)"])
    st.write("**Generated Explanation (Spanish):**")
    st.write(processed_data["Generated Explanation (Spanish)"])
    st.write("**Generated Recommendations (English):**")
    st.write(processed_data["Generated Recommendations (English)"])
    st.write("**Generated Recommendations (Spanish):**")
    st.write(processed_data["Generated Recommendations (Spanish)"])

# Basic Chat Interface
st.subheader("Chat with the Medical Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the report..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Use the refined chatbot simulation function
        simulated_response = simulate_chatbot_response_refined(prompt, processed_data, medical_knowledge_base)
        st.markdown(simulated_response)
        st.session_state.messages.append({"role": "assistant", "content": simulated_response})