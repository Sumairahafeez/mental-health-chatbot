# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from train_model import StudentMentalHealthPredictor
import os
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Student Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .gemini-response {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class MentalHealthChatbot:
    def __init__(self):
        self.predictor = StudentMentalHealthPredictor()
        self.load_model()
        self.setup_gemini()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.predictor.load_model('best_mental_health_model.pkl')
            st.success(f"✅ Model loaded successfully: {self.predictor.best_model_name}")
        except FileNotFoundError:
            st.error("❌ Model not found. Please train the model first by running student_mental_health_model.py")
            st.stop()
    
    def setup_gemini(self):
        """Setup Gemini AI API"""
        # Get API key from environment variable or user input
        self.gemini_api_key = os.getenv('AIzaSyBvblnLVgJ50eZyGzaEqx-tMC94lZNRSHk')
        
        if not self.gemini_api_key:
            with st.sidebar:
                st.subheader("🔑 Gemini API Setup")
                self.gemini_api_key = st.text_input(
                    "Enter your Gemini API Key:",
                    type="password",
                    help="Get your API key from Google AI Studio"
                )
        
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
                st.sidebar.success("✅ Gemini AI connected successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Gemini AI connection failed: {str(e)}")
                self.gemini_model = None
        else:
            self.gemini_model = None
    
    def get_user_input(self):
        """Get user input through Streamlit form"""
        st.subheader("📝 Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=30, value=20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            year_of_study = st.selectbox("Year of Study", ["1", "2", "3", "4", "Graduate"])
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
            
        with col2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
            anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
            panic_attacks = st.selectbox("Do you have Panic Attacks?", ["Yes", "No"])
            treatment = st.selectbox("Did you seek any specialist for a treatment?", ["Yes", "No"])
        
        # Additional questions
        st.subheader("🎓 Academic and Social Factors")
        
        col3, col4 = st.columns(2)
        
        with col3:
            academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5)
            social_support = st.slider("Social Support (1-10)", 1, 10, 5)
            sleep_hours = st.number_input("Average Sleep Hours", min_value=3, max_value=12, value=7)
            
        with col4:
            financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 5)
            family_support = st.slider("Family Support (1-10)", 1, 10, 5)
            exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Sometimes", "Often", "Daily"])
        
        # Create input dictionary
        user_input = {
            "Age": age,
            "Gender": gender,
            "Year of Study": year_of_study,
            "CGPA": cgpa,
            "Marital status": marital_status,
            "Do you have Anxiety?": anxiety,
            "Do you have Panic attacks?": panic_attacks,
            "Did you seek any specialist for a treatment?": treatment,
            "Academic Pressure": academic_pressure,
            "Social Support": social_support,
            "Sleep Hours": sleep_hours,
            "Financial Stress": financial_stress,
            "Family Support": family_support,
            "Exercise Frequency": exercise_frequency
        }
        
        return user_input
    
    def make_prediction(self, user_input):
        """Make prediction using the trained model"""
        try:
            result = self.predictor.predict(user_input)
            return result
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def query_gemini(self, prediction_result, user_input):
        """Query Gemini AI for mental health advice"""
        if not self.gemini_model:
            return "Gemini AI is not configured. Please add your API key."
        
        # Create a comprehensive prompt
        prompt = f"""
        As a mental health counselor, provide supportive guidance based on this student's profile:
        
        Student Information:
        - Age: {user_input.get('Age', 'N/A')}
        - Academic Year: {user_input.get('Year of Study', 'N/A')}
        - CGPA: {user_input.get('CGPA', 'N/A')}
        - Has Anxiety: {user_input.get('Do you have Anxiety?', 'N/A')}
        - Has Panic Attacks: {user_input.get('Do you have Panic attacks?', 'N/A')}
        - Sleep Hours: {user_input.get('Sleep Hours', 'N/A')}
        - Academic Pressure Level: {user_input.get('Academic Pressure', 'N/A')}/10
        - Social Support Level: {user_input.get('Social Support', 'N/A')}/10
        
        Mental Health Prediction: {prediction_result['prediction']}
        Confidence: {prediction_result['confidence']:.2%}
        
        Please provide:
        1. A compassionate assessment of their mental health status
        2. Practical coping strategies and recommendations
        3. When to seek professional help
        4. Immediate self-care tips
        
        Keep the response supportive, non-judgmental, and actionable.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting Gemini response: {str(e)}"
    
    def display_results(self, prediction_result, gemini_response):
        """Display prediction results and Gemini response"""
        st.subheader("🔍 Mental Health Assessment Results")
        
        # Prediction display
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        if prediction.lower() in ['yes', 'positive', 'depressed', '1', 1]:
            st.markdown(f"""
            <div class="prediction-box negative">
                <h3>⚠️ Prediction: Possible Depression Indicators</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p>The model suggests there may be signs of depression. Please consider seeking professional support.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box positive">
                <h3>✅ Prediction: No Strong Depression Indicators</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p>The model suggests lower likelihood of depression, but mental health is complex and can change.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence visualization
        fig = go.Figure(data=go.Bar(
            x=['Confidence'],
            y=[confidence],
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Prediction Confidence",
            yaxis=dict(range=[0, 1]),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gemini AI response
        st.subheader("🤖 AI Mental Health Counselor")
        st.markdown(f"""
        <div class="gemini-response">
            <h4>💬 Personalized Guidance from Gemini AI:</h4>
            {gemini_response}
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main app function"""
        st.markdown('<h1 class="main-header">🧠 Student Mental Health Chatbot</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the Student Mental Health Assessment Tool. This AI-powered chatbot uses machine learning 
        to assess mental health indicators and provides personalized guidance through Google's Gemini AI.
        
        **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
        """)
        
        # Sidebar information
        with st.sidebar:
            st.subheader("ℹ️ About This Tool")
            st.info("""
            This chatbot:
            • Uses ML to analyze mental health patterns
            • Provides AI-powered guidance
            • Connects you with resources
            • Maintains your privacy
            """)
            
            st.subheader("🆘 Emergency Resources")
            st.error("""
            **If you're in crisis:**
            • National Suicide Prevention Lifeline: 988
            • Crisis Text Line: Text HOME to 741741
            • Emergency Services: 911
            """)
        
        # Main content
        with st.form("mental_health_assessment"):
            user_input = self.get_user_input()
            
            submitted = st.form_submit_button("🔍 Analyze Mental Health", use_container_width=True)
            
            if submitted:
                with st.spinner("Analyzing your mental health data..."):
                    # Make prediction
                    prediction_result = self.make_prediction(user_input)
                    
                    if prediction_result:
                        # Query Gemini AI
                        with st.spinner("Getting personalized guidance from AI counselor..."):
                            gemini_response = self.query_gemini(prediction_result, user_input)
                        
                        # Display results
                        self.display_results(prediction_result, gemini_response)
                        
                        # Additional resources
                        st.subheader("📚 Additional Resources")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.info("""
                            **Mental Health Apps:**
                            • Headspace
                            • Calm
                            • BetterHelp
                            • Talkspace
                            """)
                        
                        with col2:
                            st.info("""
                            **Campus Resources:**
                            • Student Counseling Center
                            • Mental Health Services
                            • Peer Support Groups
                            • Academic Advisors
                            """)
                        
                        with col3:
                            st.info("""
                            **Self-Care Tips:**
                            • Regular exercise
                            • Adequate sleep
                            • Mindfulness practice
                            • Social connections
                            """)

def main():
    """Main function to run the Streamlit app"""
    try:
        chatbot = MentalHealthChatbot()
        chatbot.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please ensure you have trained the model first by running student_mental_health_model.py")

if __name__ == "__main__":
    main()