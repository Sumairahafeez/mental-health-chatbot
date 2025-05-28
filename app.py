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
        color: #0d47a1;
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
        self.gemini_api_key = ("AIzaSyBVw4mFEAbAuTED533jzu5cyA6DuEwOip0")
        
        # if not self.gemini_api_key:
        #     with st.sidebar:
        #         st.subheader("🔑 Gemini API Setup")
        #         self.gemini_api_key = st.text_input(
        #             "Enter your Gemini API Key:",
        #             type="password",
        #             help="Get your API key from Google AI Studio"
        #         )
        
        if self.gemini_api_key:
            try:
                genai.configure(api_key='AIzaSyBVw4mFEAbAuTED533jzu5cyA6DuEwOip0')
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
                st.sidebar.success("✅ Gemini AI connected successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Gemini AI connection failed: {str(e)}")
                self.gemini_model = None
        else:
            self.gemini_model = None
    
    def get_user_input(self):
        """Get user input through Streamlit form using exact dataset column names"""
        st.subheader("📝 Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic Demographics
            age = st.number_input("What is your age?", min_value=16, max_value=30, value=20)
            gender = st.selectbox("Choose your gender", ["Male", "Female"])
            course = st.selectbox("What is your course?", 
                                ["Engineering", "Islamic education", "BIT", "Laws", "Mathematics", 
                                 "pendidikan islam", "BCS", "Human Sciences", "KIRKHS", "BENL",
                                 "Banking Studies", "Business Administration", "Psychology", "Other"])
            
        with col2:
            # Academic Information
            year = st.selectbox("Your current year of Study", ["year 1", "year 2", "year 3", "year 4"])
            cgpa = st.number_input("What is your CGPA?", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            marital_status = st.selectbox("Marital status", ["No", "Yes"])
        
        # Mental Health Questions
        st.subheader("🧠 Mental Health Assessment")
        
        col3, col4 = st.columns(2)
        
        with col3:
            anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
            panic_attack = st.selectbox("Do you have Panic attack?", ["No", "Yes"])
            treatment = st.selectbox("Did you seek any specialist for a treatment?", ["No", "Yes"])
            
        with col4:
            # Note: We'll create synthetic timestamp as it's likely not user input
            # but might be needed for the model
            import datetime
            timestamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        # Create input dictionary with exact column names from the dataset
        user_input = {
            "Timestamp": timestamp,
            "Choose your gender": gender,
            "Age": age,
            "What is your course?": course,
            "Your current year of Study": year,
            "What is your CGPA?": cgpa,
            "Marital status": marital_status,
            "Do you have Anxiety?": anxiety,
            "Do you have Panic attack?": panic_attack,
            "Did you seek any specialist for a treatment?": treatment
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
    def get_fallback_guidance(self, prediction_result, user_input):
        """Provide comprehensive guidance when Gemini AI is unavailable"""
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        # Analyze user profile
        has_anxiety = user_input.get('Do you have Anxiety?', 'No') == 'Yes'
        has_panic = user_input.get('Do you have Panic attack?', 'No') == 'Yes'
        sought_treatment = user_input.get('Did you seek any specialist for a treatment?', 'No') == 'Yes'
        cgpa = user_input.get('What is your CGPA?', 3.0)
        year = user_input.get('Your current year of Study', 'year 1')
        
        # Generate personalized guidance
        guidance = f"""
## 🧠 Mental Health Assessment & Guidance

### Your Profile Summary:
- **Prediction**: {prediction} (Confidence: {confidence:.1%})
- **Current Symptoms**: {'Anxiety reported' if has_anxiety else 'No anxiety reported'}{', Panic attacks reported' if has_panic else ''}
- **Treatment History**: {'Previously sought professional help' if sought_treatment else 'No previous professional treatment'}
- **Academic Status**: {year.title()}, CGPA: {cgpa}

### 💡 Personalized Recommendations:

"""
        
        # Risk-based recommendations
        if str(prediction).lower() in ['yes', '1', 'positive'] or has_anxiety or has_panic:
            guidance += """
#### 🚨 Immediate Action Items:
- **Consider Professional Help**: Given the indicators, speaking with a counselor is strongly recommended
- **Campus Resources**: Visit your university's counseling center - most offer free services
- **Crisis Support**: If you're having thoughts of self-harm, contact emergency services immediately

#### 🛡️ Coping Strategies:
- **Breathing Exercises**: Practice 4-7-8 breathing (inhale 4, hold 7, exhale 8)
- **Grounding Technique**: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste
- **Daily Structure**: Maintain consistent sleep and meal schedules
"""
        else:
            guidance += """
#### ✅ Positive Indicators:
- Your responses suggest lower risk for depression
- Continue maintaining good mental health practices
- Stay aware of changes in your mood or stress levels

#### 🌟 Preventive Care:
- **Maintain Balance**: Keep up healthy study-life balance
- **Stay Connected**: Maintain social relationships and support networks
- **Monitor Changes**: Be aware of any shifts in mood, sleep, or appetite
"""
        
        # Academic-specific advice
        if float(cgpa) < 2.5:
            guidance += """
#### 📚 Academic Support:
- **Academic Counseling**: Meet with an academic advisor to discuss study strategies
- **Study Groups**: Join or form study groups with classmates
- **Time Management**: Use planners or apps to organize assignments and deadlines
- **Tutoring Services**: Take advantage of campus tutoring resources
"""
        
        # General wellness tips
        guidance += """
#### 🌈 General Wellness Tips:
- **Exercise**: Aim for 30 minutes of physical activity daily - even walking helps
- **Sleep Hygiene**: Maintain 7-9 hours of sleep with consistent bedtime
- **Nutrition**: Eat regular, balanced meals and stay hydrated
- **Mindfulness**: Try meditation apps like Headspace or Calm
- **Social Connection**: Stay in touch with friends and family regularly

#### 📱 Helpful Apps & Resources:
- **Crisis Text Line**: Text HOME to 741741
- **National Suicide Prevention Lifeline**: 988
- **Mental Health Apps**: Headspace, Calm, BetterHelp, Talkspace
- **Academic Support**: Khan Academy, Coursera for skill building

#### ⚠️ When to Seek Immediate Help:
- Persistent sadness lasting more than two weeks
- Thoughts of self-harm or suicide
- Inability to perform daily activities
- Significant changes in sleep, appetite, or energy
- Panic attacks becoming more frequent or severe

---
*Remember: This assessment is for educational purposes only. Always consult with healthcare professionals for proper mental health evaluation and treatment.*
"""
        
        return guidance
    def query_gemini(self, prediction_result, user_input):
        """Query Gemini AI for mental health advice"""
        if not self.gemini_model:
            return "Gemini AI is not configured. Please add your API key."
        
        # Create a comprehensive prompt using correct column names
        prompt = f"""
        As a mental health counselor, provide supportive guidance based on this student's profile:
        
        Student Information:
        - Age: {user_input.get('Age', 'N/A')}
        - Gender: {user_input.get('Choose your gender', 'N/A')}
        - Course: {user_input.get('What is your course?', 'N/A')}
        - Year of Study: {user_input.get('Your current year of Study', 'N/A')}
        - CGPA: {user_input.get('What is your CGPA?', 'N/A')}
        - Marital Status: {user_input.get('Marital status', 'N/A')}
        - Has Anxiety: {user_input.get('Do you have Anxiety?', 'N/A')}
        - Has Panic Attacks: {user_input.get('Do you have Panic attack?', 'N/A')}
        - Sought Treatment: {user_input.get('Did you seek any specialist for a treatment?', 'N/A')}
        
        Mental Health Prediction: {prediction_result['prediction']}
        Confidence: {prediction_result['confidence']:.2%}
        
        Please provide:
        1. A compassionate assessment of their mental health status
        2. Practical coping strategies and recommendations specific to their academic situation
        3. When to seek professional help
        4. Immediate self-care tips for students
        5. Academic stress management techniques
        
        Keep the response supportive, non-judgmental, and actionable for a university student.
        """
        print("prompt:", prompt)    
        if not prompt.strip() or prediction_result is None:
            return "No prompt or prediction provided."
        try:
            genai.configure(api_key="AIzaSyBq1zON_vbXdkKWmJcr_oc59NnxbSg76CM")
            model = genai.GenerativeModel('gemini-2.0-flash')
            print("model:", model)
            response = model.generate_content(contents = [prompt])
            if response.text is None:
                return f"Gemini AI quota reached today. Showing built-in mental health guidance."
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            # return f"Error getting Gemini response: {str(e)}"
            if '429' in error_msg or 'quota' in error_msg or 'exceeded' in error_msg:
                    st.warning("⚠️ Gemini AI quota exceeded. Providing comprehensive guidance using built-in recommendations.")
                    return self.get_fallback_guidance(prediction_result, user_input)
            elif 'api key' in error_msg or 'authentication' in error_msg:
                    st.error("❌ Gemini API authentication failed. Please check your API key.")
                    return self.get_fallback_guidance(prediction_result, user_input)
            else:
                    st.warning(f"⚠️ Gemini AI temporarily unavailable: {str(e)}")
                    return self.get_fallback_guidance(prediction_result, user_input)
    
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