import streamlit as st
import json
import re
import os
from typing import Dict, Any, Optional


# STREAMLIT SETUP
st.set_page_config(page_title="AskRivo - AI Real Estate Advisor for UAE Buyers", page_icon="🏠")

# --- TOKEN HANDLING (Adapted for Streamlit) ---
if "HF_TOKEN" in st.secrets:
    MY_HF_TOKEN = st.secrets["HF_TOKEN"]
elif "HF_TOKEN" in os.environ:
    MY_HF_TOKEN = os.environ["HF_TOKEN"]
else: # last fallback if token was there in streamli.screts and .env
    with st.sidebar:
        MY_HF_TOKEN = st.text_input("Hugging Face Token", type="password", help="Enter your HF API Token")

if not MY_HF_TOKEN:
    st.warning("Please provide a Hugging Face Token to proceed.")
    st.stop()

from huggingface_hub import InferenceClient
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# UAE MORTGAGE CONSTANTS (The "Hard Constraints")
CONSTANTS = {
    "MAX_LTV": 0.80,          # Max Loan-to-Value (80%)
    "MIN_DOWN_PAYMENT": 0.20, # Min Down Payment (20%)
    "UPFRONT_COST_RATE": 0.07,# 7% (4% Transfer + 2% Agency + Misc)
    "INTEREST_RATE": 0.045,   # 4.5% Standard
    "MAX_TENURE_YEARS": 25,
    "RENT_VS_BUY_THRESHOLD_LOW": 3, # < 3 years -> Rent
    "RENT_VS_BUY_THRESHOLD_HIGH": 5 # > 5 years -> Buy
}

# Initialize Client
client = InferenceClient(model=MODEL_NAME, token=MY_HF_TOKEN)

# DETERMINISTIC TOOLS (The Math)
class MortgageTools:
    """
    Constraint A: The Hallucination Trap.
    These functions handle all arithmetic. The LLM never calculates, only calls these.
    """
    
    @staticmethod
    def calculate_upfront_costs(property_price: float) -> float:
        """Returns the 7% hidden fees."""
        return property_price * CONSTANTS["UPFRONT_COST_RATE"]

    @staticmethod
    def calculate_loan_details(property_price: float, down_payment_percent: float = 0.20):
        """
        Calculates Loan Amount based on LTV constraints.
        Enforces the 80% Max LTV rule.
        """
        # Enforce Minimum Down Payment
        if down_payment_percent < CONSTANTS["MIN_DOWN_PAYMENT"]:
            effective_down_percent = CONSTANTS["MIN_DOWN_PAYMENT"]
        else:
            effective_down_percent = down_payment_percent
            
        down_payment_amount = property_price * effective_down_percent
        loan_amount = property_price - down_payment_amount
        
        return {
            "property_price": property_price,
            "down_payment_percent": effective_down_percent,
            "down_payment_amount": down_payment_amount,
            "loan_amount": loan_amount,
            "upfront_costs": MortgageTools.calculate_upfront_costs(property_price)
        }

    @staticmethod
    def calculate_emi(loan_amount: float, tenure_years: int) -> float:
        """
        Calculates Equated Monthly Installment (EMI).
        Formula: E = P * r * (1+r)^n / ((1+r)^n - 1)
        """
        if tenure_years > CONSTANTS["MAX_TENURE_YEARS"]:
            tenure_years = CONSTANTS["MAX_TENURE_YEARS"]
            
        r = CONSTANTS["INTEREST_RATE"] / 12 # Monthly interest
        n = tenure_years * 12 # Total months
        
        if n == 0: return 0
        
        emi = (loan_amount * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
        return round(emi, 2)

    @staticmethod
    def get_buy_vs_rent_advice(tenure_years: int) -> str:
        """
        Constraint C: Buy vs Rent Logic.
        """
        if tenure_years < CONSTANTS["RENT_VS_BUY_THRESHOLD_LOW"]:
            return "ADVISE_RENT"
        elif tenure_years > CONSTANTS["RENT_VS_BUY_THRESHOLD_HIGH"]:
            return "ADVISE_BUY"
        else:
            return "ADVISE_NEUTRAL"

class LLMService:
    @staticmethod
    def call_llm(prompt: str, max_tokens: int = 500) -> str:
        """Raw call to Hugging Face Inference API."""
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful JSON extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1, # Low temp for data extraction reliability
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def generate_chat_response(conversation_history: list, context_data: str = "") -> str:
        """Generates the empathetic 'Smart Friend' response."""
        
        system_prompt = (
            "You are AskRivo, an AI real estate advisor built to protect buyers in the UAE. "
            "Your core mission is UNBIASED GUIDANCE — unlike commission-driven agents, you reveal hidden costs, verify developer reliability, and calculate real financial risks. "
            "YOU DO NOT HAVE A SALES AGENDA. Your only goal is to empower people to buy homes based on hard facts and transparency, never sales pressure. "
            "TONE: Conversational, warm, and genuinely protective of the buyer's interests. Use 'I' and 'we'. Act as a trusted advisor, not a salesperson. "
            "CRITICAL: Do NOT calculate math yourself. Use the data provided in the context below. "
            "If the user can't afford something or faces financial risk, deliver the news kindly and constructively. Always suggest alternatives. "
            f"RULES: Assume 4.5% interest, Max 25 years tenure. Always warn about the hidden 7% upfront costs when discussing prices — this is a key protection. "
            "ALWAYS highlight red flags: unfavorable contract terms, unreliable developers, hidden fees, or unfavorable financing. "
            "If they're buying, ask about how long they plan to stay to give them real Buy vs Rent insight. "
            "IMPORTANT: NEVER end your response by asking to generate PDFs, reports, or complex tasks. Keep it simple and human. "
            "Your responses should feel like genuine, protective advice from someone who puts buyers first — every single time."
        )
        
        if context_data:
            system_prompt += f"\n\nCONTEXT DATA (Use these numbers, do not recalculate):\n{context_data}"

        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        try:
            resp = client.chat.completions.create(
                messages=messages,
                max_tokens=400,
                temperature=0.6, # Higher temp for creativity/empathy
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            return "I'm having a little trouble connecting to my brain right now. Can we try that again?"

    @staticmethod
    def extract_variables(user_text: str) -> dict:
        """
        Uses LLM to parse natural language into structured data.
        Constraint C: AI-Native Workflow (Using LLM for NLU).
        """
        prompt = f"""
        Extract the following entities from the text below and return ONLY a JSON object. 
        If a value is missing, use null.
        
        Entities to find:
        - property_price (number, in AED. If they say '2M', that is 2000000)
        - annual_income (number, monthly income in AED)
        - tenure_years (number, how long they plan to stay or loan length)
        - intent (string: 'buy', 'rent', 'advice', or 'general')
        
        Text: "{user_text}"
        
        JSON:
        """
        response_text = LLMService.call_llm(prompt)
        
        # Clean up the response to ensure valid JSON (LLMs sometimes add markdown)
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {}
        except:
            return {}

class MortgageAgent:
    def __init__(self):
        self.history = []
        # State variables
        self.user_data = {
            "property_price": None,
            "tenure_years": None,
            "monthly_income": None
        }
        self.lead_captured = False


# 1. Initialize Agent State
if "agent" not in st.session_state:
    st.session_state.agent = MortgageAgent()
    # Add initial greeting
    greeting = "Salam! I'm AskRivo, your unbiased AI real estate advisor. Unlike commission-driven agents, I reveal hidden costs, verify developer reliability, and calculate real financial risks. My only agenda: empower you with hard facts and transparency — never sales pressure. What property decision are you exploring today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

# Display Heading and Examples
st.title("🏠 AskRivo - AI Real Estate Advisor for UAE Buyers")
st.markdown("**Protecting your property investment with intelligent analysis and market insights.**")

with st.expander("📝 Example Inputs", expanded=False):
    st.markdown("""
    **Try asking AskRivo about:**
    1. I want to buy a 2M AED apartment, 20% down, 25 years
    2. I'm paying 7k rent. Considering buying a 1.2m apartment with 240k down. I think I'll stay 4 years.
    3. Outstanding balance 1,200,000 AED, current rate 5%, 12 years left. New rate offered 4.2% — worth switching?
    """)

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if prompt := st.chat_input("Type your message here..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Append to UI history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Append to Agent's internal history (for LLM context)
    st.session_state.agent.history.append({"role": "user", "content": prompt})

    with st.spinner("AskRivo is analyzing..."):
        agent = st.session_state.agent
        
        # 1. NLU Extraction
        extracted = LLMService.extract_variables(prompt)
        
        # Update state if new info is found
        if extracted.get('property_price'): agent.user_data['property_price'] = extracted['property_price']
        if extracted.get('tenure_years'): agent.user_data['tenure_years'] = extracted['tenure_years']
        
        # 2. Logic Check & Tool Execution
        context_info = ""
        
        # Scenario: Buy vs Rent / Affordability Check
        if agent.user_data['property_price']:
            price = agent.user_data['property_price']
            # Default to 25 years if not set
            tenure = agent.user_data['tenure_years'] if agent.user_data['tenure_years'] else 25
            
            # RUN THE TOOLS (The Math)
            loan_details = MortgageTools.calculate_loan_details(price)
            emi = MortgageTools.calculate_emi(loan_details['loan_amount'], tenure)
            
            # Logic: Buy vs Rent Advice
            advice = "N/A"
            if agent.user_data['tenure_years']:
                advice_key = MortgageTools.get_buy_vs_rent_advice(agent.user_data['tenure_years'])
                if advice_key == "ADVISE_RENT":
                    advice = "Advise RENTING (Staying < 3 years means transaction fees kill profit)."
                elif advice_key == "ADVISE_BUY":
                    advice = "Advise BUYING (Equity buildup beats rent over 5+ years)."
            
            # Construct Context for LLM
            context_info = (
                f"CALCULATION RESULTS:\n"
                f"- Property Price: {price:,.2f} AED\n"
                f"- Required Down Payment (20%): {loan_details['down_payment_amount']:,.2f} AED\n"
                f"- Loan Amount: {loan_details['loan_amount']:,.2f} AED\n"
                f"- HIDDEN UPFRONT COSTS (7%): {loan_details['upfront_costs']:,.2f} AED (Must warn user about this!)\n"
                f"- Monthly EMI (at 4.5% for {tenure} years): {emi:,.2f} AED\n"
                f"- Buy vs Rent Logic: {advice}"
            )

        # 3. Generate Response
        # Check if we should try to capture lead
        if agent.user_data['property_price'] and not agent.lead_captured:
            context_info += "\nINSTRUCTION: The user has given specific numbers. Briefly explain the costs, then say you can generate a full PDF report if they share their email."
            agent.lead_captured = True # Don't ask every single time
        
        response = LLMService.generate_chat_response(agent.history, context_info)
        
        # Append to Agent history
        agent.history.append({"role": "assistant", "content": response})

    # Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Append to UI history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar Info
st.sidebar.markdown("### 📊 Market Constraints")
st.sidebar.markdown(f"""
- **Interest Rate:** {CONSTANTS['INTEREST_RATE']*100}%
- **Max LTV:** {CONSTANTS['MAX_LTV']*100}%
- **Upfront Costs:** {CONSTANTS['UPFRONT_COST_RATE']*100}%
""")
