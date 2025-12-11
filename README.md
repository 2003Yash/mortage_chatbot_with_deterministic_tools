# "The Anti-Calculator" - Conversational Mortgage AI Agent with deterministic tools

------------------------------------------------------------------------

## 1. Executive Summary

This project implements a conversational AI agent designed to guide
users through the UAE mortgage market. Unlike traditional calculators,
this agent focuses on **empathy and financial context**, acting as a
"Smart Friend" rather than a raw input-output machine.

The core engineering challenge---**preventing LLM arithmetic
hallucinations**---is solved via a **Neuro-Symbolic Architecture**. The
Large Language Model (LLM) handles Natural Language Understanding (NLU)
and generation (NLG), while a deterministic Python layer handles all
financial logic (EMI calculations, Loan-to-Value constraints, and fees).

------------------------------------------------------------------------

## 2. Architecture & Tech Stack

### **Core Components**

  --------------------------------------------------------------------------------
  Component               Technology                     Justification
  ----------------------- ------------------------------ -------------------------
  **Inference Engine**    **Meta-Llama-3-8B-Instruct**   Chosen for its high
                          (via Hugging Face API)         reasoning
                                                         capabilities-to-latency
                                                         ratio. 8B parameter size
                                                         provides sufficient
                                                         instruction-following for
                                                         JSON extraction without
                                                         the latency overhead of
                                                         70B+ models.

  **Orchestration**       **Custom Python Class          We opted for a
                          (`MortgageAgent`)**            lightweight, custom state
                                                         machine over heavy
                                                         frameworks like LangChain
                                                         to reduce dependency
                                                         bloat and maximize
                                                         control over the prompt
                                                         context window.

  **Tooling Layer**       **Native Python Static         Deterministic function
                          Methods**                      execution ensures 100%
                                                         mathematical accuracy and
                                                         zero hallucination for
                                                         critical financial data.

  **Interface**           **CLI / Colab Notebook**       Rapid prototyping
                                                         environment focused on
                                                         logic validation and
                                                         tool-use verification.
  --------------------------------------------------------------------------------

### **Data Flow Architecture**

1.  **Input Acquisition:** User query is captured.
2.  **Entity Extraction (NLU):** The LLM parses unstructured text (e.g.,
    *"2M apartment, 20k income"*) into a structured JSON payload via
    Few-Shot Prompting.
3.  **State Management:** The extracted entities update the user session
    state.
4.  **Deterministic Execution (Tool Use):** If sufficient state exists,
    the `MortgageTools` class executes financial logic (EMI formula, UAE
    Mortgage Cap rules).
5.  **Context Injection:** The precise calculation results are injected
    into the system prompt as "Context Data."
6.  **Response Generation (NLG):** The LLM generates the final response,
    strictly adhering to the injected context data while adding the
    empathetic "persona layer."

------------------------------------------------------------------------

## 3. The "Math" (Hallucination Mitigation)

To satisfy **Constraint A (The Hallucination Trap)**, no arithmetic is
performed by the LLM. We utilize a **Tool-Use Pattern** where the LLM
identifies the *intent* to calculate, and the Python runtime performs
the *execution*.

### **Code Snippet: Deterministic EMI Calculation**

``` python
class MortgageTools:
    """
    Deterministic layer to prevent arithmetic hallucinations.
    Enforces UAE Mortgage Regulations (Central Bank).
    """

    CONSTANTS = {
        "MAX_LTV": 0.80,
        "UPFRONT_COST_RATE": 0.07,
        "INTEREST_RATE": 0.045,
        "MAX_TENURE_YEARS": 25
    }

    @staticmethod
    def calculate_loan_details(property_price: float, down_payment_percent: float = 0.20):
        effective_down_percent = max(down_payment_percent, 0.20)
        down_payment_amount = property_price * effective_down_percent
        loan_amount = property_price - down_payment_amount

        return {
            "property_price": property_price,
            "down_payment_percent": effective_down_percent,
            "down_payment_amount": down_payment_amount,
            "loan_amount": loan_amount,
            "upfront_costs": property_price * MortgageTools.CONSTANTS["UPFRONT_COST_RATE"]
        }

    @staticmethod
    def calculate_emi(loan_amount: float, tenure_years: int) -> float:
        if tenure_years > MortgageTools.CONSTANTS["MAX_TENURE_YEARS"]:
            tenure_years = MortgageTools.CONSTANTS["MAX_TENURE_YEARS"]

        r = MortgageTools.CONSTANTS["INTEREST_RATE"] / 12
        n = tenure_years * 12

        if n == 0: return 0

        emi = (loan_amount * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
        return round(emi, 2)
```

------------------------------------------------------------------------

## 4. Future Improvements (Production Readiness)

-   **Guardrails:** Implement NVIDIA NeMo Guardrails or Guardrails AI to
    prevent prompt injection and ensure financial advice disclaimers are
    always present.
-   **Latency Optimization:** Switch to Streaming API (Server-Sent
    Events) to lower Time-To-First-Token (TTFT).
-   **Vector Store (RAG):** Integrate a vector database (e.g., Pinecone)
    containing the full UAE Mortgage Law documents to answer legal
    questions without hallucination.
