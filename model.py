import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class LLMModel:
    def __init__(self):
        """Initialize the LLM model with configurations based on environment variables"""
        self.model_type = os.getenv("MODEL_TYPE", "gemini")  # 'local' or 'gemini'
        self.model = None
        self.tokenizer = None
        self.device = None  # Only set for local models
        self.setup_model()

    def setup_model(self):
        """Set up the model based on self.model_type"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        try:
            genai.configure(api_key=api_key)
            # Choose a suitable Gemini model, e.g., 'gemini-1.5-flash' or 'gemini-pro'
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini model configured successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini: {str(e)}")

    def generate_response(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """
        Generate response from the configured model for a given prompt

        Args:
            prompt (str): Input prompt for the model
            max_new_tokens (int): Maximum number of tokens to generate (used by local model)

        Returns:
            str: Generated response
        """
        try:
                response = self.model.generate_content(prompt)
                # Add more robust error checking if needed based on Gemini response structure
                if response.parts:
                    return response.text.strip()
                else:
                    # Handle cases where the response might be blocked or empty
                    safety_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                    finish_reason = response.candidates[0].finish_reason if response.candidates else 'N/A'
                    print(f"Warning: Gemini response was empty or blocked. Finish Reason: {finish_reason}, Safety Feedback: {safety_feedback}")
                    return f"Error: Failed to get response from Gemini. Finish Reason: {finish_reason}"
        except Exception as e:
                print(f"Error during Gemini API call: {str(e)}")
                return f"Error: Exception during Gemini API call: {str(e)}"


# Create a singleton instance
_model_instance = None

def get_model_instance():
    """Get or create a singleton instance of LLMModel"""
    global _model_instance
    if _model_instance is None:
        try:
            _model_instance = LLMModel()  # Constructor now reads env vars
        except (ValueError, RuntimeError, Exception) as e:
             # Handle initialization errors gracefully, e.g., log and exit or raise
             print(f"Fatal Error: Failed to initialize LLMModel: {str(e)}")
             # Depending on the application context, you might raise the error
             # or return None and handle it upstream.
             raise e  # Re-raise for now to make the failure obvious
    return _model_instance
