# src/core/model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai  # Added import

load_dotenv()

class LLMModel:
    def __init__(self):
        """Initialize the LLM model with configurations based on environment variables"""
        self.model_type = os.getenv("MODEL_TYPE", "local")  # 'local' or 'gemini'
        self.model = None
        self.tokenizer = None
        self.device = None  # Only set for local models
        self.setup_model()

    def setup_model(self):
        """Set up the model based on self.model_type"""
        if self.model_type == "gemini":
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
        elif self.model_type == "local":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Attempting to load local model on: {self.device}")  # Added print

            quantization_config = None
            model_kwargs = {
                "use_safetensors": True,
                "low_cpu_mem_usage": True,  # Keep this for CPU loading
                "device_map": self.device
            }

            if self.device == "cuda":
                # Only apply quantization if CUDA is available
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
                print("CUDA detected. Applying 4-bit quantization.")  # Added print
            else:
                # If on CPU, don't use quantization_config
                print("CUDA not available. Loading model on CPU without quantization.")  # Added print
                model_kwargs["device_map"] = "cpu"

            # Ensure the model identifier is correct if you change it
            model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs  # Pass arguments dictionary
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"Local model '{model_id}' loaded successfully on {self.device}.")
            except Exception as e:
                # Provide more context in the error
                print(f"Error details: Device={self.device}, Quantization applied: {quantization_config is not None}")
                raise RuntimeError(f"Failed to load local model '{model_id}': {str(e)}")
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {self.model_type}")

    def generate_response(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """
        Generate response from the configured model for a given prompt

        Args:
            prompt (str): Input prompt for the model
            max_new_tokens (int): Maximum number of tokens to generate (used by local model)

        Returns:
            str: Generated response
        """
        if self.model_type == "gemini":
            try:
                # Gemini API might have different ways to handle system prompts if needed
                # For simple cases, just sending the user prompt works.
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

        elif self.model_type == "local":
            # Existing logic for local Hugging Face model
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
        else:
             # Should not happen if setup_model worked correctly
            return "Error: Model type not configured correctly."


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
