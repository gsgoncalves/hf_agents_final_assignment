import os
import time
import logging
from google import genai
from google.genai.types import GenerateContentConfig
from ratelimit import limits, sleep_and_retry

RPM = 15
TPM = 1_000_000
PER_MINUTE = 60
SYSTEM_PROMPT_GAIA = "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class BasicAgent:
    def __init__(self):
        logger.info("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        logger.info(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        logger.info(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

class SimpleGeminiAgent(BasicAgent):
    def __init__(self, model="gemini-2.5-flash-preview-05-20"):#model="gemini-2.0-flash"):
        super().__init__()
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=gemini_key)
        self.model = model
        logger.info("AdvancedAgent initialized.")
        self.minute_start = time.time()
        self.tokens_this_minute = 0
        self.token_count = 0

    @sleep_and_retry
    @limits(calls=RPM, period=PER_MINUTE)
    def __call__(self, question: str) -> str:
        now = time.time()
        if now - self.minute_start >= 60:
            self.tokens_this_minute = 0
            self.minute_start = now
        
        # Enforce tokens per minute
        if self.tokens_this_minute + self.token_count > TPM:
            sleep_time = max(0, 60 - (now - self.minute_start))
            time.sleep(sleep_time)
            self.tokens_this_minute = 0
            self.minute_start = time.time()

        response = self.client.models.generate_content(model=self.model, 
                                                       contents=question,
                                                       config=GenerateContentConfig(system_instruction=SYSTEM_PROMPT_GAIA))
        self.tokens_this_minute += response.usage_metadata.total_token_count
        self.token_count += response.usage_metadata.total_token_count
        logger.info(f"AdvancedAgent received question (first 50 chars): {question[:50]}...")
        logger.info(f"AdvancedAgent returning answer: {response.text}")
        return response.text

if __name__ == "__main__":
    # Example usage
    agent = SimpleGeminiAgent()
    question = "What is the capital of France?"
    answer = agent(question)
    print(f"Question: {question}\nAnswer: {answer}")