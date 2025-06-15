import os
import requests
import logging
import pandas as pd
import gradio as gr

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class GAIAHFAPIClient:
    """
    A client for interacting with the GAIA HF API for the final assignment.
    This client handles dealing with the API requests.
    """

    def __init__(self, profile: gr.OAuthProfile | None, api_base_url=DEFAULT_API_URL):
        # --- Determine HF Space Runtime URL and Repo URL ---
        space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code
        if profile:
            self.username = f"{profile.username}"
            logger.info(f"User logged in: {self.username}")
        else:
            logger.warning("User not logged in.")
            return "Please Login to Hugging Face with the button.", None
        self.agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
        self.questions_url = f"{api_base_url}/questions"
        self.random_question_url = f"{api_base_url}/random-question"
        self.file_task_url = f"{api_base_url}/files/" + "{}"
        self.submit_url = f"{api_base_url}/submit"
        self.api_base_url = api_base_url

    def submit_answers(self, submission_data, results_log):
        try:
            response = requests.post(self.submit_url, json=submission_data, timeout=60)
            response.raise_for_status()
            result_data = response.json()
            final_status = (
                f"Submission Successful!\n"
                f"User: {result_data.get('username')}\n"
                f"Overall Score: {result_data.get('score', 'N/A')}% "
                f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
                f"Message: {result_data.get('message', 'No message received.')}"
            )
            print("Submission successful.")
            results_df = pd.DataFrame(results_log)
            return final_status, results_df
        except requests.exceptions.HTTPError as e:
            error_detail = f"Server responded with status {e.response.status_code}."
            try:
                error_json = e.response.json()
                error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
            except requests.exceptions.JSONDecodeError:
                error_detail += f" Response: {e.response.text[:500]}"
            status_message = f"Submission Failed: {error_detail}"
            print(status_message)
            results_df = pd.DataFrame(results_log)
            return status_message, results_df
        except requests.exceptions.Timeout:
            status_message = "Submission Failed: The request timed out."
            print(status_message)
            results_df = pd.DataFrame(results_log)
            return status_message, results_df
        except requests.exceptions.RequestException as e:
            status_message = f"Submission Failed: Network error - {e}"
            print(status_message)
            results_df = pd.DataFrame(results_log)
            return status_message, results_df
        except Exception as e:
            status_message = f"An unexpected error occurred during submission: {e}"
            print(status_message)
            results_df = pd.DataFrame(results_log)
            return status_message, results_df
        
    def get_questions(self):
        """
        Fetches the list of questions from the API.
        Returns:
            List of questions or an error message.
        """
        logger.info(f"Fetching questions from: {self.questions_url}")
        try:
            response = requests.get(self.questions_url, timeout=15)
            response.raise_for_status()
            questions_data = response.json()
            if not questions_data:
                logger.warning("Fetched questions list is empty.")
                return "Fetched questions list is empty or invalid format.", None
            logger.info(f"Fetched {len(questions_data)} questions.")
            return questions_data, 'success'
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching questions: {e}")
            return f"Error fetching questions: {e}", None
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from questions endpoint: {e}")
            logger.error(f"Response text: {response.text[:500]}")
            return f"Error decoding server response for questions: {e}", None
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching questions: {e}")
            return f"An unexpected error occurred fetching questions: {e}", None


    def get_random_question(self):
        """
        Fetches a random question from the API.
        Returns:
            A random question or an error message.
        """
        logger.info(f"Fetching a random question from: {self.random_question_url}")
        try:
            response = requests.get(self.random_question_url, timeout=15)
            response.raise_for_status()
            question_data = response.json()
            if not question_data:
                logger.warning("No random question data received.")
                return "No random question data received.", None
            logger.info(f"Received random question: {question_data.get('question', 'No question text')}")
            return question_data, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching random question: {e}")
            return f"Error fetching random question: {e}", None
        
    def get_file_task(self, task_id):
        """
        Fetches a file task by its ID from the API.
        Args:
            task_id (str): The ID of the task to fetch.
        Returns:
            The task data or an error message.
        """
        if not task_id:
            return "Task ID cannot be empty.", None
        _url = self.file_task_url.format(task_id)
        logger.info(f"Fetching file task from: {_url}")
        try:
            response = requests.get(_url, timeout=15)
            response.raise_for_status()

            # Handle the case where the response is not JSON / text or bytes. Inspect the response object
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                task_data = response.json()
            elif "text/" in content_type:
                task_data = response.text
            else:
                # Assume it's a binary file (e.g., CSV, image, etc.)
                task_data = response.content
            if not task_data:
                logger.warning(f"No data found for task ID: {task_id}")
                return f"No data found for task ID: {task_id}", None
            logger.info(f"Received file task for ID {task_id}: {task_data}")
            return task_data, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching file task {task_id}: {e}")
            return f"Error fetching file task {task_id}: {e}", None

if __name__ == "__main__":
    # Example usage
    demo = gr.Blocks(title="GAIA HF API Client Demo")
    api_client = GAIAHFAPIClient(profile=demo)
    random_question, error = api_client.get_random_question()
    if error:
        logger.error(f"Error fetching random question: {error}")
    else:
        logger.info(f"Random question: {random_question}")

    questions, error = api_client.get_questions()
    if not error:
        logger.error(f"Error fetching questions: {error}")
    else:
        logger.info(f"Fetched questions: {questions}")

    task_id = 'f918266a-b3e0-4914-865d-4faa564f1aef'
    task_data, error = api_client.get_file_task(task_id)
    if error:
        logger.error(f"Error fetching file task {task_id}: {error}")
    else:
        logger.info(f"Fetched file task: {task_data}")
