import os
import logging
import gradio as gr
import pandas as pd
from api import GAIAHFAPIClient
from agents.agent import BasicAgent, SimpleGeminiAgent
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    api_client = GAIAHFAPIClient(profile=profile)  # Initialize the API client
    agent = SimpleGeminiAgent()
    # 2. Fetch Questions
    questions_data, error = api_client.get_questions()
    if error is None or questions_data is None:
        return questions_data, error

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    logger.info(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            logger.warning(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            logger.error(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        logger.warning("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": api_client.username.strip(), "agent_code": api_client.agent_code, "answers": answers_payload}
    logger.info(f"Agent finished. Submitting {len(answers_payload)} answers for user '{api_client.username}'...")

    # 5. Submit
    logger.info(f"Submitting {len(answers_payload)} answers to: {api_client.submit_url}")
    status_message, results_df = api_client.submit_answers(submission_data, results_log)
    return status_message, results_df

def run_and_submit_one(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    api_client = GAIAHFAPIClient(profile=profile)  # Initialize the API client
    agent = SimpleGeminiAgent()
    # 2. Fetch Question
    questions_data, error = api_client.get_questions()
    if error is None or questions_data is None:
        return questions_data, error
    question_data = random.choice(questions_data)

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    
    task_id = question_data.get("task_id")
    question_text = question_data.get("question")
    if not task_id or question_text is None:
        logger.warning(f"Skipping item with missing task_id or question: {question_data}")
    try:
        submitted_answer = agent(question_text)
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
    except Exception as e:
        logger.error(f"Error running agent on task {task_id}: {e}")
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        logger.warning("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": api_client.username.strip(), "agent_code": api_client.agent_code, "answers": answers_payload}
    logger.info(f"Agent finished. Submitting {len(answers_payload)} answers for user '{api_client.username}'...")

    # 5. Submit
    logger.info(f"Submitting {len(answers_payload)} answers to: {api_client.submit_url}")
    status_message, results_df = api_client.submit_answers(submission_data, results_log)
    return status_message, results_df

def build_gradio_interface():
    # --- Build Gradio Interface using Blocks ---
    with gr.Blocks() as demo:
        gr.Markdown("# Basic Agent Evaluation Runner")
        gr.Markdown(
            """
            **Instructions:**

            1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
            2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
            3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

            ---
            **Disclaimers:**
            Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
            This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
            """
        )

        gr.LoginButton()

        run_all_button = gr.Button("Run Evaluation & Submit All Answers")
        run_one_button = gr.Button("Run a Single Evaluation")

        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        # Removed max_rows=10 from DataFrame constructor
        results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

        run_all_button.click(
            fn=run_and_submit_all,
            outputs=[status_output, results_table]
        )
        run_one_button.click(
            fn=run_and_submit_one,
            outputs=[status_output, results_table]
        )
    return demo

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo = build_gradio_interface()
    demo.launch(debug=True, share=False)