!pip install groq gradio -q 
import os import re import time import getpass 
import gradio as gr 
 
from groq import Groq 
api_key = os.getenv("GROQ_API_KEY") 
 
if not api_key: 
    api_key = getpass.getpass("Enter your Groq API key: ") 
 
client = Groq(api_key=api_key) 
 
def calculate_relevance(prompt, response): 
    """ 
    Calculates a simple keyword-overlap relevance score. 
 
    This is only a basic laboratory evaluation metric and     is not a replacement for human or benchmark evaluation. 
    """ 
 
    stop_words = { 
        "the", "a", "an", "is", "are", "was", "were", 
        "to", "of", "in", "on", "for", "and", "or", 
        "with", "what", "how", "why", "write", "explain", 
        "describe", "give", "about" 
    } 
 
    prompt_words = set(         re.findall(r"\b[a-zA-Z]{3,}\b", prompt.lower()) 
    ) 
 
    response_words = set(         re.findall(r"\b[a-zA-Z]{3,}\b", response.lower()) 
    ) 
 
    important_words = prompt_words - stop_words 
 
    if not important_words: 
        return 100.0 
 
    matched_words = important_words.intersection(response_words) 
 
    score = (         len(matched_words) / len(important_words) 
    ) * 100 
 
    return round(score, 2) 
def generate_and_evaluate(prompt, temperature, max_tokens): 
    """ 
    Sends the prompt to a cloud-hosted language model and     returns the response with evaluation metrics. 
    """ 
 
    if not prompt or not prompt.strip():         return ( 
            "Please enter a valid prompt.", 
            { 
                "Status": "No prompt provided" 
            } 
        )      try: 
        start_time = time.perf_counter() 
 
        completion = client.chat.completions.create(             model="llama-3.1-8b-instant", 
            messages=[ 
                { 
                    "role": "system", 
                    "content": ( 
                        "You are a helpful Generative AI assistant. " 
                        "Provide accurate, clear and well-structured answers." 
                    ) 
                }, 
                { 
                    "role": "user", 
                    "content": prompt 
                } 
            ], 
            temperature=float(temperature), 
            max_tokens=int(max_tokens) 
        ) 
 
        end_time = time.perf_counter() 
 
        generated_response = (             completion.choices[0].message.content.strip() 
        ) 
 
        latency = end_time - start_time         word_count = len(generated_response.split()) 
        character_count = len(generated_response) 
 
        relevance_score = calculate_relevance( 
            prompt,             generated_response         ) 
 
        evaluation = { 
            "Model": "llama-3.1-8b-instant", 
            "Response Time (seconds)": round(latency, 3), 
            "Generated Word Count": word_count, 
            "Generated Character Count": character_count, 
            "Keyword Relevance Score (%)": relevance_score, 
            "Temperature": float(temperature), 
            "Maximum Tokens": int(max_tokens), 
            "Status": "Successfully generated" 
        } 
 
        return generated_response, evaluation 
 
    except Exception as error:         return ( 
            "The application could not generate a response.", 
            { 
                "Status": "Error", 
                "Error Message": str(error) 
            } 
        ) 
 
with gr.Blocks() as application: 
 
    gr.Markdown( 
        """ 
        # Cloud-Based Generative AI Application 
 
        Enter a prompt to generate content and evaluate the         response produced by the cloud-hosted language model. 
        """ 
    ) 
 
    with gr.Row(): 
 
        with gr.Column(): 
 
            prompt_input = gr.Textbox(                 label="Enter Prompt",                 placeholder=( 
                    "Example: Explain the applications of " 
                    "Generative AI in education." 
                ),                 lines=6 
            ) 
 
            temperature_input = gr.Slider( 
                minimum=0.0,                 maximum=1.0,                 value=0.3,                 step=0.1,                 label="Temperature" 
            ) 
 
            max_tokens_input = gr.Slider(                 minimum=50,                 maximum=500,                 value=250, 
                step=50,                 label="Maximum Tokens" 
            ) 
 
            generate_button = gr.Button( 
                "Generate and Evaluate" 
            ) 
 
            clear_button = gr.ClearButton( 
                [ 
                    prompt_input 
                ] 
            ) 
 
        with gr.Column(): 
 
            response_output = gr.Textbox(                 label="Generated Response", 
                lines=14 
            ) 
 
            evaluation_output = gr.JSON( 
                label="Evaluation Metrics" 
            ) 
 
    generate_button.click(         fn=generate_and_evaluate,         inputs=[             prompt_input,             temperature_input, 
            max_tokens_input 
        ], 
        outputs=[             response_output, 
            evaluation_output 
        ] 
    ) 
 
application.launch(     share=True, 
    debug=True 
) 
Setting share=True creates a public Gradio link for the running application. For permanent hosting, the same Gradio application can be uploaded to a Hugging Face Space.  
 
 
Sample Input 
Explain the role of Generative Artificial Intelligence in education. 
Include its benefits and three practical applications. 
Selected Parameters Temperature: 0.3 
Maximum Tokens: 250 
 
Sample Output Generated Response 
Generative Artificial Intelligence supports education by creating, adapting and presenting learning content according to student needs. 
 
Its major benefits include personalized learning, faster content development, immediate feedback and improved accessibility. 
 
Three practical applications are: 
 
1.	Generating personalized study materials and quizzes. 
2.	Providing AI-based tutoring and question-answering support. 3. Assisting teachers in preparing lesson plans, assessments and    classroom activities. 
 
Generative AI should be used responsibly, and its outputs should be reviewed by teachers before being used for academic purposes. 
Evaluation Metrics 
Model: llama-3.1-8b-instant 
Response Time: 1.246 seconds 
Generated Word Count: 78 
Generated Character Count: 608 
Keyword Relevance Score: 83.33% 
Temperature: 0.3 
Maximum Tokens: 250 
Status: Successfully generated 
Deployment Output Running on public URL: 
https://xxxxxxxxxxxxxxxx.gradio.live 
The public link remains available only while the Colab or local runtime is active. 
 
