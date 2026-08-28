!pip install transformers torch sentencepiece accelerate -q from transformers import pipeline 
# 1. Load a pretrained code-generation model 
assistant = pipeline(     task="text2text-generation", 
    model="Salesforce/codet5-base" 
) 
 
# 2. Function to generate model response def generate_response(prompt, max_tokens=250):     result = assistant( 
        prompt, 
        max_new_tokens=max_tokens,         do_sample=False 
    ) 
 
    return result[0]["generated_text"] 
 
# 3. Code Generation code_task = """ 
Write a Python program to calculate the factorial of a number using a recursive function. 
 
Requirements: 
1.	Accept a number from the user. 
2.	Use recursion. 
3.	Display the factorial. 
4.	Handle negative input. 
""" 
 
generation_prompt = f""" 
Generate Python code for the following task. 
 
Task: 
{code_task} 
 
Return only a complete and properly indented Python program. 
""" 
 
generated_code = generate_response(generation_prompt) 
 
# 4. Debugging Assistant 
 
f_code = """ def calculate_average(numbers):     total = sum(numbers)     average = total / len(number)     return average 
 
values = [10, 20, 30, 40] 
print("Average:", calculate_average(values)) 
""" 
 
debugging_prompt = f""" 
Analyze the following Python program. 
 
code: 
{f_code} 
 
Tasks: 
1.	Identify the error. 
2.	Explain the cause briefly. 
3.	Provide the corrected Python program. 
 
Use the following format: 
 
Error: 
Explanation: 
Corrected Code: 
""" 
 
debugging_result = generate_response(debugging_prompt) 
 
# 5. Display Results 
 
print("AI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT") print("=" * 60) 
 
print("\n1. GENERATED CODE") 
print("-" * 60) 
print(generated_code) 
 
print("\n2. DEBUGGING RESULT") 
print("-" * 60) 
print(debugging_result) 
 
Sample Input 
Code-Generation Task 
Write a Python program to calculate the factorial of a number using a recursive function. Handle negative input. Program def calculate_average(numbers):     total = sum(numbers)     average = total / len(number)     return average 
 
values = [10, 20, 30, 40] print("Average:", calculate_average(values)) 
 
Sample Output Generated Code 
def factorial(number):     if number < 0:         return None     if number == 0 or number == 1: 
        return 1 
    return number * factorial(number - 1) 
 
number = int(input("Enter a number: ")) 
 
result = factorial(number) 
 
if result is None: 
    print("Factorial is not defined for negative numbers.") else: 
    print("Factorial:", result) 
 
Debugging Result Error: 
The variable "number" is used inside len(), but it is not defined. 
 
Explanation: 
The function parameter is named "numbers". Therefore, len(numbers) must be used to calculate the number of elements in the list. 
 
Corrected Code: 
def calculate_average(numbers):     total = sum(numbers)     average = total / len(numbers)     return average 
 
values = [10, 20, 30, 40] print("Average:", calculate_average(values)) 
Corrected Program Output 
Average: 25.0 
