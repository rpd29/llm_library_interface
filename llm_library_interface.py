"""
This program was created to explore the possibility of using OpenAI languge models
to create natural language interfaces to software libraries.
"""

import argparse # import argparse module to enable user to pass labeled arguments to program through command line
import re # import re to use to extract file paths from model responses
import os # import os to enable retrieval of API key from environment variable
from pathlib import Path # import Path module to use to extract file names from file paths
import pandas as pd # Import pandas to use to read data from files
import openai # import openai module to use for api calls
import tiktoken # import openai tokenizer to use to determine how many tokens are in a text string and to decode tokenized strings


# The code below parses arguments provided by user. 
parser = argparse.ArgumentParser() #Create parser object

parser.add_argument("--library_summary_filepath", type=str, required = True, help = "filepath of file containing a list of the software libraries available to use, summaries of the types of functionality each library enables, and the filepaths of files that list all functions available in each library and contain natural language descriptions of what each function does.")
parser.add_argument("--log_filepath", type=str, required = True, help = "filepath of directory where user wishes log file containing sequences of prompts and model responses to be stored.")
parser.add_argument("--user_prompt", type=str, required = True, help = "prompt specifying what sort of information processing/computational task the user wants to complete.")


args = parser.parse_args() # get object containing values of all variables specified by user


# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") # Set value of OpenAI API key by retrieving value from environment variable. 

def language_to_code(library_summary_filepath, log_filepath, user_prompt):

	
	library_summary = pd.read_excel(library_summary_filepath) # Load file containing list of software libraries and descriptions of their functionality as a pandas dataframe

	
	library_summary_str = library_summary.to_string() #Convert the dataframe to a string

	prompt = "The user would like to complete the following computational task(s):\n\n" + user_prompt + "\n\nBelow is a list of software libraries, along with descriptions of each library's functionality and filepaths indicating where a more detailed description of each library can be found. Please identify any of these software libraries that might be useful for the user's task and respond with the library names followed by the associated file paths. Please put a <filepath> tag immediately before and a </filepath> tag immediately after each filepath (e.g. <filepath>/Users/Mary/secret_project/secret_project.py</filepath>). To reiterate, please put a <filepath> tag immediately before and a </filepath> tag immediately after each filepath (e.g. <filepath>/Users/Mary/secret_project/secret_project.py</filepath>). Software libraries:\n\n" + library_summary_str

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", 
		temperature = 0,
		messages=[
			{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
			{"role": "user", "content": prompt}], 
	)['choices'][0]['message']['content']

	pattern = r'<filepath>(.*?)</filepath>' # Define pattern to use to extract file paths from model response

	filepaths = re.findall(pattern, response) # Create list of file paths from model response

	log = "PROMPT:\n\n" + prompt + "\n\nRESPONSE:\n\n" + response + "\n\n" # Start a log of prompts sent to API and model responses		

	useful_functions = [] # Initialize list that will store information about potentially useful functions the model identifies
	keys = ["function_name", "input_variables", "function_description"] # List of keys to use to create a list of dictionaries of potentially useful functions
	function_name_pattern = r'<function_name>(.*?)</function_name>' # Define pattern to use to extract function names from model response
	input_variables_pattern = r'<input_variables>(.*?)</input_variables>' # Define pattern to use to extract input variables from model response
	function_description_pattern = r'<function_description>(.*?)</function_description>' # Define pattern to use to extract function descriptions from model response

	for i in filepaths: # For each potentially useful software library the model identified, ask the model to identify potentially useful functions from that library and compile information about these functions into a list of dictionaries 
		file_name_with_extension = os.path.basename(i) # Extract file name with extension from file path
		file_name_without_extension = Path(file_name_with_extension).stem # Extract file name without extension from file name with extension
		
		function_descriptions = pd.read_excel(i) # Load file containing detailed documentation on library as a pandas data frame

		function_descriptions_str = function_descriptions.to_string() # Convert the dataframe to a string 

		prompt = "The user would like to complete the following computational task(s):\n\n" + user_prompt + "\n\nYou previously identified the " + file_name_without_extension + " software library as a library that might be useful for this task. Below is a list of the functions in this library, along with the names of the input variables for each function and a description of what the function does. Please identify functions that might be useful for the user's task. For each function you identify, please respond with the function name, input variables, and function description. Please put a <function_name> tag immediately before and a </function_name> tag immediately after each function name (e.g. <function_name>add_two_numbers</function_name>). Please put a <input_variables> tag immediately before and a </input_variables> tag immediately after the input variables for each function (e.g. <input_variables>var_a, var_b</input_variables>). Please put a <function_description> tag immediately before and a </function_description> tag immediately after each function derscription (e.g. <function_description>This function adds the numbers a and b together</function_description>). Function list:\n\n" + function_descriptions_str

		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			temperature = 0,
			messages=[
				{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
				{"role": "user", "content": prompt}], 
		)['choices'][0]['message']['content']
		
		log = log + "PROMPT:\n\n" + prompt + "\n\nRESPONSE:\n\n" + response + "\n\n"

		function_names = re.findall(function_name_pattern, response) # Create list of function names from model response
		input_variables = re.findall(input_variables_pattern, response) # Create list of input variables from model response
		function_descriptions = re.findall(function_description_pattern, response) # Create list of function descriptions from model response
		new_useful_functions = [dict(zip(keys, values)) for values in zip(function_names, input_variables, function_descriptions)] # Create a list of dictionaries where each dictionary contains information on a function the model identified as potentially useful
		new_useful_functions = [dict(**d, **{"library": file_name_without_extension}) for d in new_useful_functions] # add the name of the library to each dictionary
		useful_functions = useful_functions + new_useful_functions # append information about newly identified potentially useful functions to the running list of potentially useful functions

	useful_functions_df = pd.DataFrame(useful_functions) # Convert list of useful functions to a Pandas DataFrame
	useful_functions_df_str = useful_functions_df.to_string() # Convert useful functions Pandas DataFrame to a string to include in prompt to model

	prompt = "The user would like to complete the following computational task(s):\n\n" + user_prompt + "\n\nYou previously identified a few proprietary Python software libraries that the user has access to that might be useful for this task. You also identified some functions from within each of these libraries you thought might be useful. Those functions, their input variables, and a description of what each function does is listed below. Please propose some code that will accomplish the computational task the user specified using the functions listed below and potentially functions from other Python libraries as well. Potentially useful functions:\n\n" + useful_functions_df_str

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", 
		temperature = 0,
		messages=[
			{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
			{"role": "user", "content": prompt}], 
	)['choices'][0]['message']['content']

	log = log + "PROMPT:\n\n" + prompt + "\n\nRESPONSE:\n\n" + response + "\n\n"

	# Insert code here asking model to write a program to achieve the task specified by the user using the functions it previously identified as being potentially useful
 
	with open(log_filepath, "w") as log_file: 
		log_file.write(log)


language_to_code(library_summary_filepath = args.library_summary_filepath, log_filepath = args.log_filepath, user_prompt = args.user_prompt)
