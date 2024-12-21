import os
import json
from openai import OpenAI
import dotenv
from upload import search_db, Internet_search
import tools

dotenv.load_dotenv()
tools = tools.tools

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def run_conversation(user_input):
    messages=user_input
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",  
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_functions = {
        "search_db": search_db,
        "Internet_search": Internet_search,
        }
        messages.append(response_message)  
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "search_db":
                function_response = function_to_call(
                    collection_name=function_args.get("collection_name"),
                    input_query=function_args.get("input_query"),
                    n=function_args.get("n"),
                )
            elif function_name == "Internet_search":
                function_response = function_to_call(
                    input=function_args.get("input"),
                )
                
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  
        second_response = client.chat.completions.create(model="gpt-4o-mini",messages= messages)
        output=second_response.choices[0].message.content
        return str(output)
    else:
        return str(response_message.content)
    
def main():
    messages = [
        {"role": "system", "content": """You are a smart AI Bot. Your task is to answer the user's query based on information from three relevant database collections:
            1. `Agent_Post`: Contains information on LLM Powered Autonomous Agents, including task decomposition, memory, and tool use, as well as case studies like 
                scientific discovery agents and generative agent simulations.
            2. `Prompt_Engineering_Post`: Contains detailed resources on prompt engineering, including techniques like zero-shot, few-shot, chain-of-thought prompting, 
                and automatic prompt design, as well as the use of external APIs and augmented language models.
            3. `Adv_Attack_LLM_Post`: Contains content on adversarial attacks on LLMs, including text generation, white-box vs black-box attacks, jailbreak prompting, 
                and various mitigation strategies.

            Attempt to search in the relevant database collection that matches the user's query. If no relevant information is found in the database, 
            perform an internal search. 
            Also perform Internet search to get realtime data.
            While responding, also mention where you sourced the data from.
         """},
        {"role": "assistant", "content": "How can I help you?"}
    ]
    
    print("AI Assistant: How can I help you? (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        response = run_conversation(messages)
        messages.append({"role": "assistant", "content": response})
        print("\nAI Assistant:", response, "\n")

if __name__ == "__main__":
    main()