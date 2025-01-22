# import openai
from rich import print as rprint
import time
from typing import Union
# from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
from .utils import convert_messages_to_prompt
from transformers import pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from .testing import infer_pipeline

# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
}


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self, 
                 role_messages, 
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=3):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        if "meta-llama" in self.model:
            # tokenizer = AutoTokenizer.from_pretrained(self.model,  device_map="auto")
            # self.pipe = pipeline("text-generation", model=self.model, tokenizer=tokenizer, device_map="auto")
            pass
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",  device_map="auto")
        # self.pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", tokenizer=tokenizer, device_map="auto")


        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    # @retry_with_exponential_backoff
    def query(self, key=None, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        # openai.api_key = key 
        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        get_response = False
        retry_count = 0
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            try:  
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages) 
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                if "meta-llama" in self.model: 
                    # print('\n\nmessages = \n\n{}\n\n'.format(messages))
                    print("Here", messages)
                    # exit()
                    # messages = [{'role': 'system', 'content': 'Instructions:\n- The Overcooked_AI game requires two players to work together as a team with the goal of achieving the highest possible score.\n- To get the points, the team need to make soup according to recipe, fill the soup in a dish and immediately deliver the soup. Once a delivery is made, the team gets 20 points. Plus, this soup and dishes will disappear, no need to wash and recycle the dishes.\n- recipe: THREE onions in the <Pot>\n- To make a soup, the team must pick up THREE onions one by one, put them in a <Pot>. <Pot> will automatically start cooking when <Pot> is full (contains 3 onions) and it will take 20 time steps to complete the soup.\n- Each player can only pick up and hold one item at a time.To put down the thing which player is holding and empty his hand, he should use place_obj_on_counter().\n- <Pot> can hold THREE ingredients.\n- After start cooking, <Pot> needs 20 cooking timesteps to finish a food. Before the food is finished, players should: pickup(dish) and then fill_dish_with_soup().\n\nSkill: \nIn this game, each player can ONLY perform the following 6 allowed skill: pickup(onion),pickup(dish), place_obj_on_counter(), put_onion_in_pot(), fill_dish_with_soup(), deliver_soup(). Do not attempt to use any other skills that are not listed here.\n    - pickup(onion)\n        - I need to have nothing in hand.\n    - put_onion_in_pot()\n        - I need to have an onion in hand.\n    - pickup(dish)\n        - Need to have a soup cooking or ready in <Pot>.\n        - I need to have nothing in hand.\n        - If there isn\'t a  cooking or ready soup in the current scene , I shouldn\'t pickup(dish).\n    - fill_dish_with_soup()\n        - Need to have soup  cooking or ready in <Pot>.\n        - I need to pickup(dish) first or have a dish in hand.\n        - Then I must deliver_soup().\n    - deliver_soup()\n        - I must do deliver_soup() when I hold a dish with soup!\n        - I need to have soup in hand.\n        - The dish and soup will both disappear.\n    - place_obj_on_counter()\n        - I need to have something in hand.\n        - Don\'t  place_obj_on_counter() when I hold a dish with soup!   \n\nSuppose you are an assistant who is proficient in the overcooked_ai game. Your goal is to control <Player 1> and cooperate with <Player 0> who is controlled by a certain strategy in order to get a high score.\n- <Player 1> and <Player 0> cannot communicate. \n- You cannot use move actions and don\'t use the location information in the observation.\n- <Pot> is full when it contains 3 onions.\n- You must do deliver_soup() when you hold a soup!\n- For each step, you will receive the current scene (including the kitchen, teammates status).\n- Based on the current scene, you need to \n    1. Describe the current scene and analysis it,\n    2. Plan ONLY ONE best skill for <Player 1>  to do right now. Format should not use natural language, but use the allowed functions,don\'t respond with skill that is not in the allowed skills.\n\nExamples: \n###\nScene 888: <Player 1> holds nothing.<Player 0> holds nothing. Kitchen states: <Pot 0> has 1 onion.\nAssistant:\n- Analysis: Both players are currently not holding anything. <Pot 0> is not full and still needs 2 more onions to be full and start cooking.\n- Plan for Player 1: "pickup(onion)". \n###\nScene 999: <Player 1> holds one onion.<Player 0> holds nothing. Kitchen states: <Pot 0> is empty. \n Player 1 is currently holding one onion. Player 0 is not holding anything. The <Pot 0> is empty and needs to be filled with onions to start cooking.\nAssistant:\n- Plan for Player 1: "put_onion_in_pot()".\n###'}, {'role': 'user', 'content': "Here's the layout of the kitchen: <Onion Dispenser 0>, <Onion Dispenser 1>, <Dish Dispenser 0>, <Serving Loc 0>, <Pot 0>.\nScene 0: <Player 1> holds nothing. <Player 0> holds nothing. Kitchen states: <Pot 0> is empty; "}]
                    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]) + "\nAssistant:"
                    print(prompt)
                    # exit()
                    tokenizer = AutoTokenizer.from_pretrained(self.model,  device_map="auto")
                    pipe = pipeline("text-generation", model=self.model, tokenizer=tokenizer, device_map="auto")
                    # response = self.pipe(prompt, max_new_tokens=100)
                    response = pipe(prompt,max_new_tokens=256)
                    print("The response: ", response)
                    exit()  
                    # time.sleep(10)  
                elif 'gpt' in self.model:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                    # time.sleep(10) 
                else:
                    raise Exception(f"Model {self.model} not supported.")
                
                get_response = True

            except Exception as e:
                retry_count += 1
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(20)  
        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude': 
            return response 
        elif self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        elif self.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314']:
            return response["choices"][0]["message"]["content"]
        elif "meta-llama" in self.model: 
            return response[0]['generated_text']

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []

