# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import torch

def infer_pipeline(messages):
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]) + "\nAssistant: "
    print(prompt)

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf",  device_map="auto")
    # pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf", tokenizer=tokenizer, device_map="auto", model_kwargs={"torch_dtype": torch.bfloat16})
    # pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf",  device_map="auto")


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",  device_map="auto")
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct",  model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    # print(pipe.model.device)

    output = pipe(prompt, max_new_tokens=100)
    # Check lai xem coi output co dung khong !!!
    print("Output:\n", output[0]['generated_text'])
    return output

messages = [{'role': 'system', 'content': 'Instructions:\n- The Overcooked_AI game requires two players to work together as a team with the goal of achieving the highest possible score.\n- To get the points, the team need to make soup according to recipe, fill the soup in a dish and immediately deliver the soup. Once a delivery is made, the team gets 20 points. Plus, this soup and dishes will disappear, no need to wash and recycle the dishes.\n- recipe: THREE onions in the <Pot>\n- To make a soup, the team must pick up THREE onions one by one, put them in a <Pot>. <Pot> will automatically start cooking when <Pot> is full (contains 3 onions) and it will take 20 time steps to complete the soup.\n- Each player can only pick up and hold one item at a time.To put down the thing which player is holding and empty his hand, he should use place_obj_on_counter().\n- <Pot> can hold THREE ingredients.\n- After start cooking, <Pot> needs 20 cooking timesteps to finish a food. Before the food is finished, players should: pickup(dish) and then fill_dish_with_soup().\n\nSkill: \nIn this game, each player can ONLY perform the following 6 allowed skill: pickup(onion),pickup(dish), place_obj_on_counter(), put_onion_in_pot(), fill_dish_with_soup(), deliver_soup(). Do not attempt to use any other skills that are not listed here.\n    - pickup(onion)\n        - I need to have nothing in hand.\n    - put_onion_in_pot()\n        - I need to have an onion in hand.\n    - pickup(dish)\n        - Need to have a soup cooking or ready in <Pot>.\n        - I need to have nothing in hand.\n        - If there isn\'t a  cooking or ready soup in the current scene , I shouldn\'t pickup(dish).\n    - fill_dish_with_soup()\n        - Need to have soup  cooking or ready in <Pot>.\n        - I need to pickup(dish) first or have a dish in hand.\n        - Then I must deliver_soup().\n    - deliver_soup()\n        - I must do deliver_soup() when I hold a dish with soup!\n        - I need to have soup in hand.\n        - The dish and soup will both disappear.\n    - place_obj_on_counter()\n        - I need to have something in hand.\n        - Don\'t  place_obj_on_counter() when I hold a dish with soup!   \n\nSuppose you are an assistant who is proficient in the overcooked_ai game. Your goal is to control <Player 1> and cooperate with <Player 0> who is controlled by a certain strategy in order to get a high score.\n- <Player 1> and <Player 0> cannot communicate. \n- You cannot use move actions and don\'t use the location information in the observation.\n- <Pot> is full when it contains 3 onions.\n- You must do deliver_soup() when you hold a soup!\n- For each step, you will receive the current scene (including the kitchen, teammates status).\n- Based on the current scene, you need to \n    1. Describe the current scene and analysis it,\n    2. Plan ONLY ONE best skill for <Player 1>  to do right now. Format should not use natural language, but use the allowed functions,don\'t respond with skill that is not in the allowed skills.\n\nExamples: \n###\nScene 888: <Player 1> holds nothing.<Player 0> holds nothing. Kitchen states: <Pot 0> has 1 onion.\nAssistant:\n- Analysis: Both players are currently not holding anything. <Pot 0> is not full and still needs 2 more onions to be full and start cooking.\n- Plan for Player 1: "pickup(onion)". \n###\nScene 999: <Player 1> holds one onion.<Player 0> holds nothing. Kitchen states: <Pot 0> is empty. \n Player 1 is currently holding one onion. Player 0 is not holding anything. The <Pot 0> is empty and needs to be filled with onions to start cooking.\nAssistant:\n- Plan for Player 1: "put_onion_in_pot()".\n###'}, {'role': 'user', 'content': "Here's the layout of the kitchen: <Onion Dispenser 0>, <Onion Dispenser 1>, <Dish Dispenser 0>, <Serving Loc 0>, <Pot 0>.\nScene 0: <Player 1> holds nothing. <Player 0> holds nothing. Kitchen states: <Pot 0> is empty; "}]

infer_pipeline(messages)
