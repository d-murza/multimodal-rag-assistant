from transformers import pipeline
import torch

# TASK 1 - Model Setup and Prompt Structure 
# print("TASK 1")
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Device:", device)

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, device=device)

# config = pipe.model.config
# print("Context window: ", config.max_position_embeddings) # to find max tokens available



# Implement a clear prompt construction strategy that separates:
# a system prompt (global instructions and behavior)
system_prompt = "You are a ”Help me get things done chatbot”. " \
                "Reply with a short intro sentence, " \
                "then a numbered list of clear consice steps (1., 2. ...). " \
                "Keep each step to a single sentence. " \
                "If a context block is provided, ground your answer in it. " \
                "Do not include extra commentary, greetings or questions." \

# an optional context block (empty for now, reserved for later retrieval),
context_block = "" # this will be set later and retrieved in the chatbot loop

# the conversation history consisting of role–content pairs.
conversation_history = []

# creating the prompt
def prompt_construction(system_prompt, context_block, conversation_history):
    prompt = ""

    # first get system prompt, end with </s> so that the model knows this is end of sequence
    prompt += "<|system|>\n"
    prompt += system_prompt + "</s>\n"

    # add context (for now it's empty)
    prompt += "<|context|>\n"
    prompt += context_block + "</s>\n"

    # go through conversation history and get the "person" + the text
    for key in conversation_history:
        if key["role"] == "user":
            prompt += "<|user|>\n"
        elif key["role"] == "assistant":
            prompt += "<|assistant|>\n"

        prompt += key["content"] + "</s>\n"

    # tell the chat bot it can generate after this
    prompt += "<|assistant|>\n"
    return prompt

# code for testing the prompt_construction
#prompt = prompt_construction(system_prompt, context_block, conversation_history)
#print("Prompt:")
#print(prompt)

#outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
#print(outputs[0]["generated_text"])


# TASK 2 - Chat Loop and Conversation Management

# max tokens available for new generated text is max - 256
max_new_tokens = 256 # needs to have this much free to do another step of generating
overall_max_tokens = 2048
max_tokens = overall_max_tokens - max_new_tokens

def chat_loop():
    print("----- Starting chatbot -----")
    while True:
        # reads user input
        user_message = input("User: ")

        # breaks and stops if you write "quit"
        if user_message == "quit":
            print("Shutting chatbot down...")
            break

        # append user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_message
            })

        # this is where we would insert the context block
        # essentially calling (the function that retrieves from earlier assignment)
        # context_block = retrieval_test(user_message, combined_index, connect)

        # builds the full prompt (including the system prompt and context),
        prompt = prompt_construction(system_prompt, context_block, conversation_history)

        # checking so that the prompts are not using all tokens
        token_count = len(tokenizer.encode(prompt))

        # if so, truncate = remove old messages
        while token_count > max_tokens:
            conversation_history.pop(0)
            # print("Popping history")
            # build new prompt with the now limited information
            prompt = prompt_construction(system_prompt, context_block, conversation_history)
            token_count = len(tokenizer.encode(prompt))

        #print("Token count: ", token_count)

        # generates a model response + having stopping criteria as a configurable maximum number of new tokens.
        outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        # get only the latest answer to show to the user
        generated = outputs[0]["generated_text"]
        model_response = generated[len(prompt):].strip()

        model_response = output_validation(model_response) # implement the validation from task 3

        # if an error occured - break loop and go to user input again
        if model_response == "Error":
            continue

        print("Chat bot: ", model_response)
        # appends the response to the conversation history.
        conversation_history.append({
            "role": "assistant",
            "content": model_response
            })


# TASK 3 - Output Validation and RAG Preparation

# function that takes the model response and checks the three conditions
def output_validation(model_response):
    words = model_response.split()
    length = len(words)

    # check it doesn't leak system or user role tokens
    role_tokens = ["<|user|>","<|assistant|>","<|context|>","<|system|>" ]
    for token in role_tokens:
        model_response = model_response.replace(token, "") # exchange role token to nothing
        words = model_response.split() # update list with words
        length = len(words)

    # incomplete or empty outputs are handled gracefully by checking number of words
    if length < 10:
        print("Error: Generated incomplete response. Try again.")
        return "Error"

    # degenerate outputs are handled gracefully
    # check ratio of unique words - https://doc.atlasti.com/ManualWin/Exploring/ExploringTypeTokenRatio.html
    number_unique_words = len(set(words))
    unique_word_ratio = number_unique_words/length
    if unique_word_ratio < 0.3: # set a little lower than recommended in page above
        print("Error: Output is degenerate. Unique word ratio is to low. Can't be proper and meaningful sentences.")
        return "Error"

    # malformed responses are detected. - check so that there is a list structure
    if "1." not in model_response:
        print("Error: Malformed response. List structure is missing.")
        return "Error"

    # want cleaned up response returned
    return model_response


# standalone entry point: loads the model and runs the interactive chat loop.
# guarded so importing this module (Assignment 10) does not trigger any of it.
if __name__ == "__main__":
    print("TASK 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, device=device)

    config = pipe.model.config
    print("Context window: ", config.max_position_embeddings) # to find max tokens available

    print("")

    print("TASK 2")
    # count tokens
    tokenizer = pipe.tokenizer

    print("\nTASK 3")

    # starting chatloop
    chat_loop()
