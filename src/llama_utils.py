from groq import Groq
import os
from groq import Groq
# $env:GROQ_API_KEY="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" (Powershell)

os.environ["GROQ_API_KEY"] = "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"
client = Groq()

MODEL = "llama-3.1-8b-instant"

def process_user_input(st, user_prompt):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history,
    ]

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Prompt the LLM and show results to user. Hide initial prompt from user.
def process_hidden_prompt(st, hidden_prompt):
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": hidden_prompt},
    ]

    # Send the hidden prompt to the LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content

    # Add response to chat history and show it to the user
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    print("Chat history after added hidden prompt: ", st.session_state.chat_history)
    #with st.chat_message("assistant"):
    #    st.markdown(assistant_response)


# Prompt the LLM with a silent instruction (no output)
def process_silent_instruction(hidden_instruction):
    #Sends a silent instruction to the LLM, with no visible input or output to the user.

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": hidden_instruction},
    ]

    # Send the instruction to the LLM (no output displayed)
    client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
