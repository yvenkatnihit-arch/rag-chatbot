import streamlit as st   # the UI library, always imported as "st" by convention
from qa_chain import answer_question_streaming
from citations import get_unique_citations

st.title("Nike Documents Chatbot")
st.caption("Ask questions about Nike's 10-K filing and growth story. Answers are grounded in these documents only.")

# Session state: this list survives between reruns, so the chat doesn't reset
# every time you send a message. It only resets if you refresh the page.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (question, answer) tuples, for the rewrite step

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of dicts, for displaying the chat on screen

# Redraw every past message each time the script reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):   # "role" is either "user" or "assistant"
        st.markdown(message["content"])

# The chat input box at the bottom of the page
user_question = st.chat_input("Ask a question about Nike...")

if user_question:
    # Show the user's question immediately
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Get and display the streaming answer
    with st.chat_message("assistant"):
        token_stream, chunks = answer_question_streaming(
            user_question,
            history=st.session_state.chat_history,
        )
        # st.write_stream displays each piece as it arrives, and returns the full text at the end
        full_answer = st.write_stream(token_stream)

        # Show the sources underneath the answer
        citations = get_unique_citations(chunks)
        st.markdown("**Sources:**")
        for citation in citations:
            st.markdown(f"- {citation}")

    # Save this turn to both histories, for display and for future rewriting
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.chat_history.append((user_question, full_answer))