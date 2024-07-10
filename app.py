import streamlit as st
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load TF-IDF and model functions
def load_tfidf():
    with open("tf_idf.pkt", "rb") as f:
        tfidf = pickle.load(f)
    return tfidf

def load_model():
    with open("toxicity_model.pkt", "rb") as f:
        nb_model = pickle.load(f)
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text])
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    return class_name

# Streamlit app
st.header("Toxicity Detection App")

st.subheader("Input your text")
text_input = st.text_input("Enter your text")

if text_input:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info(f"The result is {result}.")

# Initialize session state for comments if not already initialized
if 'comments' not in st.session_state:
    st.session_state.comments = {}

# Function to display comments and their replies in a styled format
def display_comments(comments):
    for comment_id, comment_data in comments.items():
        if comment_data['is_toxic']:
            st.markdown(f"<span style='color: red;'>⚠️ Toxic Comment {comment_id}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Comment {comment_id}**: {comment_data['text']}")
        
        for reply_id, reply_data in comment_data['replies'].items():
            if reply_data['is_toxic']:
                st.markdown(f"&emsp;<span style='color: red;'>⚠️ Toxic Reply {reply_id} to Comment {comment_id}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"&emsp;<b>Reply {reply_id} to Comment {comment_id}</b>: {reply_data['text']}", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

# Comment Section
st.subheader("Comment Section")
new_comment = st.text_area("Write your comment here...", key='new_comment')

if new_comment:
    if st.button("Post Comment", key='post_comment'):
        toxicity = toxicity_prediction(new_comment)
        is_toxic = toxicity == "Toxic"
        comment_id = len(st.session_state.comments) + 1
        st.session_state.comments[comment_id] = {
            'text': new_comment,
            'is_toxic': is_toxic,
            'replies': {}
        }
        st.experimental_rerun()  # Refresh to display the new comment

# Display comments
display_comments(st.session_state.comments)

# Reply Section
st.subheader("Reply to a Comment")
comment_id_to_reply = st.number_input("Enter comment ID to reply to", min_value=1, step=1)
reply_text = st.text_area("Write your reply here...", key='reply_text')

if reply_text:
    if st.button("Post Reply", key='post_reply'):
        if comment_id_to_reply in st.session_state.comments:
            toxicity = toxicity_prediction(reply_text)
            is_toxic = toxicity == "Toxic"
            reply_id = len(st.session_state.comments[comment_id_to_reply]['replies']) + 1
            st.session_state.comments[comment_id_to_reply]['replies'][reply_id] = {
                'text': reply_text,
                'is_toxic': is_toxic
            }
            st.experimental_rerun()  # Refresh to display the new reply

# Display comments again to show replies
display_comments(st.session_state.comments)













# import streamlit as st
# import pickle 
# import numpy as np 
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # Load TF-IDF and model functions
# def load_tfidf():
#     with open("tf_idf.pkt", "rb") as f:
#         tfidf = pickle.load(f)
#     return tfidf

# def load_model():
#     with open("toxicity_model.pkt", "rb") as f:
#         nb_model = pickle.load(f)
#     return nb_model

# def toxicity_prediction(text):
#     tfidf = load_tfidf()
#     text_tfidf = tfidf.transform([text])
#     nb_model = load_model()
#     prediction = nb_model.predict(text_tfidf)
#     class_name = "Toxic" if prediction[0] == 1 else "Non-Toxic"
#     return class_name

# # Streamlit app
# st.header("Toxicity Detection App")

# st.subheader("Input your text")
# text_input = st.text_input("Enter your text")

# if text_input:
#     if st.button("Analyse"):
#         result = toxicity_prediction(text_input)
#         st.subheader("Result:")
#         st.info(f"The result is {result}.")

# # Initialize session state for comments if not already initialized
# if 'comments' not in st.session_state:
#     st.session_state.comments = {}

# # Function to display comments and their replies in a styled format
# def display_comments(comments):
#     for comment_id, comment_data in comments.items():
#         st.markdown(f"**Comment {comment_id}**: {comment_data['text']}")
#         if comment_data['is_toxic']:
#             st.markdown(f"<span style='color: red;'>⚠️ Toxic Comment {comment_id}</span>", unsafe_allow_html=True)
#         for reply_id, reply_data in comment_data['replies'].items():
#             st.markdown(f"&emsp;<b>Reply {reply_id} to Comment {comment_id}</b>: {reply_data['text']}", unsafe_allow_html=True)
#             if reply_data['is_toxic']:
#                 st.markdown(f"&emsp;<span style='color: red;'>⚠️ Toxic Reply {reply_id}</span>", unsafe_allow_html=True)
#         st.markdown("<hr>", unsafe_allow_html=True)

# # Comment Section
# st.subheader("Comment Section")
# new_comment = st.text_area("Write your comment here...", key='new_comment')

# if new_comment:
#     if st.button("Post Comment", key='post_comment'):
#         toxicity = toxicity_prediction(new_comment)
#         is_toxic = toxicity == "Toxic"
#         comment_id = len(st.session_state.comments) + 1
#         st.session_state.comments[comment_id] = {
#             'text': new_comment,
#             'is_toxic': is_toxic,
#             'replies': {}
#         }
#         st.experimental_rerun()  # Refresh to display the new comment

# # Display comments
# display_comments(st.session_state.comments)

# # Reply Section
# st.subheader("Reply to a Comment")
# comment_id_to_reply = st.number_input("Enter comment ID to reply to", min_value=1, step=1)
# reply_text = st.text_area("Write your reply here...", key='reply_text')

# if reply_text:
#     if st.button("Post Reply", key='post_reply'):
#         if comment_id_to_reply in st.session_state.comments:
#             toxicity = toxicity_prediction(reply_text)
#             is_toxic = toxicity == "Toxic"
#             reply_id = len(st.session_state.comments[comment_id_to_reply]['replies']) + 1
#             st.session_state.comments[comment_id_to_reply]['replies'][reply_id] = {
#                 'text': reply_text,
#                 'is_toxic': is_toxic
#             }
#             st.experimental_rerun()  # Refresh to display the new reply

# # Display comments again to show replies
# display_comments(st.session_state.comments)

