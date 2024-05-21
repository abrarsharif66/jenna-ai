import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
import time
from streamlit_lottie import st_lottie
import requests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_photography = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")



st.title("Bravo Technologies")
st.write("Connect With our smart agent")
st_lottie(lottie_photography, height=300, key="photography")
time.sleep(2)
st.write("You are now talking to our agent Jenna")
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_api = os.getenv('QDRANT_KEY')
hf_token=os.getenv('HF_TOKEN')
q_url=os.getenv('QDRANT_URL')




from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document


from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./pdfs/', glob="./*.pdf", loader_cls=PyPDFLoader)

pages = loader.load_and_split()

text_documents=loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(text_documents)
docs[:5]




from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("line 39")
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": DEVICE})
print("line 42")
from langchain_community.vectorstores import Qdrant


url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_KEY")

qdrant = Qdrant.from_documents(
    documents=text_documents,
    embedding=instructor_embeddings,
    url=url,
    collection_name="chatbot_4",
    api_key=api_key
)
#qdrant=Qdrant.from_existing_collection(collection_name="chat_bot1",api_key=api_key,url=url,embedding=instructor_embeddings)



#using chat completion
conversation_history=[

            {
"role":"system",
                "content":"""You are a business development associate for Bravo Technologies, a leading software company. Your role is to have natural conversations with potential customers who are calling in to learn more about Bravo's products and services. 

            Since this interaction stems from a phone call, you should:

            - Greet the caller politely and introduce yourself as Jenna, a Bravo business development associate
            - Ask for the caller's name and use it throughout the conversation  
            - Speak in a friendly yet professional tone
            - Allow the caller to drive the conversation by asking questions, but also be prepared to highlight Bravo's key offerings
            - Gather information about the caller's needs and interests related to software solutions
            - Provide relevant details about Bravo's products/services that could address their requirements
            - Avoid jargon and explain technical concepts in plain language
            - Make it clear you are an AI assistant having a friendly conversation, not an actual Bravo employee
            - End the call by thanking the person for their time and interest, and invite follow-up

            Your goal is to have an engaging dialogue, build rapport, and position Bravo as an innovative solution for the caller's software needs."""
},
]





import speech_recognition as sr
import os
from groq import Groq
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def transcribe_and_process_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("User said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
def get_llm_response(user_input, history):
    # Update the conversation history with user input
    history.append({"role": "user", "content": f"now using this {context} answer this user's query {user_input}"})

    # Call the LLM with the conversation history
    chat_completion = client.chat.completions.create(
        messages=history,
        model="Llama3-70b-8192",
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Extract the LLM response and update the conversation history
    llm_response = chat_completion.choices[0].message.content
    history.append({"role": "assistant", "content": llm_response})

    return llm_response

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


def classify_convo(whole_chat):
    analyse_chat=[

                    {
        "role": "system",
        "content": """You will receive a chat history between a user and an LLM assistant in the following format:

        {"role":"system","content":"#instructions"}
        {"role":"user","content":"#message" },
        {"role":"assistant","content":"#response"}
        ...

        Your task is to analyze the overall sentiment of the chat history and strictly output one of the following:

        positive
        negative
        neutral

        Do not provide any explanations, sentences, or additional output beyond one of those three words."""
        },
        ]
    analyse_chat.append({"role": "user", "content": f"now classify this chat: {whole_chat}"})
    chat_completion = client.chat.completions.create(
        messages=analyse_chat,
        model="Llama3-70b-8192",
        temperature=0,
        max_tokens=100,
        top_p=1,
        stop=None,
        stream=False,
    )
    sentiment = chat_completion.choices[0].message.content
    return sentiment
def context_convo(whole_chat2):
    analyse_chat2=[

                    {
        "role": "system",
        "content": """From the conversation history between the user and the LLM assistant, analyze the topic or domain that the user appears to be most interested in or focused on. Your output should strictly be one of the following class names or sub-class names:

                class-cloudsolution
                - Applicationdevelopment
                - Devops
                - cloudcomputing
                class-ERP
                - CRM
                - SAPAMS
                class-Businessintelligence
                - AIandML
                class-Dataanalytics
                - Bigdata
                - Databasemigration
                - Powerplatform

                Do not provide any explanations or additional output beyond one of those class names or sub-class names representing the detected domain of interest."""
        },
        ]
    analyse_chat2.append({"role": "user", "content": f"now classify this chat: {whole_chat2}"})
    chat_completion = client.chat.completions.create(
        messages=analyse_chat2,
        model="Llama3-70b-8192",
        temperature=0,
        max_tokens=100,
        top_p=1,
        stop=None,
        stream=False,
    )
    domain = chat_completion.choices[0].message.content
    return domain   

import csv
from datetime import datetime

def write_to_csv(sentiment, domain):
    # Get the current month and year
    current_month = datetime.now().strftime("%B_%Y")
    
    # Create the filename with the current month and year
    filename = f"{current_month}.csv"
    
    # Check if the file exists, if not, create it with headers
    try:
        with open(filename, 'x', newline='') as csvfile:
            fieldnames = ['Sentiment', 'Domain']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass
    
    # Open the file in append mode and write the sentiment and domain
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentiment, domain])

# Conversational loop
while True:
    user_input = transcribe_and_process_input()
    query = user_input
    context=qdrant.similarity_search(query, k=3)
    if "goodbye" in user_input.lower():
        print("Goodbye!")
        by="goodbye have a great day"
        text_to_speech(by)
        print("Classifying...")
        sentiment=classify_convo(conversation_history)
        print(f"the overall conversation is {sentiment}")
        print("analysing..")
        conv_domain=context_convo(conversation_history)
        print(f"the class is = {conv_domain}")
        print("writing to csv")
        write_to_csv(sentiment,conv_domain)
        break
    print("transcribe done")
    llm_response = get_llm_response(user_input, conversation_history)
    print("llm response aagya")
    text_to_speech(llm_response)



