import fitz  # PyMuPDF
from transformers import pipeline

# Load spaCy's English model
import spacy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        pdf_reader = fitz.open(pdf_file_path)
        for page_num in range(pdf_reader.page_count):
            page = pdf_reader[page_num]
            text += page.get_text()
    except Exception as e:
        print(f"PDF Chatbot: Error extracting text from PDF - {e}")
    return text

# Function to process user input and provide responses
def chat_with_pdf(pdf_file_path):
    print("PDF Chatbot: Hello! What would you like to know about the PDF?")
    pdf_text = extract_text_from_pdf(pdf_file_path)

    # Initialize the BERT-based question-answering pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    if not pdf_text:
        print("PDF Chatbot: Error extracting text from the PDF. Please try again.")
        return

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("PDF Chatbot: Goodbye!")
            break

        if not user_input:
            print("PDF Chatbot: Please enter a valid question.")
            continue

        try:
            # Process user input using spaCy
            user_doc = nlp(user_input)

            # Apply question-answering on the entire PDF text
            answer = qa_pipeline(context=pdf_text, question=user_input)
            confidence_threshold = 0.2  # Adjust as needed

            if answer['score'] > confidence_threshold:
                # Implement post-processing if necessary
                # Example: answer['answer'] = post_process(answer['answer'])
                print("PDF Chatbot:", answer['answer'])
            else:
                print("PDF Chatbot: I couldn't find relevant information in the PDF.")
        except Exception as e:
            print(f"PDF Chatbot: Error processing question - {e}")

if __name__ == "__main__":
    pdf_file_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\RNSIT\\7 SEM\\final_year_project\\Speaklink\\data\\Generative AI.pdf"  # Replace with the path to your PDF file
    chat_with_pdf(pdf_file_path)
