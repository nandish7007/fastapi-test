import os
import fitz
import openai
import uvicorn

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")


# ==========================================================================================
# Utils
# ==========================================================================================

pdf_text = ""


def extract_text_from_pdf(file_path):
    """ Function to extract text from uploaded PDF """

    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    return text


def query_openai(context, question):
    """ Function to query OpenAI with context """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()


# ==========================================================================================
# Endpoints
# ==========================================================================================


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """ Endpoint for PDF upload """

    global pdf_text
    file_location = f"./{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    pdf_text = extract_text_from_pdf(file_location)
    return {"message": "PDF uploaded and text extracted successfully."}


class Query(BaseModel):
    question: str


@app.post("/ask/")
async def ask_question(query: Query):
    """ Endpoint to ask questions """

    global pdf_text
    if not pdf_text:
        return {"error": "No PDF uploaded yet."}

    response = query_openai(pdf_text, query.question)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
