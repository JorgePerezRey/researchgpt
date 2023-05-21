from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import requests
import redis
from _md5 import md5
from typing import Optional

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# db=redis.from_url(os.environ['REDISCLOUD_URL'])
db = redis.StrictRedis(host='localhost', port=6379, db=0)


class Chatbot():
    
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                    })

            _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ''
            processed_text = []

            for t in page_text:
                if t['fontsize'] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 2000:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                        blob_font_size = None
                        blob_text = ''
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                    blob_font_size = t['fontsize']
                    blob_text = t['text']
                paper_text += processed_text
        print("Done parsing paper")
        # print(paper_text)
        return paper_text

    def paper_df(self, pdf):
        print('Creating dataframe')
        filtered_pdf= []
        for row in pdf:
            if len(row['text']) < 30:
                continue
            filtered_pdf.append(row)
        df = pd.DataFrame(filtered_pdf)
        # print(df.shape)
        # remove elements with identical df[text] and df[page] values
        df = df.drop_duplicates(subset=['text', 'page'], keep='first')
        df['length'] = df['text'].apply(lambda x: len(x))
        print('Done creating dataframe')
        return df

    def calculate_embeddings(self, df):
        print('Calculating embeddings')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        embedding_model = "text-embedding-ada-002"
        embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        df["embeddings"] = embeddings
        print('Done calculating embeddings')
        return df

    def search_embeddings(self, df, query, n=3, pprint=True):
        query_embedding = get_embedding(
            query,
            engine="text-embedding-ada-002"
        )
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
        
        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
        results = results.head(n)
        global sources 
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)
    
    def create_prompt(self, df, user_input):
        result = self.search_embeddings(df, user_input, n=3)
        print(result)
        prompt = """You are a large language model whose expertise is reading and summarizing scientific papers. 
        You are given a query and a series of text embeddings from a paper in order of their cosine similarity to the query.
        You must take the given embeddings and return a very detailed summary of the paper that answers the query.
            
            Given the question: """+ user_input + """
            
            and the following embeddings as data: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """

            Return a detailed answer based on the paper:"""

        print('Done creating prompt')
        return prompt

    def gpt(self, prompt):
        print('Sending request to GPT-3')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        r = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.4, max_tokens=1500)
        answer = r.choices[0]['text']
        print('Done sending request to GPT-3')
        response = {'answer': answer, 'sources': sources}
        return response

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_pdf")
async def process_pdf(request: Request):
    print("Processing pdf")
    print(request)
    body = await request.body()
    key = md5(body).hexdigest()
    print(key)
    if db.get(key) is not None:
        return JSONResponse({"key": key})
    file = body
    pdf = PdfReader(BytesIO(file))
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    if db.get(key) is None:
        db.set(key, df.to_json())
    # print(db.set(key, df.to_json()))
    # print(db.get(key))
    print("Done processing pdf")
    return JSONResponse({"key": key})

@app.post("/download_pdf")
async def download_pdf(url: str):
    chatbot = Chatbot()
    r = requests.get(str(url))
    key = md5(r.content).hexdigest()
    if db.get(key) is not None:
        return JSONResponse({"key": key})
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.parse_paper(pdf)
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    if db.get(key) is None:
        db.set(key, df.to_json())
    print("Done processing pdf")
    return JSONResponse({"key": key})

@app.post("/reply")
async def reply(request: Request):
    data = await request.json()
    key = data.get('key')
    query = data.get('query')
    
    chatbot = Chatbot()
    query = str(query)
    df = pd.read_json(BytesIO(db.get(key)))
    print(df.head(5))
    prompt = chatbot.create_prompt(df, query)
    response = chatbot.gpt(prompt)
    print(response)
    return JSONResponse(response)
