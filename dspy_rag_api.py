import json

import dspy
import litellm
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy RAG",
    version="1.0.0"
)

persist_directory = "embeddings_db"
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
retriever = vectordb.as_retriever()

def retrieve(inputs):
  return [doc.page_content for doc in retriever.invoke(inputs["question"])]

class COT_RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")
        
    def forward(self, question):
        context = retrieve({"question": question})
        return self.respond(context=context, question=question) 
    
    
# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure language model and 'asyncify' DSPy program.
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
dspy_program = COT_RAG()
dspy_program.load('./dspy_docs_rag/cot_rag.json')
streaming_dspy_program = dspy.streamify(dspy_program)
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/stream")
async def stream(question: Question):
    async def generate():
        try:
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    try:
                        data = {"prediction": value.labels().toDict()}
                    except AttributeError:
                        data = {"error": "Prediction object missing required attributes"}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
