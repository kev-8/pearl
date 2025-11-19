# load in libraries
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import boto3

# load in data 
df = pd.read_csv('/Users/kevin/Desktop/ds/other/test_df.csv')


# use text splitter to chunk articles (chunk_size=500 to be under Cohere embedding model token length)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50,
                                               separators=["\n\n", "\n", " ", ""])

texts = text_splitter.create_documents(df['article_text'])

chunks = text_splitter.split_documents(documents=texts)
print(f'Split into {len(chunks)} chunks')

# convert chunks into json
json_chunks = json.dumps([{'page_content': doc.page_content} for doc in chunks], indent=4)

# create bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")


# define embedding model parameters
input_type = "search_document"
model_id = "cohere.embed-v4:0"

# create JSON
json_params = {
    'texts': json_chunks,
    'input_type': input_type,
}
json_body = json.dumps(json_params)
params = {'body': json_body, 'modelId': model_id}


# invoke the model and print the response
result = client.invoke_model(**params)
response = json.loads(result['body'].read().decode())
print(response)

# use pinecone to store the embeddings
