# load in libraries
import pandas as pd
import pandas_gbq
from google.oauth2 import service_account
from newspaper import Article
import boto3
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


# define credentials object for GCP to run queries
credentials = service_account.Credentials.from_service_account_file(
    "./pearl-336700-91f798ac4dc7.json"
)

# Perform query
query = """
    SELECT GlobalEventID, SQLDATE, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone, 
    GoldSteinScale, NumMentions, Sourceurl
    
    FROM `gdelt-bq.full.events`
    
    WHERE (SQLDATE >= 20130101 AND SQLDATE <= 20231231) 
    
    AND (ActionGeo_CountryCode = 'HA' OR Actor1Geo_CountryCode = 'HA' OR Actor2Geo_CountryCode = 'HA')
    
    GROUP BY SQLDATE, GlobalEventID, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone, 
    GoldSteinScale, NumMentions, Sourceurl
    
    ORDER BY SQLDATE, GlobalEventID, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone, 
    GoldSteinScale, NumMentions, Sourceurl
    """

news_df = pandas_gbq.read_gbq(query, credentials=credentials)

# creating dictionary to hold the urls and their respective text
link_text = {}


# function to scrape the text from articles and attach them to df
def get_article_text(df):

    # keep unique articles only
    df = df.drop_duplicates(subset=["Sourceurl"])

    # creates a list of URLs to use for nested function
    url_list = df["Sourceurl"].tolist()

    # function to scrape the text from article URLs
    def scraper(url):

        # using the Article function from newspaper package
        article = Article(i)
        try:
            article.download()
            article.parse()
            link_text[i] = article.text

        # if the URL is not active, include below text
        except:
            link_text[i] = "URL not found."
        return

    # looping through the list of URLs
    for i in url_list:
        scraper(i)

    # creating a df from the dict
    temp_df = pd.DataFrame(
        list(link_text.items()), columns=["Sourceurl", "article_text"]
    )

    # joining the text into the existing df
    output_df = pd.merge(df, temp_df, on="Sourceurl", how="left")

    return output_df


articles = get_article_text(news_df)

# saving raw input data
# articles.to_csv("./articles.csv", index=False)


# use text splitter to chunk articles (chunk_size=500 to be under Cohere embedding model token length)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50,
                                               separators=["\n\n", "\n", " ", ""])

texts = text_splitter.create_documents(df['article_text'])

chunks = text_splitter.split_documents(documents=texts)
print(f'Split into {len(chunks)} chunks')

# convert chunks into json


# create bedrock client
# bedrock = boto3.client(service_name='bedrock-runtime')

# define embedding model parameters
# input_type = "search_document"
# model_id = "cohere.embed-english-v3" # "cohere.embed-multilingual-v3"

# # create JSON
# json_params = {
#     'texts': ,
#     'input_type': ,
# }
# json_body = json.dumps(json_params)
# params = {'body': json_body, 'modelId': model_id}

# # invoke the model and print the response
# result = bedrock.invoke_model(**params)
# response = json.loads(result['body'].read().decode())
# print(response)
#



# use pinecone to store the embeddings
