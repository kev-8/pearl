
# load in libraries

import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
from time import sleep
from newspaper import Article

# import csv file to filter sources by country 

country_filter = pd.read_csv('/Users/KevinLubin/Desktop/ds/pearl/gdelt_country_sources.csv')
country_filter.head()

# define credentials object for GCP to run queries
credentials = service_account.Credentials.from_service_account_file(
    '/Users/KevinLubin/Desktop/ds/pearl/pearl-336700-0fa91569420d.json')

# Perform query

query = """

    SELECT GlobalEventID, SQLDATE, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone,
    GoldSteinScale, NumMentions, Sourceurl

    FROM `gdelt-bq.full.events`
    
    WHERE (SQLDATE >= 20210101 AND SQLDATE <= 20210115) 
    
    AND (ActionGeo_CountryCode = 'HA' OR Actor1Geo_CountryCode = 'HA' OR Actor2Geo_CountryCode = 'HA')
    
    GROUP BY SQLDATE, GlobalEventID, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone, 
    GoldSteinScale, NumMentions, Sourceurl
    
    ORDER BY SQLDATE, GlobalEventID, EventCode, EventBaseCode, EventRootCode, Quadclass, AvgTone, 
    GoldSteinScale, NumMentions, Sourceurl
    
"""

news_df = pandas_gbq.read_gbq(query, credentials=credentials)


# function to look for the base URL for country specific sources and return filtered df

def get_articles(fips):
    
    # find all sources from chosen country
    temp_sources = country_filter[country_filter.fips == fips]
    
    # create a pattern to search for sources in query result
    source_list = temp_sources.source.to_list()
    pattern = '|'.join(source_list)
    
    # create df with results from chosen country
    articles = news_df[news_df.Sourceurl.str.contains(pattern) == True]
    
    # keep unique articles only
    articles = articles.drop_duplicates(subset=['Sourceurl'])
    
    return(articles)

canada_articles = get_articles('CA')



# creating dictionary to hold the urls and their respective text 
link_text = {}


# function to scrape the text from articles and attach them to df
def get_article_text(df):
    
    # creates a list of URLs to use for nested function
    url_list = df['Sourceurl'].tolist()

    # function to scrape the text from article URLs
    def scraper(url):

        # using the Article function from newspaper package
        article = Article(i)
        try:
            article.download()
            article.parse()
            article.text
            link_text[i] = article.text

        # if the URL is not active, include below text    
        except:
            link_text[i] = 'URL not found.'
        return
    
    # looping through the list of URLs
    for i in url_list:
        scraper(i)
        
    # creating a df from the dict
    temp_df = pd.DataFrame(list(link_text.items()), columns = ['Sourceurl', 'article_text'])
    
    # joining the text into the existing df
    output_df = pd.merge(df, temp_df, on='Sourceurl', how='left')
    
    return(output_df)

final_df = get_article_text(canada_articles)

# saving final df to use in modeling script
final_df.to_csv('./final_df.csv')


