
#load in libraries and data

import pandas as pd
import numpy as np
import spacy
# from spacy import displacy
from collections import Counter


test_df = pd.read_csv('./test_df.csv')
nlp = spacy.load("en_core_web_lg")



# ner
# displacy.serve(doc, style="ent")


# available_labels = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT',
#                     'WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT',
#                     'MONEY','QUANTITY','ORDINAL','CARDINAL']


# adding additional empty columns to dataframe
test_df['top_entity_names'] = np.nan
test_df['top_entity_names'] = test_df['top_entity_names'].astype('object')
test_df['top_entity_labels'] = np.nan
test_df['top_entity_labels'] = test_df['top_entity_labels'].astype('object')


# creating empty lists and getting index for function
entity_name = []
entity_label = []
df_index = test_df.index.values.tolist()

def get_entity_info(df):
    
    for row in df_index:
        doc = nlp(test_df.article_text[row])

        # getting entity info and assigning to lists 
        for ent in doc.ents:
            entity_name.append(ent.text)
            entity_label.append(ent.label_)
        
        top_ent_names = Counter(entity_name).most_common(3)
        top_ent_labels = Counter(entity_label).most_common(3)

        # adding top 3 of each to df
        df.at[row,'top_entity_names'] = top_ent_names
        df.at[row,'top_entity_labels'] = top_ent_labels

    return(df)

ner_df = get_entity_info(test_df)

# TODO: figure out how to skip over rows where URL not found

# topic modeling






# sentiment analysis
# token_list = [token for token in doc]
# filtered_tokens = [token for token in doc if not token.is_stop]
# lemmatizer = nlp.get_pipe('lemmatizer')
# lemmas = [token.lemma_ for token in doc]
# filtered_tokens[0].vector


