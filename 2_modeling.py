# load in libraries and data
import pandas as pd
import numpy as np
import matplotlib as plt
import spacy

# from spacy import displacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pyLDAvis.gensim_models
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from collections import Counter


temp_df = pd.read_csv("./test_df.csv")
nlp = spacy.load("en_core_web_lg")


#%% ner
# displacy.serve(doc, style="ent")


# available_labels = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT',
#                     'WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT',
#                     'MONEY','QUANTITY','ORDINAL','CARDINAL']


# adding additional empty columns to dataframe
temp_df["top_entity_names"] = np.nan
temp_df["top_entity_names"] = temp_df["top_entity_names"].astype("object")
temp_df["top_entity_labels"] = np.nan
temp_df["top_entity_labels"] = temp_df["top_entity_labels"].astype("object")


# creating empty lists and getting index for function
entity_name = []
entity_label = []
df_index = list(temp_df.index.values)


def get_entity_info(df):

    for row in df_index:
        doc = nlp(df.article_text[row])

        if doc.text == "URL not found.":

            df.at[row, "top_entity_names"] = "NA"
            df.at[row, "top_entity_labels"] = "NA"

        else:

            # getting entity info and assigning to lists
            for ent in doc.ents:
                entity_name.append(ent.text)
                entity_label.append(ent.label_)

            top_ent_names = Counter(entity_name).most_common(3)
            top_ent_labels = Counter(entity_label).most_common(3)

            # adding top 3 of each to df
            df.at[row, "top_entity_names"] = top_ent_names
            df.at[row, "top_entity_labels"] = top_ent_labels

    return df


ner_df = get_entity_info(temp_df)


#%% topic modeling
nlp.disable_pipes("ner")
test_df = ner_df.article_text

# Tags I want to remove from the text
removal = ["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE", "NUM", "SYM"]

tokens = []
for summary in nlp.pipe(test_df):
    proj_tok = [
        token.lemma_.lower()
        for token in summary
        if token.pos_ not in removal and not token.is_stop and token.is_alpha
    ]
    tokens.append(proj_tok)

ner_df["tokens"] = tokens
dictionary = Dictionary(ner_df["tokens"])
# using the no below value of 1 for small test df, will have to increase for larger df
dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=1000)
corpus = [dictionary.doc2bow(doc) for doc in ner_df["tokens"]]


# when using larger dataset, run below code to determine the optimal number of topics
# topics = []
# score = []

# for i in range(1,10,1):
#     lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10,
#     num_topics=i, workers = 2, passes=10, random_state=100)

#     cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary,
#     coherence='u_mass')

#     topics.append(i)
#     score.append(cm.get_coherence())

# _=plt.plot(topics, score)
# _=plt.xlabel('Number of Topics')
# _=plt.ylabel('Coherence Score')
# plt.show()


lda_model = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    iterations=50,
    num_topics=10,
    workers=2,
    passes=10,
)

lda_model.print_topics(-1)

lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
# view results in browser
pyLDAvis.save_html(lda_display, "lda.html")

# adding topics to dataframe
topic_modeling_df = ner_df
topic_modeling_df["topic"] = [
    sorted(lda_model[corpus][text])[0][0]
    for text in range(len(topic_modeling_df["article_text"]))
]

# count frequency of each topic
topic_modeling_df.topic.value_counts()


#%% sentiment analysis
nlp.add_pipe("spacytextblob")

# adding additional empty columns to dataframe
topic_modeling_df["polarity"] = np.nan
topic_modeling_df["polarity"] = topic_modeling_df["polarity"].astype("object")
topic_modeling_df["subjectivity"] = np.nan
topic_modeling_df["subjectivity"] = topic_modeling_df["subjectivity"].astype("object")

# creating empty lists and getting index for function
polarity_value = []
subjectivity_value = []
df_index = list(topic_modeling_df.index.values)


def get_sentiment(df):

    for row in df_index:
        doc_sent = nlp(df.article_text[row])

        if doc_sent.text == "URL not found.":

            df.at[row, "polarity"] = "NA"
            df.at[row, "subjectivity"] = "NA"

        else:

            # getting sentiment and assigning to lists
            polarity_value = doc_sent._.blob.polarity
            subjectivity_value = doc_sent._.blob.subjectivity

            # adding to df
            df.at[row, "polarity"] = polarity_value
            df.at[row, "subjectivity"] = subjectivity_value

    return df


sentiment_df = get_sentiment(topic_modeling_df)

# saving final df to use in visualization script
sentiment_df.to_csv("./modeling_df.csv")
