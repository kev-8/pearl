
#load in libraries

import pandas as pd
import spacy
from spacy import displacy

test_df = pd.read_csv('./test_df.csv')

nlp = spacy.load("en_core_web_sm")
doc = nlp(test_df.article_text[4])

displacy.serve(doc, style="ent")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)



