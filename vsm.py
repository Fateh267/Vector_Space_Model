import glob
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import defaultdict
import string
from collections import Counter
import numpy as np
from collections import OrderedDict

def create_dic(path):
  dic = {}
  files = glob.glob(path)
  for file in files:
      name = file.split('/')[-1]
      with open(file, 'r') as f:
          text = f.read()
      dic[name] = text
  return dic


def normalized_wordlist(dic):
    stop = stopwords.words('english') + list(string.punctuation) + ['\n']
    word_list = []
    for doc in dic.values():
        for word in word_tokenize(doc.lower().strip()):
            if not word in stop and len(word)!=1:
                word_list.append(word)
    return word_list


'''  

# trying to normalise words alongside the dictionary with name as the key and list of normalised words as values 

def word_dic(dic):

    stop = stopwords.words('english') + list(string.punctuation) + ['\n']
    doc_word = {}
    for name, text in dic.items():
        doc_word = []
        for word in word_tokenize(text.lower().strip()): 
            if not word in stop and len(word)!= 1:
                doc_word.append(word)
        doc_word[name] = doc_word
    return doc_word
'''


'''
The defaultdict data structure is used to simplify the code by automatically 
creating a new dictionary for each document name encountered. The for loop is also 
simplified by iterating over both the document name and its corresponding text in the dic dictionary. 
Finally, the function returns a standard Python dictionary 
instead of a defaultdict, since this is usually more convenient for further processing.
'''

def termFrequencyInDoc(word_list, dic):
    tf = defaultdict(dict)
    for name, doc in dic.items():
        for word in word_list:
            tf[name][word] = doc.count(word)
    return dict(tf)

'''
the function uses the defaultdict data structure to automatically 
create a new entry for each word in the vocabulary, 
initialized with a count of zero. It also uses the set function to 
extract unique words from each document, 
avoiding the need to count the same word multiple times within the same document. 
Finally, it returns a standard Python dictionary instead of a defaultdict for convenience.
'''

def wordDocFre(vocab, doc_dict):
    df = defaultdict(int)
    for doc in doc_dict.values():
        words = set(word_tokenize(doc.lower().strip()))
        for word in words:
            if word in vocab:
                df[word] += 1
    return dict(df)



def inverseDocFre(word_list,doc_fre,length):
    idf= {}
    for word in word_list:
        idf[word] = np.log10((length+1) / doc_fre[word])
    return idf


'''
using the defaultdict data structure to simplify the code by automatically creating 
a new dictionary for each document ID encountered. It uses the get method to safely retrieve the TF and IDF values 
for each word, returning 0 if a word is not present in the corresponding dictionary. Finally, it returns a standard
Python dictionary instead of a defaultdict for convenience.
'''

def tfidf(word_list, tf, idf, doc_dict):
    tf_idf_scr = defaultdict(dict)
    for doc_id, doc in doc_dict.items():
        for word in word_list:
            tf_idf_scr[doc_id][word] = tf[doc_id].get(word, 0) * idf.get(word, 0)
    return dict(tf_idf_scr)


'''
this function uses a set instead of a list to store the query vocabulary, 
which eliminates the need for a duplicate check. 
It also includes a check for missing terms in the tfidf dictionary using the in operator to prevent a 
KeyError from occurring.

If a term is not present in the tfidf dictionary for a particular document, the corresponding score contribution will be ignored. This should prevent errors from occurring and allow the function to return a result for any query.
'''

def vectorSpaceModel(query, docs, tfidf):
    query_vocab = set(query.lower().split())

    query_wc = {}
    for word in query_vocab:
        query_wc[word] = query.lower().split().count(word)

    relevance_scores = {}
    for doc_id, doc in docs.items():
        score = 0
        for word in query_vocab:
            if word in tfidf[doc_id]:
                score += query_wc[word] * tfidf[doc_id][word]
        relevance_scores[doc_id] = score

    sorted_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    top_5 = {k: v for k, v in sorted_scores[:5]}

    return top_5


if __name__  == "__main__":

  path = 'Corpus/*.txt'
  docs = create_dic(path)
  w_List = normalized_wordlist(docs)           #returns a list of tokenized words
  vocab = list(set(w_List))                     #returns a list of unique words
  tf_dict = termFrequencyInDoc(vocab, docs)     #returns term frequency
  df_dict = wordDocFre(vocab, docs)             #returns document frequencies
  idf_dict = inverseDocFre(vocab,df_dict,41)     #returns idf scores
  tf_idf = tfidf(vocab,tf_dict,idf_dict,docs)   #returns tf-idf socres

  query1 = 'Developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation'
  query2 = 'Warwickshire, came from an ancient family and was the heiress to some land'
  top1 = vectorSpaceModel(query1, docs,tf_idf)    #returns top 5 documents using VSM
  top2 = vectorSpaceModel(query2, docs,tf_idf)    #returns top 5 documents using VSM
  print('Top 5 Documents for Query 1: \n', top1)
  print('\n')
  print('Top 5 Documents for Query 2: \n', top2)
