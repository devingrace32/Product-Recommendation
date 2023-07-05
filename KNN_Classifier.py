from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nsmallest
from operator import itemgetter
import numba
from fastdist import fastdist


def preprocess(text):
    text=text.lower() 
    text=re.sub(r'[^a-z ]+'," ",text)
    words = re.split("\\s+",text)
    stop = stopwords.words('english')   
    words = [word for word in words if word not in stop]
    words = [word for word in words if len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    strreturn = ' '.join([lemmatizer.lemmatize(w) for w in words])
    return strreturn

'''
#function to calculate euclidean distance bt two vectors 
def euclid(a,b):
    distance = 0.0
    for i in range(len(a)):
        distance += (a[i] - b[i])**2
    return sqrt(distance)
'''

#takes training data, a row, and number of neighbors
#returns sentiment of the majority its n nearest neighbors
def whose_my_neighbor(train,test_row, num_neighbors, train_Y):
    #dist_mat = sklearn.metrics.pairwise_distances(pd.concat([train, test_row.to_frame().T]))
    ##Optemized euclidean distance 
    euclid_row = fastdist.vector_to_matrix_distance(np.array(test_row), np.array(train),fastdist.euclidean, "euclidean")
    idx, _ = zip(*nsmallest(num_neighbors + 1, enumerate(euclid_row), key=itemgetter(1)))
    cnt = 0
    for i in range(1,len(idx)):

        cnt += train_Y[idx[i]]

    #classed = [train_y[i] for i in sorted(range(len(euclid_row)), key= lambda k: euclid_row[k])[1:num_neighbors+1]]
    if(cnt < num_neighbors/2.0):
        return "-1"
    else:
        return "+1"

##create corpus from test and training data
training= pd.read_csv("C:\\Users\\devin\\Downloads\\train_reviews.txt",
    names=['Sentiment', 'Review'], header=None, sep="1\t").dropna().reset_index(drop=True)
training['Sentiment'] = np.where((training.Sentiment == '-'), 0, 1)
testing = pd.read_csv("C:\\Users\\devin\\Downloads\\tests.txt", names=['Review'], header=None, sep="1\t").dropna().reset_index(drop=True)
vec = TfidfVectorizer(max_features = 2000, preprocessor=preprocess, min_df=5, max_df=0.75)
training_x = training.drop('Sentiment',1)
training_y = training.Sentiment


train_i = len(training_x)

corpus = pd.concat([training_x, testing])
mat = vec.fit_transform(corpus['Review'])
corpus = pd.DataFrame(mat.toarray(), columns=vec.get_feature_names())

#reseparate into train/test
train_x = corpus.iloc[:train_i,:]
train_y = training_y.iloc[:train_i]
test_x = corpus.iloc[train_i:,:]

#write to output file
with open('output1.txt', 'w') as f:
    for i in range(len(test_x)):
        neighbors = whose_my_neighbor(train_x, test_x.iloc[i], 38, train_y)
        print(i)
        f.write(neighbors + "\n")
print("done")

f.close()