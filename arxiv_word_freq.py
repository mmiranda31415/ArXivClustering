import urllib.request
import os
import PyPDF2
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import words
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def import_data(query = 'all:electron AND all:spin', max_results = 100, save = False, verbose = False,):
    """Import text data from Arxiv papers, converst the data into word counts and returns a pandas dataframe"""
    # Set the base URL for the Arxiv API
    base_url = 'http://export.arxiv.org/api/query?'

    # Construct the API query URL
    url = base_url + 'search_query=' + query + '&max_results=' + str(max_results)

    # Send the API request and retrieve the response
    response = urllib.request.urlopen(url).read()

    # Parse the XML response to extract the paper URLs
    from xml.etree import ElementTree
    root = ElementTree.fromstring(response)
    titles_and_urls = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.get('type') == 'application/pdf':
                titles_and_urls.append((title, link.get('href')))

    # download the English words corpus
    nltk.download('words')

    english_words = set(words.words())

    word_freqs_list = []

    paper_titles = []

    for i, (title, url) in enumerate(titles_and_urls):

        valid_words = []

        try:
            # Download the PDF file to a temporary location
            temp_file = 'temp.pdf'
            urllib.request.urlretrieve(url, temp_file)
            
            # Open the PDF file and extract its text content
            reader = PyPDF2.PdfReader(temp_file)
        
            page_nber = len(reader.pages)
            text = ' '
            for j in range(page_nber):
                pages = reader.pages[j]
                text = text + pages.extract_text()
            
            # Split the text content into words and count their frequencies
            pdf_words = re.findall(r'\b\w{3,}\b', text.lower())

            for word in pdf_words:
                if word.lower() in english_words:
                    valid_words.append(word)

            word_freq = dict(Counter(valid_words))
            word_freqs_list.append(word_freq)
            
            paper_titles.append(title)

            # Remove the temporary PDF file
            os.remove(temp_file)
            
            if verbose == True:
                print('Processed paper', i+1)

        except:
            print('Error processing paper:', url)
        
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(word_freqs_list)

    # Fill NaN values with zero
    df = df.fillna(0)

    df = df.transpose()

    if save == True:
    # Save the DataFrame to a CSV file
        df.to_csv('word_freqs.csv')
    
    return df, paper_titles
            
def clustering(df, n_clusters = 2):
    kmeans = KMeans(n_clusters = n_clusters)

    kmeans.fit(df.transpose())

    cluster_labels = kmeans.labels_

    return cluster_labels

def graph(df, cluster_labels, n_clusters = 2 ):
    
    tsne = TSNE( n_components = n_clusters, random_state = 0 )
    X_tsne = tsne.fit_transform( df.transpose() )
    plt.scatter( X_tsne[:, 0], X_tsne[:, 1], c = cluster_labels )
    plt.show()
    
    return None

data = import_data()

cluster_labels = clustering(data[0])

print(cluster_labels)

for title in data[1]:
    print(title)
    
graph(data[0], cluster_labels,)
