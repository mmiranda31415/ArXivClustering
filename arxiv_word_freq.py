import urllib.request
import os
import PyPDF2
import re
import nltk
import pandas as pd

from collections import Counter
from nltk.corpus import words
from sklearn.cluster import KMeans

def import_data(query = 'all:electron AND all:spin', max_results = 100, save = False):
    """Import text data from Arxiv papers, converst the data into word counts and returns a pandas dataframe"""
    # Set the base URL for the Arxiv API
    base_url = 'http://export.arxiv.org/api/query?'

    # Set the search query
    query = query

    # Set the maximum number of results to retrieve
    max_results = max_results

    # Set the output directory
    output_dir = 'arxiv_papers'

    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the API query URL
    url = base_url + 'search_query=' + query + '&max_results=' + str(max_results)

    # Send the API request and retrieve the response
    response = urllib.request.urlopen(url).read()

    # Parse the XML response to extract the paper URLs
    from xml.etree import ElementTree
    root = ElementTree.fromstring(response)
    urls = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.get('type') == 'application/pdf':
                urls.append(link.get('href'))

    for i, url in enumerate(urls):
        paper = os.path.join(output_dir, 'paper.pdf')
        urllib.request.urlretrieve(url, paper)

    # download the English words corpus
    nltk.download('words')

    # create a set of English words
    english_words = set(words.words())

    # Create a list of dictionaries to store the word frequencies for each paper
    word_freqs_list = []

    for i, url in enumerate(urls):
        # create an empty list to store valid words
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
            
            # Remove the temporary PDF file
            os.remove(temp_file)
            
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
    
    return df
            
def clustering(df, n_clusters = 2):
    # Create a KMeans clustering model with 2 clusters
    kmeans = KMeans(n_clusters = n_clusters)

    # Fit the model to the transposed dataframe
    kmeans.fit(df.transpose())

    # Get the cluster labels for each paper
    cluster_labels = kmeans.labels_

    # Add the cluster labels to the original dataframe
    return cluster_labels

print(clustering(import_data()))