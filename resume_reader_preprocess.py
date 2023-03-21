from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import pandas as pd
from nltk import word_tokenize , sent_tokenize
import re
import numpy as np
import nltk
from pdfminer.pdfpage import PDFPage

nltk.download('punkt')

path = ''
stopwords = pd.read_csv('stopwords.csv')
stopwords = stopwords['Token'].to_list()


import re


### EXTACT LIKK_EMAIL
def extract_link_email(text):
    """
    Extract Links
    """
    URL,EMAIL =[],[]
    [[URL.append(x) for x in re.findall(r'http\S+', text)],[URL.append(x) for x in re.findall(r'bit.ly/\S+', text)],   [URL.append(x) for x in re.findall(r'www\S+', text)]]
    [EMAIL.append(x) for x in re.findall(r'[\w\.-]+@[\w\.-]+', text)] 
    return {'URL':np.unique(URL),'EMAIL':np.unique(EMAIL)}


# extract links and emails from the resume 

def email_address(text):
  """   remove email """
  email = re.compile(r'[\w\.-]+@[\w\.-]+')
  return email.sub(r'',text)


def remove_links(text):
    '''Takes a string and removes web links from it'''
    text = re.sub(r'http\S+', '', text) # remove http links
    text = re.sub(r'bit.ly/\S+', '', text) # rempve bitly links
    text = re.sub(r'www\S+','',text)
    text =text.strip('[link]') # remove [links]
    return text

def clean_html(text):  
  html = re.compile('<.*?>')#regex
  return html.sub(r'',text)

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, ' ', text)

def punct(text):
    # define punctuation
    text = re.sub(pattern = "\W",
        repl = " ",
        string = text)
    return str(text)

def stopword_remover(text):
    text_tokens = word_tokenize(text)
    text_tokens = [word for word in text_tokens if not word in stopwords]
    text = ' '.join(text_tokens)
    return text

def preprocess(text):
    text =  email_address(text)
    text =  remove_links(text)
    text = remove_special_characters(text)
    text = punct(text)
    return stopword_remover(text).lower()


def arr_to_text(array):
    """
    Array to text
    """
    text =''
    for arr in array:
        text = text + ' '+arr[0]
    return text

def remove_multiple_spaces(text):
  '''
  remove multiple white space in string
  '''
  text=  re.sub(' +', ' ',text)
  return text

def open_pdf_file(file_name):

    ## open the pdf file
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    pagenums = set()
    infile = open(file_name, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()

    result = []
    
    for line in text.split('\n'):
        line2 = line.strip()
        if line2 != '':
            
            result.append(line2)
    df = pd.DataFrame(result,columns=['Text'])
    #df['Text'] = df['Text'].apply(lambda x: (preprocess(x))  )
    df.fillna(method="ffill",inplace=True)
    df['Text'] = df['Text'].apply(lambda x:  np.nan if x ==' ' else x )
    df['Text'] = df['Text'].apply(lambda x:  np.nan if x =='' else x )
    df.dropna(inplace=True)
    arr = [x for x in df['Text'].values.tolist()]
    text = []
    for sent in arr:
        text.append(sent_tokenize(sent))
        
    return arr_to_text(text)
