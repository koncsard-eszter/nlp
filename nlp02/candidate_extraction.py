import orjson # pip install orjson
import concurrent.futures

from nltk import tokenize
from nltk import everygrams
import json 
import re
import numpy as np



def load_voc():
    voc = []
    with open('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\vocabular.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # line.strip() removes \n, \r, \t, and spaces from both ends
            clean_line = line.strip()
            
            if not clean_line:
                continue
            
            voc.append(clean_line)

    # handle when the text of a link is not the same as the title
    vocabular_of_keywords = []
    for keyword in voc:
        if "*=_=*=_=*" in keyword:
            title, text = keyword.split("*=_=*=_=*")
            vocabular_of_keywords.append(text)
        else:
            vocabular_of_keywords.append(keyword)

    return vocabular_of_keywords
        


VOC = set(load_voc())


def get_ngrams_of_article(article, num_of_tokens_of_longest_keyword=197):
    sentences = tokenize.sent_tokenize(article)
    ngrams_of_article = []
    for sentence in sentences:
        tokens_in_sentence = tokenize.word_tokenize(sentence)
        ngrams_of_sentence = [" ".join(ng) for ng in everygrams(tokens_in_sentence, 1, num_of_tokens_of_longest_keyword)]
        for ngram in ngrams_of_sentence:
            if ngram in VOC:
                ngrams_of_article.append(ngram)

    return ngrams_of_article



def is_text(line):
    footnote_parts = ["External links", "Further reading", "References", "Notes", "See also", "New items", "Sources", "Website"]
    for footnote_part in footnote_parts:
        if re.search(fr"^{footnote_part}", line.strip()):
            return False
    return True


def process_article(article):
    lines_of_article = article.splitlines(False)
    i = 0
    while i < len(lines_of_article) and is_text(lines_of_article[i]):
        i+=1

    # delete footnote part
    cleaned_article = '\n'.join(lines_of_article[:i])

    return cleaned_article

        

def process_single_line(line):
    try:
        data = orjson.loads(line)
        # Drop the raw_text here to save space and I/O

        cleaned_text = process_article(data.get("text", ""))

        valid_ngrams = get_ngrams_of_article(cleaned_text)

        result = {
            "title": data.get("title", ""),
            "cleaned_text": cleaned_text,
            "valid_ngrams": valid_ngrams,
            "links": data.get("links", "")
        }
        return orjson.dumps(result) + b"\n"
    except Exception:
        print("error")
        return b"" # Handle corrupted lines safely
    

def main():
    with open('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\processed_wiki.jsonl', 'rb') as source, \
         open('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\cleaned_data4.jsonl', 'wb') as target:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for processed_line in executor.map(process_single_line, source, chunksize=100):
                if processed_line:
                    target.write(processed_line)
            

if __name__ == '__main__':
    main()
