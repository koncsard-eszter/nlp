from nltk import tokenize
from nltk import everygrams
import json 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os



def load_voc():
    voc = []
    titles_of_keywords = {}
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
            if text in titles_of_keywords:
                titles_of_keywords[text].append(title)
            else:
                titles_of_keywords[text] = [title]
        else:
            vocabular_of_keywords.append(keyword)

    return set(vocabular_of_keywords), titles_of_keywords
        


VOC, TITLES_OF_KEYWORDS = load_voc()


def get_ngrams_of_text(text, num_of_tokens_of_longest_keyword=197):
    sentences = tokenize.sent_tokenize(text)
    ngrams_of_text = []
    for sentence in sentences:
        tokens_in_sentence = tokenize.word_tokenize(sentence)
        ngrams_of_sentence = [" ".join(ng) for ng in everygrams(tokens_in_sentence, 1, num_of_tokens_of_longest_keyword)]
        for ngram in ngrams_of_sentence:
            if ngram in VOC:
                ngrams_of_text.append(ngram)

    return ngrams_of_text


def calculate_frequencies_of_terms(ngrams):
    freq_of_terms_in_overall_docs_as_keyword = {}
    freq_of_terms_in_overall_docs = {}
    for ngram in set(ngrams):
        freq_of_terms_in_overall_docs_as_keyword[ngram] = 0
        freq_of_terms_in_overall_docs[ngram] = 0

    with open('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\cleaned_data4.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            # check presence
            ngrams_of_art = set(data.get("valid_ngrams", []))
            for ngram in ngrams_of_art:
                if ngram in freq_of_terms_in_overall_docs:
                    freq_of_terms_in_overall_docs[ngram] += 1

            # check presence as keyword
            links = set(data.get("links", []))
            for link in set(links):
                searched_token = link
                if "*=_=**=_=*" in link:
                    link_title, link_text = link.split("*=_=**=_=*")
                    searched_token = link_text
                if searched_token in freq_of_terms_in_overall_docs_as_keyword:
                    freq_of_terms_in_overall_docs_as_keyword[link] += 1   
    return freq_of_terms_in_overall_docs, freq_of_terms_in_overall_docs_as_keyword


def evaluate_ngrams_kp_method(ngrams):
    # calculate frequencies
    freq_of_terms_in_overall_docs, freq_of_terms_in_overall_docs_as_keyword = calculate_frequencies_of_terms(ngrams)    
    
    keyphrase_values = {}
    for ngram in ngrams:
        keyphrase_values[ngram] = freq_of_terms_in_overall_docs_as_keyword[ngram]/freq_of_terms_in_overall_docs[ngram]

    return keyphrase_values


def select_keywords_by_kp(text, keyphrase_values_of_ngrams):
    selected_keywords = []
    # sort keywords by their keyphrase values
    k = np.array(list(keyphrase_values_of_ngrams.keys()))
    v = np.array(list(keyphrase_values_of_ngrams.values()))
    sorted_indexes = np.argsort(v)[::-1]  # descending order
    sorted_keys = k[sorted_indexes]

    # the number of keywords = number of terms in the article * 6/100
    terms_of_doc = len(tokenize.word_tokenize(text))
    ideal_num_of_keywords = int(terms_of_doc*6/100)

    current_num_of_keywords = len(keyphrase_values_of_ngrams)
    if ideal_num_of_keywords < current_num_of_keywords:
        selected_keywords = sorted_keys[:ideal_num_of_keywords]
    else:
        selected_keywords = k

    return selected_keywords


def cosine_similarity_of_two_string(txt1, txt2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([txt1, txt2])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim_matrix[0][1]
    


def complete_keywords(text, keywords):
    completed_keywords = []
    title_of_wiki_docs = np.load('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\titles_of_wiki_docs.npy')
    with open('C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\docs_for_check_cos_sim.json', 'r') as f:
        doc_text_for_cos_sim = json.load(f)
    for keyword in keywords:
        if keyword in TITLES_OF_KEYWORDS:
            possible_titles = TITLES_OF_KEYWORDS[keyword]
            text_of_doc = text
            cos_sim_with_doc = {}
            for t in possible_titles:
                # if there is no doc in my dataset with this title --> skip
                if t not in title_of_wiki_docs:
                    continue
                # find the text of the article with the given title
                text_of_possible_doc = doc_text_for_cos_sim[t]
                cos_sim = cosine_similarity_of_two_string(text_of_doc, text_of_possible_doc)
                cos_sim_with_doc[t] = cos_sim
            
            # if no doc find by title then add the ngram itself
            if len(cos_sim_with_doc) == 0:
                completed_keywords.append(f"{keyword}")
            else:
                # get the title of doc with max cos sim
                doc_title_by_cos_sim = max(cos_sim_with_doc, key=cos_sim_with_doc.get)
                if doc_title_by_cos_sim == keyword: 
                    completed_keywords.append(f"{keyword}")
                else:
                    completed_keywords.append(f"{doc_title_by_cos_sim}*=_=**=_=*{keyword}" )
        else:
            completed_keywords.append(f"{keyword}")
    return completed_keywords



def wikify(text):
    ngrams = get_ngrams_of_text(text)
    print("ngrams")
    print(ngrams)
    keyphrase_values_of_ngrams = evaluate_ngrams_kp_method(ngrams)
    print("keyphrase values:")
    print(keyphrase_values_of_ngrams)
    selected_keywords = select_keywords_by_kp(text, keyphrase_values_of_ngrams)
    print("selected keywords:")
    print(selected_keywords)
    completed_keywords = complete_keywords(text, selected_keywords)
                     
    # create 'links'
    completed_keywords.sort(key=len, reverse=True)
    for keyword in completed_keywords:
        token_in_text = keyword
        substitution = f"[[{keyword}]]"
        text = text.replace(token_in_text, substitution, 1)   
    
    return text
     


        
def main():
    txtfile_path = "C:\\Users\\koncs\\Master\\Master_second_semester\\NLP\\Labs\\nlp_01\\nlp02\\text_to_wikify.txt"        
    with open(txtfile_path, encoding='utf-8') as file:
        text = file.read()
    wikified_text = wikify(text)

    # save wikified text
    filename, extension = os.path.basename(txtfile_path).split('.')
    directory = os.path.dirname(txtfile_path)
    new_filename = f"{filename}_wikified.{extension}"
    new_full_path = os.path.join(directory, new_filename)
    with open(new_full_path, 'w', encoding='utf-8') as file:
        file.write(wikified_text)



if __name__ == '__main__':
    main()
