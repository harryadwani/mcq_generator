# -*- coding: utf-8 -*-
from flask import Flask,request,jsonify
from flask_cors import CORS
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
from summarizer import Summarizer
import pke
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import requests
import re
import random

app = Flask(__name__)
CORS(app)
res = ''
res_dict={}
full_text = ''
summarized_text = ''
keyword_sentence_mapping = ''
filtered_keys = []
orig_word = ''
orig_words = {}
orig_word_index = 1

@app.route("/")
def predict():
    computation()
    global res_dict
    return res_dict

import nltk

nltk.download('stopwords')
nltk.download('popular')

"""



    ## BERT Extractive Summarizer
    Summarize the text using BERT extractive summarizer. This is used to find important sentences and useful sentences from the complete text."""

def computation():

    def input_file_loop(filename='input.txt'):
        global summarized_text
        global full_text
        print('Reading from file... Please wait till confirmation')
        f = open(filename, "r")
        full_text = f.read()
        model = Summarizer()
        result = model(full_text, min_length=60, max_length=500, ratio=0.4)
        summarized_text = ''.join(result)
        print('Done reading and processing, please proceed to next step.')


    # print(full_text)

    """## Keyword Extraction
    Get important keywords from the text and filter those keywords that are present in the summarized text.
    """


    def keyword_extraction_loop():

        def get_nouns_multipartite(text):
            out = []
            extractor = pke.unsupervised.MultipartiteRank()
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'PROPN'}
            # pos = {'VERB', 'ADJ', 'NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.load_document(input=text, stoplist=stoplist)
            extractor.candidate_selection(pos=pos)
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                          threshold=0.75,
                                          method='average')
            keyphrases = extractor.get_n_best(n=20)
            print(keyphrases)

            for key in keyphrases:
                out.append(key[0])

            return out

        global summarized_text
        global full_text
        keywords = get_nouns_multipartite(full_text)
        # print(keywords)
        global filtered_keys
        for keyword in keywords:
            if keyword.lower() in summarized_text.lower():
                filtered_keys.append(keyword)

        # print(filtered_keys)


    """## Sentence Mapping
    For each keyword get the sentences from the summarized text containing that keyword. 
    """

    def sentence_mapping_loop():


        def tokenize_sentences(text):
            sentences = [sent_tokenize(text)]
            sentences = [y for x in sentences for y in x]
            # Remove any short sentences less than 20 letters.
            sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
            return sentences

        def get_sentences_for_keyword(keywords, sentences):
            keyword_processor = KeywordProcessor()
            keyword_sentences = {}
            for word in keywords:
                keyword_sentences[word] = []
                keyword_processor.add_keyword(word)
            for sentence in sentences:
                keywords_found = keyword_processor.extract_keywords(sentence)
                for key in keywords_found:
                    keyword_sentences[key].append(sentence)

            for key in keyword_sentences.keys():
                values = keyword_sentences[key]
                values = sorted(values, key=len, reverse=True)
                keyword_sentences[key] = values
            return keyword_sentences

        global filtered_keys
        global keyword_sentence_mapping
        sentences = tokenize_sentences(summarized_text)
        print(sentences)
        keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)

        # print(keyword_sentence_mapping)


    # print(keyword_sentence_mapping)

    """## Generate MCQ
    Get distractors (wrong answer choices) from Wordnet/Conceptnet and generate MCQ Questions.
    """


    def final_mcq_gen_loop():

        # Distractors from Wordnet
        def get_distractors_wordnet(syn, word):
            distractors = []
            word = word.lower()
            global orig_word
            global orig_word_index
            global orig_words
            orig_word = word
            orig_words[orig_word_index]=orig_word
            orig_word_index +=1

            if len(word.split()) > 0:
                word = word.replace(" ", "_")
            hypernym = syn.hypernyms()
            if len(hypernym) == 0:
                return distractors
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name()
                # print ("name ",name, " word",orig_word)
                if name == orig_word:
                    continue
                name = name.replace("_", " ")
                name = " ".join(w.capitalize() for w in name.split())
                if name is not None and name not in distractors:
                    distractors.append(name)
            return distractors

        def get_wordsense(sent, word):
            word = word.lower()

            if len(word.split()) > 0:
                word = word.replace(" ", "_")

            synsets = wn.synsets(word, 'n')
            if synsets:
                wup = max_similarity(sent, word, 'wup', pos='n')
                adapted_lesk_output = adapted_lesk(sent, word, pos='n')
                lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
                return synsets[lowest_index]
            else:
                return None

        # Distractors from http://conceptnet.io/
        def get_distractors_conceptnet(word):
            word = word.lower()
            original_word = word
            if (len(word.split()) > 0):
                word = word.replace(" ", "_")
            distractor_list = []
            url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
            obj = requests.get(url).json()

            for edge in obj['edges']:
                link = edge['end']['term']

                url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
                obj2 = requests.get(url2).json()
                for edge in obj2['edges']:
                    word2 = edge['start']['label']
                    if word2 not in distractor_list and original_word.lower() not in word2.lower():
                        distractor_list.append(word2)

            return distractor_list

        key_distractor_list = {}
        global keyword_sentence_mapping
        for keyword in keyword_sentence_mapping:
            try:
                wordsense = get_wordsense(keyword_sentence_mapping[keyword][0], keyword)
            except:
                pass
            if wordsense:
                distractors = get_distractors_wordnet(wordsense, keyword)
                if len(distractors) == 0:
                    distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
            else:

                distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors

        index = 1
        print("#############################################################################")
        print(
            "NOTE::::::::  Since the algorithm might have errors along the way, wrong answer choices generated might not be correct for some questions. ")
        print("#############################################################################\n\n")
        for each in key_distractor_list:
            try:
                sentence = keyword_sentence_mapping[each][0]
            except:
                continue
            pattern = re.compile(each, re.IGNORECASE)
            output = pattern.sub(" _______ ", sentence)
            # print("%s)" % (index), output)
            global res
            global res_dict
            res +=(("%s)" % (index))+ output + '\n')

            # print(res_dict,"res_dict")
            choices = [each.capitalize()] + key_distractor_list[each]
            try:
                top4choices = choices[:4]
            except:
                top4choices = choices[:]

            random.shuffle(top4choices)
            optionchoices = ['a', 'b', 'c', 'd']
            options_dict = {}
            for idx, choice in enumerate(top4choices):
                # print("\t", optionchoices[idx], ")", " ", choice)
                options_dict[optionchoices[idx]] = choice
                res += "\t" + optionchoices[idx] + ")" + " " + choice + '\n'
            # print("\nMore options: ", choices[4:20], "\n\n")
            # print(options_dict, "options_dict")
            res_dict[index] = {"question": (("%s)" % (index)) + output)}
            res_dict[index]["options"] = options_dict
            global orig_word
            global orig_words
            res_dict[index]["correct_option"] = orig_words[index]
            index = index + 1


    file_name_prompt = "aeen101.pdf"
    file_name_txt = "converted.txt"
    import PyPDF2
    with open(file_name_prompt, 'rb') as pdf_file, open(file_name_txt, 'w') as text_file:
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        number_of_pages = read_pdf.getNumPages()
        for page_number in range(number_of_pages):  # use xrange in Py2
            page = read_pdf.getPage(page_number)
            page_content = page.extractText()
            text_file.write(page_content)
    input_file_loop(str(file_name_txt))


    print("Getting MCQs!")
    keyword_extraction_loop()
    sentence_mapping_loop()
    final_mcq_gen_loop()
    global res_dict
    # print(res_dict)

if __name__ == "__main__":
    app.run('0.0.0.0', port=8001,debug=True)
