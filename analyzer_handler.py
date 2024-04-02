import csv
import time

import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertModel
from scipy.cluster.hierarchy import dendrogram, linkage

from textblob import TextBlob
from gensim import corpora, models
from textstat import flesch_reading_ease, gunning_fog


import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc


import pandas as pd
import numpy as np
#python3 -m spacy download en_core_web_sm

import pprint
pprint = pprint.pprint

import csv
from collections import Counter


chatgpt_humanizer_blog = 'path to txt file that contains your hand written blog'
with open(chatgpt_humanizer_blog, 'r') as f:
    reader = f.read()

form_fill_blog = 'path to txt file that contains your AI generated blog '
with open(form_fill_blog, 'r') as f:
    reader2 = f.read()


class NLP:
    def __init__(self):
        self.chatgpt_text = reader
        self.form_text = reader2
        self.model = spacy.load("en_core_web_lg")
        # self.ruler = self.model.add_pipe('entity_ruler')
        self.tokenizer = Tokenizer
        self.doc = self.model(self.chatgpt_text)
        self.doc2 = self.model(self.form_text)

    def add_pipe_ruler(self):
        ruler = self.model.add_pipe('entity_ruler')

        self.patterns = [{'label': ent.label_, 'pattern': ent.text} for ent in self.doc.ents]
        ruler.add_patterns(self.patterns)

        self.patterns2 = [{'label': ent.label_, 'pattern': ent.text} for ent in self.doc2.ents]
        ruler.add_patterns(self.patterns2)

        # with open('chatgpt_ents.txt', 'a') as writer:
        #     for ent in self.patterns:
        #         # writer.write(f"{ent['pattern']} ({ent['label']})\n")
        #         print(ent)

        # with open('form_ents.txt', 'a') as writer2:
            # for ent in self.patterns2:
                # writer2.write(f"{ent['pattern']} ({ent['label']})\n")


    def compare_patterns_similarity(self):
        self.add_pipe_ruler()
        patterns1_set = {pattern['pattern'] for pattern in self.patterns}
        patterns2_set = {pattern['pattern'] for pattern in self.patterns2}

        intersection = len(patterns1_set.intersection(patterns2_set))
        union = len(patterns1_set.union(patterns2_set))
        print("union:", union)
        print("intersection:", intersection)

        jaccard_similarity = intersection / union if union != 0 else 0
        return jaccard_similarity



    def compare_pattern_similiarity_scores(self):
        print("Similarity between patterns:", self.compare_patterns_similarity())
        print("similiarity score:", self.doc.similarity(self.doc2))


    # def combine_1(self):
    #     with open('chatgpt_ents.txt', 'r') as f1:
    #         chatgpt_lines = f1.readlines()
    #
    #     with open('form_ents.txt', 'r') as f2:
    #         form_lines = f2.readlines()
    #
    #     with open('combined.csv', 'w', newline='') as csvfile:
    #         fieldnames = ['chatgpt', 'form']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #         writer.writeheader()
    #         for chatgpt_line, form_line in zip(chatgpt_lines, form_lines):
    #             # writer.writerow({'chatgpt': chatgpt_line.strip().split(), 'form': form_line.strip().split()})
    #             writer.writerow({'chatgpt': chatgpt_line.strip(), 'form': form_line.strip()})
    #
    #     f1.close()
    #     f2.close()
    #     csvfile.close()


    def write_form_tokens_to_csv(self):
        with open('form_text.csv', 'w', newline='') as csvfile:
            fieldnames = ['text', "lemma", "pos", "tag", "dep", "shape", "alpha", "is_stop_word", "lex_id", "vector_norm", "has_vector"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for token in self.doc2:
                writer.writerow({"text": token.text, "lemma": token.lemma_, "pos": token.pos_, "tag": token.tag_, "dep": token.dep_,
                  "shape": token.shape_, "alpha": token.is_alpha, "is_stop_word": token.is_stop, "lex_id": token.lex_id,
                                "vector_norm": token.vector_norm, "has_vector": token.has_vector})

        csvfile.close()


    def write_chatgpt_tokens_to_csv(self):
        with open('chatgpt_text.csv', 'w', newline='') as csvfile:
            fieldnames = ['text', "lemma", "pos", "tag", "dep", "shape", "alpha", "is_stop_word", "lex_id", "vector_norm", "has_vector"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for token in self.doc:
                writer.writerow({"text": token.text, "lemma": token.lemma_, "pos": token.pos_, "tag": token.tag_, "dep": token.dep_,
                  "shape": token.shape_, "alpha": token.is_alpha, "is_stop_word": token.is_stop, "lex_id": token.lex_id,
                                "vector_norm": token.vector_norm, "has_vector": token.has_vector})

        csvfile.close()



    # def compare_tokens_1(self):
    #     for sent in self.doc.sents:
    #         print(len(sent))


    def sentiment_analysis(self):
        chatgpt_blob = TextBlob(self.chatgpt_text)
        form_blob = TextBlob(self.form_text)

        chatgpt_sentiment = chatgpt_blob.sentiment
        form_sentiment = form_blob.sentiment

        print("Sentiment Analysis:")
        print("Hand-written blog - Polarity: {:.2f}, Subjectivity: {:.2f}".format(chatgpt_sentiment.polarity, chatgpt_sentiment.subjectivity))
        print("AI-generated blog - Polarity: {:.2f}, Subjectivity: {:.2f}".format(form_sentiment.polarity, form_sentiment.subjectivity))

    def compare_metrics(self):
        chatgpt_tokens = [token for token in self.doc if not token.is_punct]
        form_tokens = [token for token in self.doc2 if not token.is_punct]

        chatgpt_word_count = len(chatgpt_tokens)
        form_word_count = len(form_tokens)

        chatgpt_unique_words = len(set(token.text.lower() for token in chatgpt_tokens))
        form_unique_words = len(set(token.text.lower() for token in form_tokens))

        print("\nMetrics Comparison:")
        print("Word Count - Hand-written blog: {}, AI-generated blog: {}".format(chatgpt_word_count, form_word_count))
        print("Unique Words - Hand-written blog: {}, AI-generated blog: {}".format(chatgpt_unique_words, form_unique_words))

  
    def compare_labels(self):
        chatgpt_labels = [ent.label_ for ent in self.doc.ents]
        form_labels = [ent.label_ for ent in self.doc2.ents]

        chatgpt_label_counts = Counter(chatgpt_labels)
        form_label_counts = Counter(form_labels)

        print("\nLabel Comparison:")
        print("Hand-written blog - Label Counts:")
        for label, count in chatgpt_label_counts.items():
            print("{}: {}".format(label, count))

        print("AI-generated blog - Label Counts:")
        for label, count in form_label_counts.items():
            print("{}: {}".format(label, count))

  
    def compare_patterns(self):
        chatgpt_patterns = [ent.text for ent in self.doc.ents]
        form_patterns = [ent.text for ent in self.doc2.ents]

        common_patterns = set(chatgpt_patterns) & set(form_patterns)
        unique_chatgpt_patterns = set(chatgpt_patterns) - set(form_patterns)
        unique_form_patterns = set(form_patterns) - set(chatgpt_patterns)

        print("\nPattern Comparison:")
        print("Common Patterns:")
        for pattern in common_patterns:
            print(pattern)

        print("Unique Patterns in Hand-written blog:")
        for pattern in unique_chatgpt_patterns:
            print(pattern)

        print("Unique Patterns in AI-generated blog:")
        for pattern in unique_form_patterns:
            print(pattern)

  
    def analyze_sentence_structure(self):
        chatgpt_sents = list(self.doc.sents)
        form_sents = list(self.doc2.sents)

        chatgpt_sent_lengths = [len(sent) for sent in chatgpt_sents]
        form_sent_lengths = [len(sent) for sent in form_sents]

        chatgpt_avg_sent_length = sum(chatgpt_sent_lengths) / len(chatgpt_sent_lengths)
        form_avg_sent_length = sum(form_sent_lengths) / len(form_sent_lengths)

        print("\nSentence Structure Analysis:")
        print("Average Sentence Length - Hand-written blog: {:.2f}, AI-generated blog: {:.2f}".format(
            chatgpt_avg_sent_length, form_avg_sent_length))

        chatgpt_pos_counts = Doc(self.doc.vocab, words=[t.text for t in self.doc]).count_by(spacy.attrs.POS)
        form_pos_counts = Doc(self.doc2.vocab, words=[t.text for t in self.doc2]).count_by(spacy.attrs.POS)

        print("POS Tag Distribution - Hand-written blog:")
        for pos, count in chatgpt_pos_counts.items():
            print("{}: {:.2f}%".format(self.doc.vocab[pos].text, count / len(self.doc) * 100))

        print("POS Tag Distribution - AI-generated blog:")
        for pos, count in form_pos_counts.items():
            print("{}: {:.2f}%".format(self.doc2.vocab[pos].text, count / len(self.doc2) * 100))

  
    def analyze_readability(self):
        chatgpt_flesch = flesch_reading_ease(self.chatgpt_text)
        form_flesch = flesch_reading_ease(self.form_text)

        chatgpt_gunning_fog = gunning_fog(self.chatgpt_text)
        form_gunning_fog = gunning_fog(self.form_text)

        print("\nReadability Analysis:")
        print("Flesch Reading Ease - Hand-written blog: {:.2f}, AI-generated blog: {:.2f}".format(
            chatgpt_flesch, form_flesch))
        print("Gunning Fog Index - Hand-written blog: {:.2f}, AI-generated blog: {:.2f}".format(
            chatgpt_gunning_fog, form_gunning_fog))

  
    def analyze_named_entities(self):
        chatgpt_ents = [(ent.text, ent.label_) for ent in self.doc.ents]
        form_ents = [(ent.text, ent.label_) for ent in self.doc2.ents]

        chatgpt_ent_counts = Counter(ent[1] for ent in chatgpt_ents)
        form_ent_counts = Counter(ent[1] for ent in form_ents)

        print("\nNamed Entity Analysis:")
        print("Named Entity Distribution - Hand-written blog:")
        for ent_type, count in chatgpt_ent_counts.items():
            print("{}: {}".format(ent_type, count))

        print("Named Entity Distribution - AI-generated blog:")
        for ent_type, count in form_ent_counts.items():
            print("{}: {}".format(ent_type, count))

  
    def extract_keywords(self):
        vectorizer = TfidfVectorizer(stop_words='english')

        chatgpt_text = ' '.join([token.text for token in self.doc])
        form_text = ' '.join([token.text for token in self.doc2])

        tfidf_matrix = vectorizer.fit_transform([chatgpt_text, form_text])
        feature_names = vectorizer.get_feature_names_out()

        chatgpt_keywords = self.get_top_keywords(tfidf_matrix[0].toarray()[0], feature_names)
        form_keywords = self.get_top_keywords(tfidf_matrix[1].toarray()[0], feature_names)

        print("\nKeyword Extraction:")
        print("Top keywords - Hand-written blog:")
        print(chatgpt_keywords)
        print("Top keywords - AI-generated blog:")
        print(form_keywords)

  
    def get_top_keywords(self, tfidf_row, feature_names, top_n=10):
        top_indices = tfidf_row.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        return top_keywords

  
    def analyze_semantic_coherence(self):
        chatgpt_sentences = [sent.text.lower() for sent in self.doc.sents]
        form_sentences = [sent.text.lower() for sent in self.doc2.sents]

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chatgpt_sentences + form_sentences)

        # Perform LSA
        lsa = TruncatedSVD(n_components=10)
        lsa_matrix = lsa.fit_transform(tfidf_matrix)

        chatgpt_similarity = cosine_similarity(lsa_matrix[:len(chatgpt_sentences)])
        form_similarity = cosine_similarity(lsa_matrix[len(chatgpt_sentences):])

        chatgpt_coherence = np.mean(chatgpt_similarity)
        form_coherence = np.mean(form_similarity)

        print("\nSemantic Coherence Analysis:")
        print("Average Semantic Coherence - Hand-written blog: {:.2f}".format(chatgpt_coherence))
        print("Average Semantic Coherence - AI-generated blog: {:.2f}".format(form_coherence))

    def analyze_contextual_embeddings(self):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        chatgpt_encoded = tokenizer(self.chatgpt_text, padding=True, truncation=True, return_tensors='pt')
        form_encoded = tokenizer(self.form_text, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            chatgpt_embeddings = model(**chatgpt_encoded)[0]
            form_embeddings = model(**form_encoded)[0]

        # Compute average embedding for each blog
        chatgpt_avg_embedding = torch.mean(chatgpt_embeddings, dim=1)
        form_avg_embedding = torch.mean(form_embeddings, dim=1)

        # Compute cosine similarity between average embeddings
        similarity = cosine_similarity(chatgpt_avg_embedding.numpy(), form_avg_embedding.numpy())

        print("\nContextual Embeddings Analysis:")
        print("Cosine Similarity - Hand-written vs AI-generated: {:.2f}".format(similarity[0][0]))

  
    def analyze_sentiment_flow(self):
        # Perform sentiment analysis at the sentence level
        chatgpt_sentences = [sent.text for sent in self.doc.sents]
        form_sentences = [sent.text for sent in self.doc2.sents]

        chatgpt_sentiments = [TextBlob(sent).sentiment.polarity for sent in chatgpt_sentences]
        form_sentiments = [TextBlob(sent).sentiment.polarity for sent in form_sentences]

        # Compute sentiment flow metrics
        chatgpt_sentiment_changes = np.sum(np.abs(np.diff(chatgpt_sentiments)))
        form_sentiment_changes = np.sum(np.abs(np.diff(form_sentiments)))

        print("\nSentiment Flow Analysis:")
        print("Sentiment Changes - Hand-written blog: {:.2f}".format(chatgpt_sentiment_changes))
        print("Sentiment Changes - AI-generated blog: {:.2f}".format(form_sentiment_changes))

  
    def analyze_topical_diversity(self):
        # Preprocess the text data
        chatgpt_sentences = [sent.text.lower() for sent in self.doc.sents]
        form_sentences = [sent.text.lower() for sent in self.doc2.sents]

        # Create a dictionary and corpus for each blog
        chatgpt_dictionary = corpora.Dictionary([doc.split() for doc in chatgpt_sentences])
        form_dictionary = corpora.Dictionary([doc.split() for doc in form_sentences])

        chatgpt_corpus = [chatgpt_dictionary.doc2bow(text.split()) for text in chatgpt_sentences]
        form_corpus = [form_dictionary.doc2bow(text.split()) for text in form_sentences]

        # Perform LDA topic modeling
        chatgpt_lda = models.LdaMulticore(chatgpt_corpus, num_topics=5, id2word=chatgpt_dictionary, passes=10)
        form_lda = models.LdaMulticore(form_corpus, num_topics=5, id2word=form_dictionary, passes=10)

        # Compute topic diversity scores
        chatgpt_topic_diversity = len(chatgpt_lda.show_topics())
        form_topic_diversity = len(form_lda.show_topics())

        print("\nTopical Diversity Analysis:")
        print("Topic Diversity - Hand-written blog: {}".format(chatgpt_topic_diversity))
        print("Topic Diversity - AI-generated blog: {}".format(form_topic_diversity))


def main():
    nlp = NLP()
    # nlp.add_pipe_ruler()

    # nlp.compare_tokens_1()

    # nlp.sentiment_analysis()
    # nlp.compare_metrics()
    # nlp.compare_labels()
    # nlp.compare_patterns()

    # nlp.analyze_sentence_structure()
    # nlp.analyze_readability()

    # nlp.analyze_named_entities()
    # nlp.extract_keywords()

    nlp.analyze_semantic_coherence()
    # nlp.analyze_discourse_structure()
    nlp.analyze_contextual_embeddings()
    nlp.analyze_sentiment_flow()
    nlp.analyze_topical_diversity()



if __name__ == '__main__':
    main()
