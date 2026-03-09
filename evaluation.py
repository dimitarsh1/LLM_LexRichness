#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

all_translations = {}
refs = {}
src = {}

for model_name in ["mbart", "j2", "gpt", "deepseek"]:
    for translation_name in ["news", "poetry", "literature"]:
        file_prefix = model_name+"_"+translation_name
        print("Reading file: " + file_prefix, end=". ")
        data = pd.read_csv(os.path.join("new_data", file_prefix +".out.csv"), sep='\t', header=0)
        _tmp_lines = data['translation'].tolist()
        all_translations[file_prefix] = _tmp_lines
        print(str(len(_tmp_lines)) + " read.")

for translation_name in ["news", "poetry", "literature"]:
    file_prefix = translation_name + "-1000.csv"
    print("Reading file: " + file_prefix, end=". ")
    if translation_name in ["literature"]:
        true = pd.read_csv(file_prefix, sep=';', header=0, encoding='utf8', engine='python')
    else:
        true = pd.read_csv(file_prefix, sep=';', header=0, encoding='CP1252', engine='python')

    refs[translation_name] = true['NL'].tolist()
    src[translation_name] = true['EN'].tolist()
    print(str(len(refs[translation_name])) + " read.")
    print(str(len(src[translation_name])) + " read.")


# In[3]:


from sacrebleu.metrics import BLEU, CHRF, TER

for translation_name in ["news", "poetry", "literature"]:
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name

        print(exp_prefix)
        print(BLEU().corpus_score(all_translations[exp_prefix], [refs[translation_name]], n_bootstrap = 1000), 
                CHRF().corpus_score(all_translations[exp_prefix], [refs[translation_name]], n_bootstrap = 1000),
                TER().corpus_score(all_translations[exp_prefix], [refs[translation_name]], n_bootstrap = 1000)
             )


# In[8]:


from sacrebleu.metrics import BLEU, CHRF, TER
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

def get_bleu(sys, ref):
    ''' Computing BLEU using sacrebleu

        :param sysname: the name of the system
        :param sys: the sampled sentences from the translation (type = list)
        :param ref: the reference sentences (type = list)
        :returns: a socre (float)
    '''
    bleu = BLEU().corpus_score(sys, [ref])
    return bleu.score

def get_chrf(sys, ref):
    ''' Computing CHRF using sacrebleu

        :param sysname: the name of the system
        :param sys: the sampled sentences from the translation (type = list)
        :param ref: the reference sentences (type = list)
        :returns: a socre (float)
    '''
    chrf = CHRF().corpus_score(sys, [ref])
    return chrf.score

def get_ter(sys, ref):
    ''' Computing BLEU using sacrebleu

        :param sysname: the name of the system
        :param sys: the sampled sentences from the translation (type = list)
        :param ref: the reference sentences (type = list)
        :returns: a socre (float)
    '''
    ter = TER().corpus_score(sys, [ref])
    return ter.score

def compute_metric(metric_func, sentences, ref, sample_idxs, iters):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param ref: the reference sentences
        :param lang: the langauge for detokenization
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    # 5. let's get the measurements for each sample
    scores = {}
    scores = Parallel(n_jobs=8)(delayed(eval(metric_func))([sentences[j] for j in sample_idxs[i]], [ref[j] for j in sample_idxs[i]]) for i in range(iters))
             
    return scores
    
    
def compute_significance(metrics, iterations):
    ''' Compute pairwose significance interval

        :param metrics: dictionary with systems and metrics
        :param iterations: the number of iterations
        :returns: a socre (float)
    '''
    # now, we are able to compute statistical significance
    # print('delta(xi) > delta(x):')
    scores = {}
    for system1 in metrics:
        scores[system1] = {}
        for system2 in metrics:
            s = 0.0
            for i in range(iterations):
                if round(metrics[system1][i], 4) > round(metrics[system2][i], 4):
                    s += 1.0
            if system1 == system2:
                scores[system1][system2] = -1.0
            else:
                scores[system1][system2] = s / iterations
    return scores
    

def compute_ttest_scikit(metrics, iterations):
    ''' Compute pairwose significance interval

        :param metrics: dictionary with systems and metrics
        :param iterations: the number of iterations
        :returns: a socre (float)
    '''
    scores = {}
    print('\nScikit ttest:')
    for system1 in metrics:
        scores[system1] = {}
        for system2 in metrics:
            t, p =  ttest_ind(metrics[system1], metrics[system2])
            scores[system1][system2] = p
        
    return scores


def print_latex_table(scores, metric_title):
    ''' Prints a table in latex format; ready to incorporate into a tex file

        :param scores: dictionary with scores and systems
        :param metric_title: identifying the metric (string)
    '''
    print(' '.join([str(s) for s in scores.keys()]))
    for system in scores:
        print(system + ' ' + ' '.join([str(p) for p in scores[system].values()]))

    print('\n')
    print('  & ' + ' & '.join([str(s) for s in scores[system].keys()]) + '\\\\\hline')
    for system in scores:
        print_en = False
        print(system, end='')
        for system2 in scores:
            p = scores[system][system2]
            if (p >= 0.0 and p < 0.05) or p >= 0.95:
                if print_en:
                    print(' & Y', end='')
                else:
                    print(' & ', end='')
            else:
                if print_en:
                    print(' & N', end='')
                else:
                    print(' & ', end='')

            if system == system2:
                print_en = True
        print('\\\\\hline')


# 3. read the other variables.
import numpy as np

iters = int(1000)
sample_size = int(300)

# 4. Compute Sample metric
metrics = {}

metrics['bleu'] = {}
metrics['chrf'] = {}
metrics['ter'] = {}
    
for translation_name in ["news", "poetry", "literature"]:
    for metric in metrics:
        for model_name in ["mbart", "j2", "gpt", "deepseek"]:
            exp_prefix = model_name+"_"+translation_name
            #print(exp_prefix)
            sample_idxs = np.random.randint(0, len(refs[translation_name]), size=(iters, sample_size))
            metrics[metric][model_name] = compute_metric('get_'+metric, all_translations[exp_prefix], refs[translation_name], sample_idxs, iters)
            
        print("-------------------------------------------------")
        print(translation_name + " " + metric)
        sign_scores = compute_significance(metrics[metric], iters)
        print_latex_table(sign_scores, metric)
        sign_scores = compute_ttest_scikit(metrics[metric], iters)
        print_latex_table(sign_scores, metric)


# In[16]:


get_ipython().system('pip install unbabel-comet')


# In[22]:


from comet import download_model

model_names = ["wmt22-comet-da"]
model_paths = {}
for model_name in model_names:
    
    model_path = download_model("/".join(["Unbabel", model_name]))
    model_paths[model_name] = model_path


# In[25]:


from comet import load_from_checkpoint

comet_scores = {}
model_names = ["wmt22-comet-da"]
for model_name in model_names:
    model = load_from_checkpoint(model_paths[model_name])

    data = []
    for translation_name in ["news", "poetry", "literature"]:
        for trans_model_name in ["mbart", "j2", "gpt", "deepseek"]:
            exp_prefix = trans_model_name+"_"+translation_name
            data = [{"src": src, "mt": mt, "ref": ref} for (src, mt, ref) in zip(src[translation_name], all_translations[exp_prefix], refs[translation_name])]
            comet_scores[model_name + exp_prefix] = model.predict(data, batch_size=8)
            


# In[28]:


model_names = ["wmt22-comet-da"]
for model_name in model_names:
    for translation_name in ["news", "poetry", "literature"]:
        for trans_model_name in ["mbart", "j2", "gpt", "deepseek"]:
            exp_prefix = trans_model_name+"_"+translation_name
            print(model_name + " " + exp_prefix + ": " + str(comet_scores[model_name + exp_prefix]['system_score']))


# In[30]:


get_ipython().system('pip install --upgrade pip  # ensures that pip is current')
get_ipython().system('git clone https://github.com/google-research/bleurt.git')
get_ipython().system('cd bleurt && pip install .')

!cd bleurt && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip .
!cd bleurt && unzip BLEURT-20-D3.zip
# In[3]:


from bleurt import score

checkpoint = "bleurt/BLEURT-20-D3"
scorer = score.BleurtScorer(checkpoint)

bluert_scores = {}
for translation_name in ["news", "poetry", "literature"]:
        for trans_model_name in ["mbart", "j2", "gpt", "deepseek"]:
            exp_prefix = trans_model_name+"_"+translation_name
            candidates = all_translations[exp_prefix]
            references = refs[translation_name]
            bluert_scores[checkpoint + exp_prefix] = scorer.score(references = references, candidates = candidates)


# In[2]:


from bleurt import score

checkpoint = "bleurt/BLEURT-20-D6"
scorer = score.BleurtScorer(checkpoint)

bluert_scores = {}
for translation_name in ["news", "poetry", "literature"]:
        for trans_model_name in ["mbart", "j2", "gpt", "deepseek"]:
            exp_prefix = trans_model_name+"_"+translation_name
            candidates = all_translations[exp_prefix]
            references = refs[translation_name]
            bluert_scores[checkpoint + exp_prefix] = scorer.score(references = references, candidates = candidates)


# In[1]:


for score in bluert_scores:
    print(bluert_scores[score])


# In[11]:


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

for score in bluert_scores:
    print(score, end=": ")
    print (mean(bluert_scores[score]))


# In[16]:


get_ipython().system('pip install lexical_diversity')
get_ipython().system('pip install lexicalrichness')
get_ipython().system('pip install spacy_udpipe')


# In[17]:


import itertools
from lexical_diversity import lex_div as ld
from lexicalrichness import LexicalRichness as lr
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import statistics
import spacy_udpipe
import time
import pickle
import os
from nltk.probability import FreqDist
import logging

def plot_freqdist_freq(fd,
                       max_num=None,
                       cumulative=False,
                       title='Frequency plot',
                       linewidth=2):
    """
    As of NLTK version 3.2.1, FreqDist.plot() plots the counts
    and has no kwarg for normalising to frequency.
    Work this around here.

    INPUT:
        - the FreqDist object
        - max_num: if specified, only plot up to this number of items
          (they are already sorted descending by the FreqDist)
        - cumulative: bool (defaults to False)
        - title: the title to give the plot
        - linewidth: the width of line to use (defaults to 2)
    OUTPUT: plot the freq and return None.
    """

    tmp = fd.copy()
    norm = fd.N()
    for key in tmp.keys():
        tmp[key] = float(fd[key]) / norm

    if max_num:
        tmp.plot(max_num, cumulative=cumulative,
                 title=title, linewidth=linewidth)
    else:
        tmp.plot(cumulative=cumulative,
                 title=title,
                 linewidth=linewidth)

    return

def get_lemmas(sentences, nlpD, system_name, freq_voc = None, cache = True):
    ''' Computes the lemmas and their frequencies for the given sentences

        :param sentences: a list of sentences
        :param nlpd: the data model for the lematizer
        :param freq_voc: a frequency vocabulary
        :param cache: whether to create new file or not (True).
        :returns: a dictionary of lemmas and frequencies
    '''
    a = time.time()

    lemmas = {}

    if os.path.exists(system_name + ".spacy_udpipe.lemmas") and cache:
        logging.debug("Lemmas dict loading from file")
        with open(system_name + ".spacy_udpipe.lemmas", "rb") as SpUpM:
            lemmas = pickle.load(SpUpM)
        logging.debug("Lemmas dict loaded")
    else:
        logging.debug("Lemmas dict building from scratch")
        #nlps = list(nlpD.pipe(sentences, n_process=-1))
        nlps = list(nlpD.pipe(sentences))

        for doc in nlps:
            for token in doc:
                lemma=token.lemma_
                tokenLow=str(token).lower()

                if lemma in lemmas: # existing lemma
                    if tokenLow not in lemmas[lemma]:
                        lemmas[lemma][tokenLow]=1
                    else:
                        lemmas[lemma][tokenLow]+=1
                else:                       # unexisting lemma
                    lemmas[lemma]={}        # if this is the first time we have a lemma then there are no tokens
                    lemmas[lemma][tokenLow]=1

        with open(system_name + ".spacy_udpipe.lemmas", "wb") as PoF:
            pickle.dump(lemmas, PoF)

        logging.debug("Lemmas dict built and saved")

    #print("Length of all lemmas: " + str(len(lemmas)))
    all_lemas_len = len(lemmas)
    singleton_lemmas = [lemma + "\t" + str(len(lemmas[lemma])) for lemma in lemmas if len(lemmas[lemma]) < 2]
    #print("Length of singleton lemmas: " + str(len(singleton_lemmas)))
    all_sing_lemmas_len = len(singleton_lemmas)
    singleton_matching_lemmas = []

    with open(system_name + ".lemmas", "w") as oF:
        oF.write("\n".join([lemma + ": " + "\t".join(str(f) + "|" + str(g) for (f,g) in zip(lemmas[lemma].keys(), lemmas[lemma].values())) for lemma in lemmas]))

    if freq_voc is not None:
        tmp_lemmas = {}
        for lemma in lemmas:
            if len(lemmas[lemma]) > 1:
                for form in lemmas[lemma]:
                    if form in freq_voc:
                        tmp_lemmas[lemma] = lemmas[lemma]
                        break           # we only need one occurance to match
            else:
                singleton_matching_lemmas.append(lemma)
        lemmas = tmp_lemmas

    # print("Length of matched lemmas: " + str(len(lemmas)))
    # print("Length of singleton maching lemmas: " + str(len(singleton_matching_lemmas)))

    return (lemmas, singleton_lemmas)

def simpson_diversity(wordFormDict):
    ''' Computes the Simpson Diversity Index

        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number)
    '''

    def p(n, N):
        ''' Relative abundance
        '''
        if n ==  0:
            return 0
        else:
            return float(n)/N

    N = sum(wordFormDict.values())
    return sum(p(n, N)**2 for n in wordFormDict.values() if n != 0)

def inverse_simpson_diversity(wordFormDict):
    ''' Computes the inverse Simpson Diversity Index
    
        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number) 
    '''
    return float(1)/simpson_diversity(wordFormDict)

"""# Shannon Diversity #
The Shannon-Weiner diversity represent the proportion of species abundance in the population. Its being at maximum when all species occur in similar number of individuals and the lowest when the sample contain one species. From my experience there is no limit to compare the diversity value with as for evenness, which resricted between 0-1. For Example, if the sample contain 4 species each represented by 5o individuals the, diversity H equal 1.3863, and if the sample contain 5 species (one more) and each represented by similar number of individuals (50), the diversity equal 1.6094.
"""

def shannon_diversity(wordFormDict):
    '''
    
        :param wordFormDict: a dictionary { 'species': count }
        :returns: Shannon Diversity Index
    '''
    #>>> sdi({'a': 10, 'b': 20, 'c': 30,})
    #1.0114042647073518
    
    from math import log as ln
    
    def p(n, N):
        """ Relative abundance """
        if n ==  0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)
            
    N = sum(wordFormDict.values())
    
    return -sum(p(n, N) for n in wordFormDict.values() if n != 0)

def compute_simpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a nested dictionary
        :returns: a dictionary with the simpson diversity for every lemma 
    '''
    simpsonDict = {}
    for l in nestedDict:
        simpsonDict[l]=simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values())

def compute_invSimpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
    
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the inversed simpson diversity
    '''
    simpsonDict={}
    for l in nestedDict:
        simpsonDict[l]=inverse_simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values()) 

def compute_shannonDiv(nestedDict):
    ''' Computes the shannon diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the shannon diversity
    '''
    shannonDict={}
    for lem in nestedDict:
        shannonDict[lem]=shannon_diversity(nestedDict[lem])
    return statistics.mean(shannonDict.values())

def compute_yules_i(sentences):
    ''' Computing Yules I measure

        :param sentences: dictionary with all words and their frequencies
        :returns: Yules I (the inverse of yule's K measure) (float) - the higher the better
    '''
    _total, vocabulary = get_vocabulary(sentences)
    M1 = float(len(vocabulary))
    M2 = sum([len(list(g))*(freq**2) for freq,g in itertools.groupby(sorted(vocabulary.values()))])

    try:
        return (M1*M1)/(M2-M1)
    except ZeroDivisionError:
        return 0

def compute_ttr(sentences):
    ''' Computes the type token ratio
    
        :param sentences: the sentences
        :returns: The type token ratio (float)
    '''      

    total, vocabulary = get_vocabulary(sentences)    
    return len(vocabulary)/total
    
def compute_mtld(sentences):
    ''' Computes the MTLD
    
        :param sentences: sentences
    
        :returns: The MTLD (float)
    '''      
    
    def my_mtld(lex, threshold, reverse=False):
        """
        Parameters
        ----------
        threshold: float
            Factor threshold for MTLD. Algorithm skips to a new segment when TTR goes below the
            threshold (default=0.72).
        reverse: bool
            If True, compute mtld for the reversed sequence of text (default=False).
        Returns:
            mtld measure (float)
        """
        if reverse:
            word_iterator = iter(reversed(lex.wordlist))
        else:
            word_iterator = iter(lex.wordlist)

        terms = set()
        word_counter = 0
        factor_count = 0

        for word in word_iterator:
            word_counter += 1
            terms.add(word)
            ttr = len(terms)/word_counter

            if ttr <= threshold:
                word_counter = 0
                terms = set()
                factor_count += 1

        # partial factors for the last segment computed as the ratio of how far away ttr is from
        # unit, to how far away threshold is to unit
        if word_counter > 0:
            factor_count += (1-ttr) / (1 - threshold)

        # ttr never drops below threshold by end of text
        if factor_count == 0:
            ttr = lex.terms / lex.words
            if ttr == 1:
                factor_count += 1
            else:
                factor_count += (1-ttr) / (1 - threshold)

        return len(lex.wordlist) / factor_count

    ll = '\n'.join(sentences)
    lex = lr(ll)
    return lex.mtld()
#    return ld.mtld(ll)
    
def get_vocabulary(sentence_array):
    ''' Compute vocabulary

        :param sentence_array: a list of sentences
        :returns: a list of tokens
    '''
    data_vocabulary = {}
    total = 0
    
    for sentence in sentence_array:
        for token in sentence.strip().split():
            if token not in data_vocabulary:
                data_vocabulary[token] = 1 #/len(line.strip().split())
            else:
                data_vocabulary[token] += 1 #/len(line.strip().split())
            total += 1
            
    return total, data_vocabulary

def get_vocabulary_lowercase(sentence_array):
    ''' Compute vocabulary but converts everything to lowercase first

        :param sentence_array: a list of sentences
        :returns: a list of tokens
    '''
    data_vocabulary = {}
    total = 0
    
    for sentence in sentence_array:
        for token in sentence.lower().strip().split():
            if token.lower() not in data_vocabulary:
                data_vocabulary[token.lower()] = 1 #/len(line.strip().split())
            else:
                data_vocabulary[token.lower()] += 1 #/len(line.strip().split())
            total += 1
            
    return total, data_vocabulary

def compute_ld_metric(metric_func, sentences, sample_idxs, iters):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    # 5. let's get the measurements for each sample
    scores = Parallel(n_jobs=-1)(delayed(eval(metric_func))([sentences[j] for j in sample_idxs[i]]) for i in range(iters))

    return scores

def compute_gram_diversity(sentences, lang="en", system_name="", freq_voc=None, cache=True):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    nlpD = spacy_udpipe.load(lang).tokenizer
    nlpD.max_length = 300000000

    (lemmas, singleton_lemmas) = get_lemmas(sentences, nlpD, system_name, freq_voc, cache)

    return (len(lemmas), len(singleton_lemmas), compute_simpDiv(lemmas), compute_invSimpDiv(lemmas), compute_shannonDiv(lemmas))

def textToLFP(sentences, step=1000, last=2000):
    '''we are not lowercasing, tokenizing, removing stopwords, numerals etc.
    this is because we are looking into algorithmic bias and as such into the effect of the algorithm
    on the text it is offered. The text is already tokenized. Might add Lowercasing too.'''

    #create Frequency Dictionary
    fdist = FreqDist(" ".join(sentences).split()) # our text is already tokenized. We merge all sentences together
                                                  # and create one huge list of tokens.

    # get size range
    end = last + step
    sizes = list(range(0, end, step))

    #Get words for every frequency band
    freqs = [fdist.most_common(size+step)[size:size+step] for size in sizes[:-1]]
    freqs.append(fdist.most_common()[last:])

    #total tokens
    totalCount=fdist.N()

    #percentage frequency band
    percs = [sum([count for (_word,count) in freq])/totalCount for freq in freqs]

    #plot
    #plot_freqdist_freq(fdist, 20)

    return percs


# In[7]:


get_ipython().system('pip install sacremoses')


# In[10]:


from sacremoses import MosesTokenizer
mtok = MosesTokenizer(lang='nl')

all_tok_translations = {}
refs_tok = {}

for translation_name in ["news", "poetry", "literature"]:
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        all_tok_translations[exp_prefix] = [" ".join(mtok.tokenize(_tok_sent, escape=False)) for _tok_sent in all_translations[exp_prefix]]

    refs_tok[translation_name] = [" ".join(mtok.tokenize(_tok_sent, escape=False)) for _tok_sent in refs[translation_name]]


# In[28]:


# ----------------------------------
# Yules I, TTR, MTLD (non tokenized)
# ----------------------------------
print("Domain & model & Yule's I & TTR & MTLD\\\\\hline")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref.", end=" & ")
    lex_scores = [compute_yules_i(refs[translation_name]),
              compute_ttr(refs[translation_name]),
              compute_mtld(refs[translation_name])]
    print(" & ".join([str(round(score, 4)) for score in lex_scores]) + "\\\\\\hline")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        lex_scores = [compute_yules_i(all_translations[exp_prefix]),
              compute_ttr(all_translations[exp_prefix]),
              compute_mtld(all_translations[exp_prefix])]
        print(" & ".join([str(round(score, 4)) for score in lex_scores]) + "\\\\\\hline")
    


# In[29]:


# ------------------------------
# Yules I, TTR, MTLD (TOKENIZED)
# ------------------------------
print("Domain & model & Yule's I & TTR & MTLD\\\\\hline")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref", end=" & ")
    lex_scores = [compute_yules_i(refs_tok[translation_name]),
              compute_ttr(refs_tok[translation_name]),
              compute_mtld(refs_tok[translation_name])]
    print(" & ".join([str(round(score, 4)) for score in lex_scores]) + "\\\\\\hline")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        lex_scores = [compute_yules_i(all_tok_translations[exp_prefix]),
              compute_ttr(all_tok_translations[exp_prefix]),
              compute_mtld(all_tok_translations[exp_prefix])]
        print(" & ".join([str(round(score, 4)) for score in lex_scores]) + "\\\\\\hline")

    
    


# In[30]:


# -------------------------
# LEXICAL FREQUENCY PROFILE
# -------------------------
print("Lexical Frequency Profile")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end=" & ")
    print("Refs.", end = " & ")
    print(" & ".join([str(round(band, 4)) for band in textToLFP(refs_tok[translation_name])]), end="\\\\\hline\n")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        print(" & ".join([str(round(band, 4)) for band in textToLFP(all_tok_translations[exp_prefix])]), end="\\\\\hline\n")



# In[22]:


# Downloading the nl model for spacy udpipe
spacy_udpipe.download("nl")
spacy_udpipe.download("en")


# In[32]:


# ----------------------------------------------
# GRAM DIVERSITY (Shannon, Inv Simpson, Simpson)
# ----------------------------------------------
print("Domain & model & Num. all lemmas & Num. signle lemmas & Shannon & Inv. Simpson & Simpson\\\\\hline")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref.", end=" & ")
    gram_div_scores = compute_gram_diversity(refs_tok[translation_name], "nl", translation_name, None, False)
    print(" & ".join([str(round(score, 4)) for score in gram_div_scores ]) + "\\\\hline")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        gram_div_scores = compute_gram_diversity(all_tok_translations[exp_prefix], "nl", exp_prefix, None, False)
        print(" & ".join([str(round(score, 4)) for score in gram_div_scores]) + "\\\\hline")


# In[24]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stops_nl = stopwords.words('dutch')


# In[31]:


# ----------------------------------------------
# MOST COMMON WORDS AND VOC SIZRS (TODO)
# ----------------------------------------------
from nltk.corpus import stopwords
stops_nl = list(stopwords.words('dutch'))

import string

def get_most_freq(vocab, stopwords = None, k = 10):
    tmp_vocab = vocab
    if stopwords is not None:
        for word in stopwords:
            if word in tmp_vocab:
                del tmp_vocab[word]
    sorted_vocab = dict(sorted(tmp_vocab.items(), key=lambda item: item[1], reverse=True))
    
    print([w + " " + str(sorted_vocab[w]) for w in list(sorted_vocab.keys())[:k]])
    
# We also remove punctuation
print("Domain & model & voc size\\\\\hline")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref.", end=" & ")
    vocab_size, _vocab  = get_vocabulary([s.translate(str.maketrans("", "", string.punctuation)) for s in refs_tok[translation_name]])
    vocab_size_low, _vocab  = get_vocabulary_lowercase([s.translate(str.maketrans("", "", string.punctuation)) for s in refs_tok[translation_name]])
    print(str(vocab_size) + "\\\\\\hline")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        vocab_size, _vocab  = get_vocabulary([s.translate(str.maketrans("", "", string.punctuation)) for s in all_tok_translations[exp_prefix]])
        vocab_size_low, _vocab  = get_vocabulary_lowercase([s.translate(str.maketrans("", "", string.punctuation)) for s in all_tok_translations[exp_prefix]])
        print(str(vocab_size) + "\\\\\\hline")



# In[26]:


print("Most frequent words")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end = " & " )
    print("Ref.", end=" & ")
    _vocab_size_low, vocab  = get_vocabulary_lowercase([s.translate(str.maketrans("", "", string.punctuation)) for s in refs_tok[translation_name]])    
    print("Stopwords Incl.:", end=" ")
    get_most_freq(vocab)
    print("\\\\\\hline")
    print(" & & Stopwords Excl.:", end=" ")
    get_most_freq(vocab, stops_nl)
    print("\\\\\\hline")
    for model_name in ["mbart", "j2", "gpt"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        _vocab_size_low, vocab  = get_vocabulary_lowercase([s.translate(str.maketrans("", "", string.punctuation)) for s in all_tok_translations[exp_prefix]])
        print("Stopwords Incl.:", end=" ")
        get_most_freq(vocab)
        print("\\\\\\hline")
        print("& & Stopwords Excl.:", end=" ")
        get_most_freq(vocab, stops_nl)
        print("\\\\\\hline")


# In[12]:


# Get Average sentence length
def get_average_sentlength(sentences):
    all_lengths = [len(s.strip().split()) for s in sentences]
    return round(sum(all_lengths) / len(all_lengths), 3)

print("Domain & model & avg. sent length\\\\\hline")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref.", end=" & ")
    print(str(get_average_sentlength(refs_tok[translation_name])) + "\\\\\hline")
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        print(str(get_average_sentlength(all_tok_translations[exp_prefix])) + "\\\\\hline")


# In[35]:


sent_num = 42
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name, end="")
    print(" & Ref.", end=" & ")
    print(refs_tok[translation_name][sent_num])
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        print(" & " + model_name, end=" & ")
        print(all_tok_translations[exp_prefix][sent_num])


# # Regression

# In[4]:


get_ipython().system('pip install gensim')


# In[27]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re

get_ipython().run_line_magic('matplotlib', 'inline')
nltk.download('stopwords')

# df = pd.read_csv('combined-lit.csv', encoding = 'ISO-8859-2',sep = ";", header = 0)
# df['translation'] = df['translation'].fillna("")
# df = df[pd.notnull(df['model'])]
# print(df.head(10))
# print(df['translation'].apply(lambda x: len(x.split(' '))).sum())

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('dutch'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

combined_X = {} # the key is the domain the rest are just sentences
combined_y = {} # the key is the domain the rest are just the model names
combined_X_norefs = {} # the key is the domain the rest are just sentences
combined_y_norefs = {} # the key is the domain the rest are just the model names

for translation_name in ["news", "poetry", "literature"]:
    print(translation_name)
    combined_X[translation_name] = [clean_text(sent) for sent in refs_tok[translation_name]]
    combined_y[translation_name] = ["ref" for _ in range(len(refs_tok[translation_name]))]    
    combined_X_norefs[translation_name] = []
    combined_y_norefs[translation_name] = []
    for model_name in ["mbart", "j2", "gpt", "deepseek"]:
        exp_prefix = model_name+"_"+translation_name
        combined_X[translation_name] = combined_X[translation_name] + [clean_text(sent) for sent in all_tok_translations[exp_prefix]]
        combined_y[translation_name] = combined_y[translation_name] + [model_name for _ in range(len(all_tok_translations[exp_prefix]))]    
        combined_X_norefs[translation_name] = combined_X_norefs[translation_name] + [clean_text(sent) for sent in all_tok_translations[exp_prefix]]
        combined_y_norefs[translation_name] = combined_y_norefs[translation_name] + [model_name for _ in range(len(all_tok_translations[exp_prefix]))]    

# def print_plot(index):
#     example = df[df.index == index][['translation', 'model']].values[0]
#     if len(example) > 0:
#         print(example[0])
#         print('Tag:', example[1])
# print_plot(10)


# In[83]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

def get_logreg_best(X_train, X_test, y_train, y_test):
    logreg = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(n_jobs=-1, C=1e5)),
                   ])
    
    grid_params = {
      'lr__penalty': ['l1', 'l2'],
      'lr__C': [1, 10, 100, 1000],
      'lr__max_iter': [20, 50, 100, 150, 200],
      'lr__solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
    }
    clf = GridSearchCV(logreg, grid_params)
    clf.fit(X_train, y_train)
    print("Best Score: ", clf.best_score_)
    print("Best Params: ", clf.best_params_)


# In[84]:


for translation_name in ["news", "poetry", "literature"]:
    X_train, X_test, y_train, y_test = train_test_split(combined_X[translation_name], combined_y[translation_name], test_size=0.3, random_state = 42)
    get_logreg_best(X_train, X_test, y_train, y_test)


# In[87]:


def get_logreg(X_train, X_test, y_train, y_test):
    logreg = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(n_jobs=1, C=1, max_iter=150, penalty='l1', solver='saga')),
                   ])
    
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)
    
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=set(y_train+y_test)))


# In[88]:


for translation_name in ["news", "poetry", "literature"]:
    print(translation_name)
    X_train, X_test, y_train, y_test = train_test_split(combined_X_norefs[translation_name], combined_y_norefs[translation_name], test_size=0.3, random_state = 42)
    get_logreg(X_train, X_test, y_train, y_test)

print("With reference")
for translation_name in ["news", "poetry", "literature"]:
    print(translation_name)
    X_train, X_test, y_train, y_test = train_test_split(combined_X[translation_name], combined_y[translation_name], test_size=0.3, random_state = 42)
    get_logreg(X_train, X_test, y_train, y_test)


# In[64]:


import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

for i in range(1, 20):
    print(i)
    dbs = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                       ('db', DBSCAN(eps=0.3, min_samples=i)),
                    ])
    
    for translation_name in ["news", "poetry", "literature"]:
        print(translation_name)
    
        dbs.fit(combined_X_norefs[translation_name])
        labels = dbs.named_steps['db'].labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
    
    print("With refernce")
    for translation_name in ["news", "poetry", "literature"]:
        print(translation_name)
        dbs.fit(combined_X[translation_name])
        labels = dbs.named_steps['db'].labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)


# In[ ]:




