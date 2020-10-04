import pandas as pd
import numpy as np
from nltk.stem import *
from tqdm.auto import tqdm
tqdm.pandas()

stemmer = PorterStemmer()


def get_counts_df_and_stem(input_df, keep_na = False):
    """
    input the raw dataframe and output the counts of aspects, opinions, stemmed
    
    if keep_na is true, you will get aspects/opinions with no pair. replacement pair would be the empty string
    
    two progress bars are shown
    """
    if keep_na:
        input_df['aspects'][input_df['aspects'].isna()] = ''
        input_df['opinions'][input_df['opinions'].isna()] = ''
    main_df = input_df.groupby(['aspects', 'opinions']).count().sort_values('review_ids', ascending = False)
    main_df.reset_index(inplace= True)
    main_df.drop('sentiments', axis = 1, inplace = True)
    main_df = main_df.rename({'review_ids': 'samples'},axis = 1)
    counts = main_df['aspects'].value_counts()
    stemmed_idx = [stemmer.stem(aspect) for aspect in counts.index]
    stemming_df = pd.DataFrame({'word': counts.index, 'stemmed': stemmed_idx, 'count': counts}, index = None)
    aspects_counts = stemming_df.groupby('stemmed').agg({'word': 'first', 'count': 'sum'}).reset_index().sort_values('count', ascending = False)
    counts = main_df['opinions'].value_counts()
    stemmed_idx = [stemmer.stem(aspect) for aspect in counts.index]
    stemming_df = pd.DataFrame({'word': counts.index, 'stemmed': stemmed_idx, 'count': counts}, index = None)
    opinions_counts = stemming_df.groupby('stemmed').agg({'word': 'first', 'count': 'sum'}).reset_index().sort_values('count', ascending = False)
    stem2aspect = {row['stemmed']: row['word'] for idx, row in aspects_counts.iterrows()}
    stem2opinion = {row['stemmed']: row['word'] for idx, row in opinions_counts.iterrows()}
    main_df['aspects'] = main_df['aspects'].progress_apply(lambda x: stem2aspect[stemmer.stem(x)])
    main_df['opinions'] = main_df['opinions'].progress_apply(lambda x: stem2opinion[stemmer.stem(x)])
    main_df = main_df.groupby(['aspects', 'opinions']).sum().sort_values('samples', ascending = False).reset_index()
    return main_df


def get_baseline_proportions(df, threshold = 0):
    """
    input a dataframe and return a dictionary of two dataframes,
    one for aspects and opinions with columns: 
    [word, samples, proportion, std]
    std is calculated by sqrt(p*(1-p))
    if samples is below threshold the word is excluded
    """
    ret_dict = dict()
    for kind in ['aspects', 'opinions']:
        counts = df[[kind, 'samples']]
        counts = counts.groupby(kind).sum().reset_index().sort_values('samples', ascending = False)
        sum_ = counts['samples'].sum()
        counts = counts[counts['samples'] >= threshold]
        counts['proportion'] = counts['samples']/sum_
        counts = counts.rename({kind: 'word'}, axis = 1)
        ret_dict[kind] = counts
    return ret_dict

def get_pct_change_df(to_score, reference_df, threshold = 0):
    """
    to_score: must be a dataframe with a word and samples column, can be gotten from counts_df(series)
    reference_df: output of get_proportion_and_std(main_df) to use as true mean and proportion
    returns df with columns [word, samples, proportion, baseline_prop, change] included
    """
    if to_score.columns[0] != 'word':
        to_change = to_score.columns[0]
        to_score = to_score.rename({to_change: 'word'}, axis = 1)
    sum_ = to_score['samples'].sum()
    to_score = to_score[to_score['samples'] > threshold]
    subset = reference_df[['word', 'proportion']]
    merged = pd.merge(to_score, subset, on = 'word')
    words = merged['word'].values
    samples = merged['samples'].values
    sample_props = samples/sum_
    true_prop = merged['proportion'].values
    pcts = (sample_props - true_prop)/true_prop
    return pd.DataFrame({'word': words, 'samples': samples, 'proportion': sample_props, 'baseline_prop': true_prop, 'change': pcts}).sort_values('change', ascending = False)
    
    
def get_pct_change_for_word(counts_df, baseline, word, wordtype = 'aspects', threshold = 0):
    """
    pass in a word and wordtype (must be in [aspects, opinions])
    baseline must be a df that has baseline proportions for change comparison
    counts_df must be dataframe of pairs counts as in the get_counts function
    threshold is min number an aspect/opinion must be mentioned for it to be in the returned df
    """
    assert wordtype in ['aspects', 'opinions'], "wordtype must be 'aspects' or 'opinions'"
    other_samples = get_counts_for_word(word, counts_df, wordtype)
    return get_pct_change_df(other_samples, baseline, threshold)
    
    
    
def get_pct_change_for_words(counts_df, words, wordtype = 'aspects', threshold = 0, baseline_threshold = 0):
    """
    does more work for you get_pct_change_for_word by calculating the baseline
    words is a list of words,
    counts_df is a df like get_counts function returns
    word type must be 'aspects' or 'opinions', depending on what the words are
    threshold is min number an aspect/opinion must be mentioned for it to be in the returned df
    baseline_threshold is min number an aspect/opinion must be mentioned for it to be in the baseline and even considered
    """
    assert wordtype in ['aspects', 'opinions'], "wordtype must be 'aspects' or 'opinions'"
    if wordtype == 'aspects':
        other_type = 'opinions'
    else:
        other_type = 'aspects'
    baseline = get_baseline_proportions(counts_df, threshold = baseline_threshold)
    baseline = baseline[other_type]
    ret_dict = dict()
    for word in tqdm(words):
        ret_df = get_pct_change_for_word(counts_df, baseline, word, wordtype = wordtype, threshold = threshold)
        ret_dict[word] = ret_df
    return ret_dict

def add_dfs(dfs):
    """
    input is a list of dataframes with ['word', 'samples'], and this data is aggregated and returned as a single dataframe
    """
    for df in dfs:
        if df.columns[0] != 'word':
            to_change = df.columns[0]
            df.rename({to_change: 'word'}, axis = 1, inplace = True)
    return pd.concat(dfs).groupby('word').sum().reset_index().sort_values('samples', ascending = False)

def get_counts_for_word(word, counts_df, wordtype = 'aspects'):
    """
    input is a word, returns the list of aspects/opinions associated w that word and their sample count
    """
    assert wordtype in ['aspects', 'opinions'], "wordtype must be 'aspects' or 'opinions'"
    if wordtype == 'aspects':
        other_type = 'opinions'
    else:
        other_type = 'aspects'
    return counts_df[counts_df[wordtype] == word][[other_type ,'samples']]