#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
N-Gram Language Model for Auto-Completion
This module implements functions for loading, processing, and making predictions with
N-gram language models.
"""

import os
import math
import pickle
import nltk
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any


def preprocess_pipeline(data: str) -> List[List[str]]:
    """
    Preprocess text data for training an N-gram model.
    
    Args:
        data: Raw text data
        
    Returns:
        List of tokenized sentences
    """
    # Split by newline character
    sentences = data.split('\n')

    # Remove leading and trailing spaces
    sentences = [s.strip() for s in sentences]

    # Drop Empty Sentences
    sentences = [s for s in sentences if len(s) > 0]

    # Empty List to hold Tokenized Sentences
    tokenized = []

    # Iterate through sentences
    for sentence in sentences:
        # Convert to lowercase
        sentence = sentence.lower()

        # Convert to a list of words
        token = nltk.word_tokenize(sentence)

        # Append to list
        tokenized.append(token)

    return tokenized


def count_the_words(sentences: List[List[str]]) -> Dict[str, int]:
    """
    Count word frequency in the dataset.
    
    Args:
        sentences: List of tokenized sentences
        
    Returns:
        Dictionary mapping words to their frequency counts
    """
    # Creating a Dictionary of counts
    word_counts = {}

    # Iterating over sentences
    for sentence in sentences:
        # Iterating over Tokens
        for token in sentence:
            # Add count for new word
            if token not in word_counts.keys():
                word_counts[token] = 1
            # Increase count by one
            else:
                word_counts[token] += 1

    return word_counts


def handling_oov(tokenized_sentences: List[List[str]], count_threshold: int) -> List[str]:
    """
    Create a closed vocabulary with words appearing at least count_threshold times.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        count_threshold: Minimum frequency to include a word in vocabulary
        
    Returns:
        List of words in the vocabulary
    """
    # Empty list for closed vocabulary
    closed_vocabulary = []

    # Obtain frequency dictionary
    words_count = count_the_words(tokenized_sentences)

    # Iterate over words and counts
    for word, count in words_count.items():
        # Append if it's more(or equal) to the threshold
        if count >= count_threshold:
            closed_vocabulary.append(word)

    return closed_vocabulary


def unk_tokenize(tokenized_sentences: List[List[str]], vocabulary: List[str], unknown_token: str = "<unk>") -> List[List[str]]:
    """
    Replace words not in vocabulary with UNK token.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        vocabulary: List of words in vocabulary
        unknown_token: Token to use for out-of-vocabulary words
        
    Returns:
        List of tokenized sentences with OOV words replaced by UNK
    """
    # Convert Vocabulary into a set for faster lookup
    vocabulary_set = set(vocabulary)

    # Create empty list for sentences
    new_tokenized_sentences = []

    # Iterate over sentences
    for sentence in tokenized_sentences:
        # Iterate over sentence and add <unk> if token is absent from the vocabulary
        new_sentence = []
        for token in sentence:
            if token in vocabulary_set:
                new_sentence.append(token)
            else:
                new_sentence.append(unknown_token)

        # Append sentence to the new list
        new_tokenized_sentences.append(new_sentence)

    return new_tokenized_sentences


def cleansing(train_data: List[List[str]], test_data: List[List[str]], count_threshold: int) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Clean training and test data by handling OOV words.
    
    Args:
        train_data: Training data as list of tokenized sentences
        test_data: Test data as list of tokenized sentences
        count_threshold: Minimum word frequency to include in vocabulary
        
    Returns:
        Tuple with cleaned train data, cleaned test data, and vocabulary
    """
    # Get closed Vocabulary
    vocabulary = handling_oov(train_data, count_threshold)

    # Updated Training Dataset
    new_train_data = unk_tokenize(train_data, vocabulary)

    # Updated Test Dataset
    new_test_data = unk_tokenize(test_data, vocabulary)

    return new_train_data, new_test_data, vocabulary


def count_n_grams(data: List[List[str]], n: int, start_token: str = "<s>", end_token: str = "<e>") -> Dict[Tuple[str, ...], int]:
    """
    Count n-gram frequencies in the dataset.
    
    Args:
        data: List of tokenized sentences
        n: Size of n-gram
        start_token: Token to use at beginning of sentence
        end_token: Token to use at end of sentence
        
    Returns:
        Dictionary mapping n-grams to their frequency counts
    """
    # Empty dict for n-grams
    n_grams = {}

    # Iterate over all sentences in the dataset
    for sentence in data:
        # Append n start tokens and a single end token to the sentence
        sentence = [start_token]*n + sentence + [end_token]

        # Convert the sentence into a tuple
        sentence = tuple(sentence)

        # Temp var to store length from start of n-gram to end
        m = len(sentence) if n==1 else len(sentence)-1

        # Iterate over this length
        for i in range(m):
            # Get the n-gram
            n_gram = sentence[i:i+n]

            # Add the count of n-gram as value to our dictionary
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

    return n_grams


def prob_for_single_word(word: str, previous_n_gram: Tuple[str, ...], 
                        n_gram_counts: Dict[Tuple[str, ...], int], 
                        nplus1_gram_counts: Dict[Tuple[str, ...], int], 
                        vocabulary_size: int, k: float = 1.0) -> float:
    """
    Calculate probability of a word given its n-gram context.
    
    Args:
        word: The word to calculate probability for
        previous_n_gram: Context n-gram preceding the word
        n_gram_counts: Dictionary of n-gram counts
        nplus1_gram_counts: Dictionary of (n+1)-gram counts
        vocabulary_size: Size of vocabulary 
        k: Smoothing parameter for add-k smoothing
        
    Returns:
        Probability of the word given the context
    """
    # Convert the previous_n_gram into a tuple
    previous_n_gram = tuple(previous_n_gram)

    # Calculating the count, if exists from our freq dictionary otherwise zero
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)

    # The Denominator
    denom = previous_n_gram_count + k * vocabulary_size

    # previous n-gram plus the current word as a tuple
    nplus1_gram = previous_n_gram + (word,)

    # Calculating the nplus1 count, if exists from our freq dictionary otherwise zero
    nplus1_gram_count = nplus1_gram_counts.get(nplus1_gram, 0)

    # Numerator
    num = nplus1_gram_count + k

    # Final Fraction
    prob = num / denom
    return prob


def probs(previous_n_gram: Tuple[str, ...], 
         n_gram_counts: Dict[Tuple[str, ...], int], 
         nplus1_gram_counts: Dict[Tuple[str, ...], int], 
         vocabulary: List[str], k: float = 1.0) -> Dict[str, float]:
    """
    Get probabilities for all words in the vocabulary given a context.
    
    Args:
        previous_n_gram: The context n-gram
        n_gram_counts: Dictionary of n-gram counts
        nplus1_gram_counts: Dictionary of (n+1)-gram counts
        vocabulary: List of words in vocabulary
        k: Smoothing parameter
        
    Returns:
        Dictionary mapping words to their probabilities
    """
    # Convert to Tuple
    previous_n_gram = tuple(previous_n_gram)

    # Add end and unknown tokens to the vocabulary
    vocab_with_special = vocabulary + ["<e>", "<unk>"]

    # Calculate the size of the vocabulary
    vocabulary_size = len(vocab_with_special)

    # Empty dict for probabilities
    probabilities = {}

    # Iterate over words
    for word in vocab_with_special:
        # Calculate probability
        probability = prob_for_single_word(word, previous_n_gram,
                                          n_gram_counts, nplus1_gram_counts,
                                          vocabulary_size, k=k)
        # Create mapping: word -> probability
        probabilities[word] = probability

    return probabilities


def predict_next_word(input_text: str, n_gram_counts_list: List[Dict[Tuple[str, ...], int]], 
                     vocabulary: List[str], k: float = 1.0, 
                     num_predictions: int = 5) -> List[Tuple[str, float]]:
    """
    Predict the next word given input text - improved robustness.
    
    Args:
        input_text: Input text to base prediction on
        n_gram_counts_list: List of n-gram count dictionaries
        vocabulary: List of words in vocabulary
        k: Smoothing parameter
        num_predictions: Number of predictions to return
        
    Returns:
        List of tuples (word, probability) for top predictions
    """
    # Convert vocabulary to set for faster lookups
    vocab_set = set(vocabulary)
    
    # Tokenize the input text - handle empty input gracefully
    if not input_text:
        tokens = []
    else:
        try:
            tokens = nltk.word_tokenize(input_text.lower())
        except Exception as e:
            print(f"Tokenization error: {e}")
            tokens = input_text.lower().split()
    
    # Replace OOV words with <unk>
    tokens = ['<unk>' if token not in vocab_set else token for token in tokens]
    
    # Get the maximum n-gram size from our model counts
    max_n = len(n_gram_counts_list) if n_gram_counts_list else 0
    if max_n == 0:
        return [("<unk>", 1.0)]
    
    # If we have fewer tokens than our maximum n-gram size,
    # we'll use what we have and pad with start tokens if needed
    context_size = min(max_n, len(tokens))
    context = tokens[-context_size:] if tokens else []
    
    # Pad with start tokens if needed
    if context_size < max_n:
        context = ['<s>'] * (max_n - context_size) + context
    
    # We'll use the largest n-gram model for prediction (if available)
    n = len(n_gram_counts_list)
    
    # Safety check to ensure we have at least 2 n-gram levels
    if n < 2:
        return [("<unk>", 1.0)]
    
    try:
        n_gram_counts = n_gram_counts_list[n-2]  # n-1 gram counts
        nplus1_gram_counts = n_gram_counts_list[n-1]  # n gram counts
    except IndexError:
        # Fallback to lower n-grams if higher ones aren't available
        if n > 2:
            n_gram_counts = n_gram_counts_list[0]  # Use unigrams
            nplus1_gram_counts = n_gram_counts_list[1]  # Use bigrams
        else:
            return [("<unk>", 1.0)]
    
    # Get previous n-gram (context)
    previous_n_gram = tuple(context[-(n-1):])
    
    # Get probabilities for all words in vocabulary
    try:
        all_probs = probs(previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary, k=k)
    except Exception as e:
        print(f"Error calculating probabilities: {e}")
        return [("<unk>", 1.0)]
    
    # Sort words by probability and get top predictions
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top_predictions = sorted_probs[:num_predictions]
    
    return top_predictions


def load_model(model_path: str, vocab_path: str) -> Tuple[List[Dict[Tuple[str, ...], int]], List[str]]:
    """
    Load a saved n-gram model and vocabulary.
    
    Args:
        model_path: Path to the n-gram counts file
        vocab_path: Path to the vocabulary file
        
    Returns:
        Tuple with n-gram counts list and vocabulary list
    """
    try:
        with open(model_path, 'rb') as f:
            n_gram_counts_list = pickle.load(f)
        
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f)
        
        return n_gram_counts_list, vocabulary
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# Function to evaluate perplexity
def evaluate_perplexity(test_sentences: List[List[str]], 
                       n_gram_counts_list: List[Dict[Tuple[str, ...], int]], 
                       vocabulary: List[str], k: float = 1.0) -> float:
    """
    Evaluate model performance using perplexity on test data.
    Lower perplexity indicates better model performance.
    
    Args:
        test_sentences: List of tokenized test sentences
        n_gram_counts_list: List of n-gram count dictionaries
        vocabulary: The vocabulary list
        k: Smoothing parameter
        
    Returns:
        Average perplexity across test sentences
    """
    n = len(n_gram_counts_list)
    n_gram_counts = n_gram_counts_list[n-2]  # (n-1)-gram counts
    nplus1_gram_counts = n_gram_counts_list[n-1]  # n-gram counts
    
    total_log_prob = 0
    total_tokens = 0
    
    # Use a subset of test sentences for efficiency
    test_subset = test_sentences[:100] if len(test_sentences) > 100 else test_sentences
    
    for sentence in test_subset:
        # Add start and end tokens
        sentence = ['<s>'] * (n-1) + sentence + ['<e>']
        
        # Calculate probability of each word given previous n-1 words
        for i in range(n-1, len(sentence)):
            previous_n_gram = tuple(sentence[i-(n-1):i])
            word = sentence[i]
            
            # Get probability of this word
            word_prob = prob_for_single_word(word, previous_n_gram, 
                                           n_gram_counts, nplus1_gram_counts, 
                                           len(vocabulary)+2, k)
            
            # Add log probability
            if word_prob > 0:  # Avoid log(0)
                total_log_prob += math.log2(word_prob)
            total_tokens += 1
    
    # Calculate perplexity
    if total_tokens == 0:
        return float('inf')
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = 2 ** (-avg_log_prob)
    
    return perplexity