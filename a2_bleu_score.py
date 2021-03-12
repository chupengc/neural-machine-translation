"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
"""

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    """Get all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of token ids or words representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    ngrams = []

    for i in range(len(seq) - n + 1):
        ngrams.append(seq[i:i + n])

    return ngrams


def n_gram_precision(reference, candidate, n):
    """Compute the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    total = len(grouper(candidate, n))
    count = 0

    if len(candidate) == 0 or total == 0:
        return 0

    for ngram in grouper(candidate, n):
        if ngram in grouper(reference, n):
            count += 1

    return count / total


def brevity_penalty(reference, candidate):
    """Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    c = len(candidate)
    r = len(reference)

    if c == 0:
        return 0

    brevity = r / c
    BP = 1 if brevity < 1 else exp(1 - brevity)

    return BP


def BLEU_score(reference, candidate, n):
    """Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """
    bp = brevity_penalty(reference, candidate)
    p = 1

    for i in range(n):
        p_i = n_gram_precision(reference, candidate, i + 1)
        p *= p_i

    bleu = bp * p ** (1 / n)

    return bleu
