import numpy as np

def find_token_range(tok_doc, tok_trgt, few_rel_location=False):
    """
    Returns the range where the tokens belong to from within a tokenized doc
    NB: Notice that this currently assumes the tokenizer of the short series will be the same as the tokenization of the long one
    There are cases where sub-series will tokenize differently, esp when using cased tokenizers
    """
    res = []
    len_trgt = len(tok_trgt)
    if len_trgt:
        values = np.array(tok_doc)
        search_val = tok_trgt[0]
        cand_bgn = list(np.where(values == search_val)[0])
        for bgn in cand_bgn:
            if tok_doc[bgn:bgn+len_trgt] == tok_trgt:
                if few_rel_location:
                    range_found = list(range(bgn,bgn+len_trgt))
                    range_found = range_found*2 if len(range_found)==1 else range_found
                    res.append(range_found)
                else:
                    res.append([bgn,bgn+len_trgt])
    return res