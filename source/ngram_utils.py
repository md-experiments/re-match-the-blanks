import numpy as np

def find_token_range(tok_doc, tok_trgt):
    """
    Returns the range where the tokens belong to from within a tokenized doc
    """
    res = []
    len_trgt = len(tok_trgt)
    if len_trgt:
        values = np.array(tok_doc)
        search_val = tok_trgt[0]
        cand_bgn = list(np.where(values == search_val)[0])
        for bgn in cand_bgn:
            if tok_doc[bgn:bgn+len_trgt] == tok_trgt:
                res.append([bgn,bgn+len_trgt])
    return res