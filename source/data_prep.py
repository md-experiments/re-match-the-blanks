from source.ngram_utils import find_token_range

def update_loc(sent_toks,tail, dict_loc):
    tail_loc = find_token_range(sent_toks,tail.split(), few_rel_location=True)
    if len(dict_loc) == 2:
        dict_loc.append(tail_loc)
    else:
        dict_loc[2] = tail_loc
    return dict_loc


def sentence_masker(few_rel_entry, entity_tokens, method = 'bracket'):
    sent_toks = few_rel_entry['tokens']
    result = sent_toks.copy()
    if few_rel_entry['h'][2][0][0] > few_rel_entry['t'][2][0][0]:
        insert_sequence = ['h','t']
    else:
        insert_sequence = ['t','h']
    for h_or_t in insert_sequence:
        result = insert_tokens_bracket(result, few_rel_entry, entity_tokens, h_or_t, method = method)
    return ' '.join(result)

def insert_tokens_bracket(sent_toks, few_rel_entry, entity_tokens, h_or_t, method = 'bracket'):
    brn_ent = few_rel_entry[h_or_t][2][0][0]
    end_ent = few_rel_entry[h_or_t][2][0][-1]
    if method == 'bracket':
        result = sent_toks[:brn_ent] + \
            [entity_tokens[h_or_t][0]] + sent_toks[brn_ent:end_ent+1] + \
            [entity_tokens[h_or_t][1]] + sent_toks[end_ent+1:]
    elif method == 'mask_one':
        result = sent_toks[:brn_ent] + \
            [entity_tokens[h_or_t][0]] + sent_toks[end_ent+1:]
    elif method == 'mask_span':
        len_toks = int(end_ent - brn_ent+1)
        if len_toks==1:
            toks_insert = [entity_tokens[h_or_t][0]]
        elif len_toks>1:
            toks_insert = [entity_tokens[h_or_t][0]]*(len_toks-1) + [entity_tokens[h_or_t][1]]
        result = sent_toks[:brn_ent] + \
            toks_insert + sent_toks[end_ent+1:]   
    elif method == 'ignore':
        result = sent_toks.copy()
    return result