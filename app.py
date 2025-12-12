from flask import Flask, render_template, request, jsonify
import re
from collections import defaultdict
import math

app = Flask(__name__)

documents = {
    "Doc1": "breakthrough drug for schizophrenia",
    "Doc2": "new schizophrenia drug",
    "Doc3": "new approach for treatment of schizophrenia",
    "Doc4": "new hopes for schizophrenia patients"
}

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
              'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
              'that', 'the', 'to', 'was', 'will', 'with'}

def tokenize(text):
    return re.sub(r'[.,!?;:"\']', '', text.lower()).split()

def remove_stop_words(tokens):
    return [t for t in tokens if t not in stop_words]

def stem(word):
    if word.endswith('sses'): return word[:-2]
    if word.endswith('ies'): return word[:-3] + 'i'
    if word.endswith('ss'): return word
    if word.endswith('s'): return word[:-1]
    if word.endswith('ing'): return word[:-3]
    if word.endswith('ed'): return word[:-2]
    return word

def process_text(text):
    tokens = tokenize(text)
    without_stop = remove_stop_words(tokens)
    stemmed = [stem(t) for t in without_stop]
    return {'tokens': tokens, 'without_stop_words': without_stop, 'stemmed': stemmed}

def build_inverted_index():
    index = defaultdict(list)
    for doc_id, text in documents.items():
        for term in set(process_text(text)['stemmed']):
            index[term].append(doc_id)
    return {k: sorted(v) for k, v in index.items()}

def build_positional_index():
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in documents.items():
        for pos, term in enumerate(process_text(text)['stemmed']):
            index[term][doc_id].append(pos)
    return dict(index)

def add_skip_pointers(lst):
    if len(lst) <= 2: return []
    skip = int(math.sqrt(len(lst)))
    return [{'from_index': i, 'from_value': lst[i], 'to_index': i + skip, 'to_value': lst[i + skip]}
            for i in range(0, len(lst), skip) if i + skip < len(lst)]

def boolean_and_with_skip(l1, l2, use_skip=False):
    result, comps, skips = [], [], []
    skip_map1 = {sp['from_index']: sp for sp in add_skip_pointers(l1)} if use_skip else {}
    skip_map2 = {sp['from_index']: sp for sp in add_skip_pointers(l2)} if use_skip else {}
    i = j = 0
    
    while i < len(l1) and j < len(l2):
        comps.append({'list1_idx': i, 'list1_val': l1[i], 'list2_idx': j, 'list2_val': l2[j]})
        
        if l1[i] == l2[j]:
            result.append(l1[i])
            comps[-1]['result'] = 'match'
            i += 1
            j += 1
        elif l1[i] < l2[j]:
            if use_skip and i in skip_map1 and skip_map1[i]['to_value'] <= l2[j]:
                skips.append({'list': 1, 'from': skip_map1[i]['from_value'], 'to': skip_map1[i]['to_value']})
                comps[-1]['skip'] = True
                i = skip_map1[i]['to_index']
            else:
                i += 1
        else:
            if use_skip and j in skip_map2 and skip_map2[j]['to_value'] <= l1[i]:
                skips.append({'list': 2, 'from': skip_map2[j]['from_value'], 'to': skip_map2[j]['to_value']})
                comps[-1]['skip'] = True
                j = skip_map2[j]['to_index']
            else:
                j += 1
    
    return result, comps, skips

def boolean_and(l1, l2):
    result, i, j = [], 0, 0
    while i < len(l1) and j < len(l2):
        if l1[i] == l2[j]:
            result.append(l1[i])
            i += 1
            j += 1
        elif l1[i] < l2[j]:
            i += 1
        else:
            j += 1
    return result

def boolean_or(l1, l2):
    return sorted(set(l1 + l2))

def boolean_not(l1, all_docs):
    return [d for d in all_docs if d not in l1]

def soundex(word):
    word = word.upper()
    if not word: return ""
    first = word[0]
    word = re.sub(r'[AEIOUHWY]', '0', word[1:])
    word = re.sub(r'[BFPV]', '1', word)
    word = re.sub(r'[CGJKQSXZ]', '2', word)
    word = re.sub(r'[DT]', '3', word)
    word = re.sub(r'[L]', '4', word)
    word = re.sub(r'[MN]', '5', word)
    word = re.sub(r'[R]', '6', word)
    word = re.sub(r'(.)\1+', r'\1', word).replace('0', '')
    return (first + word + '000')[:4]

def process_boolean_query(query, use_skip=False):
    inv_idx = build_inverted_index()
    all_docs = sorted(documents.keys())
    steps = []
    
    idx_data = {t: {'postings': p, 'skip_pointers': add_skip_pointers(p) if use_skip else []} 
                for t, p in inv_idx.items()}
    steps.append({'type': 'index', 'label': f'Inverted Index{" (with Skip Pointers)" if use_skip else ""}', 
                  'data': idx_data, 'use_skip': use_skip})
    
    processed = process_text(query)
    steps.append({'type': 'processing', 'label': 'Query Processing', 'data': processed})
    
    tokens = query.lower().split()
    
    if 'and' in tokens:
        terms = [t for t in processed['stemmed'] if t != 'and']
        result = inv_idx.get(terms[0], [])
        steps.append({'type': 'operation', 'label': f'Postings for "{terms[0]}"', 
                     'data': {'term': terms[0], 'postings': result, 
                             'skip_pointers': add_skip_pointers(result) if use_skip else []}})
        
        for term in terms[1:]:
            next_post = inv_idx.get(term, [])
            steps.append({'type': 'operation', 'label': f'Postings for "{term}"',
                         'data': {'term': term, 'postings': next_post,
                                 'skip_pointers': add_skip_pointers(next_post) if use_skip else []}})
            
            if use_skip:
                result, comps, skips = boolean_and_with_skip(result, next_post, True)
                steps.append({'type': 'merge_with_skip', 'label': 'AND with Skip Pointers',
                             'data': {'result': result, 'comparisons': comps, 'skips_used': skips,
                                     'total_comparisons': len(comps), 'total_skips': len(skips)}})
            else:
                result = boolean_and(result, next_post)
                steps.append({'type': 'merge', 'label': 'AND Operation', 'data': result})
        
        return {'result': result, 'steps': steps}
    
    elif 'or' in tokens:
        terms = [t for t in processed['stemmed'] if t != 'or']
        result = inv_idx.get(terms[0], [])
        for term in terms[1:]:
            result = boolean_or(result, inv_idx.get(term, []))
        steps.append({'type': 'result', 'label': 'Result', 'data': result})
        return {'result': result, 'steps': steps}
    
    elif 'not' in tokens:
        idx = tokens.index('not')
        if idx + 1 < len(processed['stemmed']):
            term = processed['stemmed'][idx + 1]
            result = boolean_not(inv_idx.get(term, []), all_docs)
            steps.append({'type': 'result', 'label': 'Result', 'data': result})
            return {'result': result, 'steps': steps}
    
    term = processed['stemmed'][0] if processed['stemmed'] else ''
    result = inv_idx.get(term, [])
    steps.append({'type': 'result', 'label': 'Result', 'data': result})
    return {'result': result, 'steps': steps}

def process_phrase_query(phrase):
    pos_idx = build_positional_index()
    processed = process_text(phrase)
    terms = processed['stemmed']
    steps = []
    
    steps.append({'type': 'index', 'label': 'Positional Index', 'data': pos_idx})
    steps.append({'type': 'processing', 'label': 'Phrase Processing', 
                 'data': {'original': tokenize(phrase), 'stemmed': terms}})
    
    if not terms:
        return {'result': [], 'steps': steps, 'details': []}
    
    # Show documents containing each term with positions
    term_docs = {}
    for term in terms:
        if term in pos_idx:
            term_docs[term] = list(pos_idx[term].keys())
            steps.append({'type': 'operation', 'label': f'Documents with "{term}"',
                         'data': {'term': term, 'docs': term_docs[term], 
                                 'positions': pos_idx[term]}})
        else:
            steps.append({'type': 'operation', 'label': f'Term "{term}" not found', 
                         'data': {'term': term, 'docs': []}})
            return {'result': [], 'steps': steps, 'details': []}
    
    # Find common documents
    common_docs = set(term_docs[terms[0]])
    for term in terms[1:]:
        common_docs &= set(term_docs.get(term, []))
    
    if common_docs:
        steps.append({'type': 'operation', 'label': 'Common Documents',
                     'data': {'docs': sorted(common_docs)}})
    
    result, details = [], []
    
    # Check each common document for consecutive positions
    for doc in common_docs:
        first_positions = pos_idx[terms[0]][doc]
        
        for start_pos in first_positions:
            found = True
            for i in range(1, len(terms)):
                expected_pos = start_pos + i
                if expected_pos not in pos_idx[terms[i]][doc]:
                    found = False
                    break
            
            if found:
                if doc not in result:
                    result.append(doc)
                    positions = [start_pos + i for i in range(len(terms))]
                    details.append({'doc': doc, 'start_pos': start_pos, 'phrase_positions': positions})
    
    steps.append({'type': 'result', 'label': 'Phrase Query Result', 'data': result})
    
    if details:
        steps.append({'type': 'phrase_details', 'label': 'Match Details', 'data': details})
    
    return {'result': result, 'steps': steps, 'details': details}

def process_soundex_query(query):
    q_sdx = soundex(query)
    steps = [{'type': 'soundex', 'label': f'Soundex for "{query}"', 'data': {'query': query, 'code': q_sdx}}]
    
    all_terms = set()
    for text in documents.values():
        all_terms.update(process_text(text)['stemmed'])
    
    matches = [{'term': t, 'soundex': soundex(t)} for t in all_terms if soundex(t) == q_sdx]
    steps.append({'type': 'soundex_matches', 'label': 'Matching Terms', 'data': matches})
    
    inv_idx = build_inverted_index()
    result = sorted(set(doc for m in matches for doc in inv_idx.get(m['term'], [])))
    
    return {'result': result, 'steps': steps, 'matches': matches}

@app.route('/')
def index():
    return render_template('index.html', documents=documents)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', 'boolean')
    use_skip = data.get('use_skip', False)
    
    if mode == 'boolean':
        return jsonify(process_boolean_query(query, use_skip))
    elif mode == 'phrase':
        return jsonify(process_phrase_query(query))
    elif mode == 'soundex':
        return jsonify(process_soundex_query(query))
    return jsonify({'result': [], 'steps': []})

@app.route('/add_document', methods=['POST'])
def add_document():
    data = request.json
    doc_id, doc_text = data.get('doc_id', ''), data.get('doc_text', '')
    if doc_id and doc_text:
        documents[doc_id] = doc_text
        return jsonify({'success': True, 'documents': documents})
    return jsonify({'success': False})

@app.route('/get_documents', methods=['GET'])
def get_documents():
    return jsonify(documents)

if __name__ == '__main__':
    app.run(debug=True, port=5000)