import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict

class TripletExtractor:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/medicine-LLM", use_fast=False)
        self.encoder = AutoModelForCausalLM.from_pretrained("AdaptLLM/medicine-LLM")
        self.encoder.eval()
        self.nlp = spacy.load("en_core_sci_lg")
        self.valid_relations = ['be', 'have', 'cause', 'treat', 'prevent', 'diagnose', 'increase', 'decrease', 'associate', 'affect', 'reduce', 'enhance', 'induce', 'inhibit', 'block', 'promote', 'stimulate', 'suppress', 'regulate', 'modulate', 'activate', 'bind', 'interact']
        
    def get_attention_mask(self, length):
        return torch.tensor([1] * length).unsqueeze(0)

    def process_sentence(self, sentence):
        input_tokens, token_idx_to_chunk_idx, chunk_idx_to_text, spacy_text, spacy_chunks = self.prepare_inputs(sentence)

        # Create Model Input
        model_inputs = {
            'input_ids': torch.tensor([self.tokenizer.bos_token_id] + input_tokens + [self.tokenizer.eos_token_id]).unsqueeze(0),
            'attention_mask': self.get_attention_mask(len(input_tokens) + 2)  # all token are attended to
        }

        with torch.no_grad():
            outputs = self.encoder(**model_inputs, output_attentions=True)

        # Take the mean of the last 6 layers of the model
        attention = outputs.attentions[-1][0].mean(dim=0)
        attention = attention.detach().cpu().numpy()
        attention = attention[1:-1, 1:-1]
        
        attention_compressed = self.compress_attention(attention, token_idx_to_chunk_idx)

        #  create a graph from the attention matrix
        graph = defaultdict(list)
        for idx in range(0, len(attention_compressed)):
            for jdx in range(idx+1, len(attention_compressed)):
                graph[idx].append((jdx, attention_compressed[idx][jdx]))

        tail_head_pairs = []
        for head in spacy_chunks:
            for tail in spacy_chunks:
                if head != tail:
                    tail_head_pairs.append((head, tail))

        all_relation_pairs = []
        for output in [self.BFS(pair[0], pair[1], graph) for pair in tail_head_pairs]:
            if len(output):
                all_relation_pairs += [(o, chunk_idx_to_text) for o in output]
        triplets = [self.filter_relation_sets(relation_pair) for relation_pair in all_relation_pairs]
        triplets = [triplet for triplet in triplets if triplet and triplet['confidence'] > 0.2]
        return triplets

    # Use Scispacy to combine the selected entities into a single entity
    def prepare_inputs(self, sentence):
        doc = self.nlp(sentence)

        token_index, chunk_index, noun_index = 0, 0, 0
             
        start_indexes = [chunk.start for chunk in doc.ents]
        end_indexes = [chunk.end for chunk in doc.ents]
        chunk_idx_to_text =  {}
        token_idx_to_chunk_idx = []
        input_tokens = []
        spacy_text, spacy_chunks = [], []
        while (token_index < len(doc)):
            if token_index in start_indexes:
                # record that all the token indices in the noun chunk are the same 
                try:              
                    input_ids = self.tokenizer(doc.ents[noun_index].text, add_special_tokens = False)['input_ids']
                except:
                    import pdb
                    pdb.set_trace()
                input_tokens += input_ids
                token_idx_to_chunk_idx += [chunk_index] * len(input_ids)
                chunk_idx_to_text[chunk_index] = doc.ents[noun_index].text
                
                # record the spacy text and chunk index
                spacy_chunks.append(chunk_index)
                spacy_text.append(doc.ents[noun_index].text)

                # update - move forward by the number of tokens in the noun chunk
                token_index += end_indexes[noun_index] - start_indexes[noun_index]
                chunk_index += 1
                noun_index += 1
            else:
                input_ids = self.tokenizer(doc[token_index].text, add_special_tokens = False)['input_ids']
                input_tokens += input_ids
                token_idx_to_chunk_idx += [chunk_index] * len(input_ids)
                chunk_idx_to_text[chunk_index] = doc[token_index].text

                # update
                token_index += 1
                chunk_index += 1

        return input_tokens, token_idx_to_chunk_idx, chunk_idx_to_text, spacy_text, spacy_chunks

    def compress_attention(self, attention, token2idword_mapping):
        new_index = []
        prev = -1
        for idx, row in enumerate(attention):
            if token2idword_mapping[idx] != prev:
                new_index.append([row])
                prev = token2idword_mapping[idx]
            else:
                new_index[-1].append(row)
        
        new_attention = []
        for row in new_index:
            new_attention.append(np.mean(row, axis=0))

        new_attention = np.array(new_attention)
        attention = np.array(new_attention).T
        prev = -1
        new_index = []
        for idx, row in enumerate(attention):
            if token2idword_mapping[idx] != prev:
                new_index.append([row])
                prev = token2idword_mapping[idx]
            else:
                new_index[-1].append(row)
        new_attention = []
        for idx, row in enumerate(new_index):
            new_attention.append(np.mean(row, axis=0))
        new_attention = np.array(new_attention)

        for idx, row in enumerate(new_attention):
            new_attention[idx] = row / row.sum()
        return new_attention
    
    
    def BFS(self, s, end, graph):
        visited = [False] * (max(graph.keys()) + 100)
        queue = []
        queue.append((s, [(s, 0)]))
        found_paths = []
        visited[s] = True
        while queue:
            s, path = queue.pop(0)
            for i, conf in graph[s]:
                if i == end:
                    found_paths.append(path + [(i, conf)])
                    break
                if visited[i] == False:
                    queue.append([i, path + [(i, conf)]])
                    visited[i] = True
            
        candidate_facts = []
        for path_pairs in found_paths:
            if len(path_pairs) < 3:
                continue
            path = []
            cum_conf = 0
            for (node, conf) in path_pairs:
                path.append(node)
                cum_conf += conf
            
            candidate_facts.append((path, cum_conf))

        candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)
        return candidate_facts
    
    def filter_relation_sets(self, relation_pair):
        relation_pair, id2token = relation_pair
        triplet_idx, conf = relation_pair
        head, tail = triplet_idx[0], triplet_idx[-1]
        if head in id2token and tail in id2token:
            head = id2token[head]
            tail = id2token[tail]
            relations = [self.nlp(id2token[idx])[0].lemma_ for idx in triplet_idx[1:-1] if idx in id2token]
            relations = list(filter(lambda x: x not in [';', ','], relations))
            relations = list(filter(lambda x: x in self.valid_relations, relations))

            if len(relations) > 0 :
                return {
                    'head': head,
                    'relations': relations,
                    'tail': tail,
                    'confidence': conf
                }
        return {}
    
