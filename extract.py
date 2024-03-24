from src.dataset import load_data
from src.triplet_extractor import TripletExtractor

def extract_triplets(data, max_sentences = 100):
    extracted_triplets = []
    triplet_extractor = TripletExtractor()
    for sentence in data[:max_sentences]:
        found_triplets = triplet_extractor.process_sentence(sentence)
        for triplet in found_triplets:
            print(f"{triplet['confidence']}\t{triplet['head']} -> {triplet['relations']} -> {triplet['tail']}")
        extracted_triplets += found_triplets
    return extracted_triplets

if __name__ == "__main__":
    data = load_data()

    extracted_triplets = extract_triplets(data)
