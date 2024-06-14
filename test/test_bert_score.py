from bert_score import score

def calculate_bert_score(ref_sentence, hyp_sentence):
    # Calculate BERTScore
    P, R, F1 = score([hyp_sentence], [ref_sentence], lang='en', rescale_with_baseline=True)
    bert_score = F1.item()
    return bert_score

# Example usage
ref_sentence = "This is a reference sentence."
hyp_sentence = "This is a hypothesis sentence."
score = calculate_bert_score(ref_sentence, hyp_sentence)
print("BERTScore:", score)
