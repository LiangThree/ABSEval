import transformers
import torch


def load_vera():
    tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera')
    model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera')
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
    model.eval()
    t = model.shared.weight[32097, 0].item()  # temperature for calibration

    return model, tokenizer, linear, t


def calculate_confidence(model, tokenizer, linear, t, statement):
    input_ids = tokenizer.batch_encode_plus([statement], return_tensors='pt', padding='longest',
                                            truncation='longest_first', max_length=128).input_ids
    with torch.no_grad():
        output = model(input_ids)
        last_hidden_state = output.last_hidden_state
        hidden = last_hidden_state[0, -1, :]
        logit = linear(hidden).squeeze(-1)
        logit_calibrated = logit / t
        score_calibrated = logit_calibrated.sigmoid()
        score_calibrated = float(score_calibrated)
        print(score_calibrated)

        return score_calibrated


if __name__ == '__main__':
    statement = 'Apple is fruit.'
    model, tokenizer, linear, t = load_vera()
    confidence_score = calculate_confidence(model, tokenizer, linear, t, statement)
    print(confidence_score)
