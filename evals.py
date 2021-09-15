import numpy as np
import torch


def evaluate_ts(gold_ts, pred_ts, length):
    
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)
    n_ts_hits, n_gold_ts, n_pred_ts = 0., 0., 0.
    golden, pred = [], []

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        length_i = int(length[i])
        g_ts = g_ts[:length_i]
        p_ts = p_ts[:length_i]
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts, ts_tag_vocab={"O":0, "B":1, "I":2}), tag2ts(ts_tag_sequence=p_ts, ts_tag_vocab={"O":0, "B":1, "I":2})
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        
        assert len(g_ts_sequence) == g_ts.count(1)

        n_ts_hits   += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count

        golden.append(g_ts_sequence)
        pred.append(p_ts_sequence)

    return n_ts_hits, n_gold_ts, n_pred_ts, golden, pred 

# this function takes the predicted sequence and output the corresponding list of two-tuples.
# such as the input is ['B', 'I', 'B', 'O', 'O', 'B'].
# then the output will be [(0,1), (2,2), (5,5)].
# each element in the output is the begin index and end index of an extracted aspect.
def tag2ts(ts_tag_sequence, ts_tag_vocab):

    ts_tag_vocab_ = {ts_tag_vocab[key]:key for key in ts_tag_vocab.keys()}

    n_tags = len(ts_tag_sequence)
    ts_sequence = []
    begin, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        ts_tag = ts_tag_vocab_[ts_tag]
        
        if ts_tag == 'B':
            if begin != -1:
                end = i -1
                if end >= begin > -1:
                    ts_sequence.append((begin, end))
                    begin, end = -1, -1
                else:
                    begin, end = -1, -1

            begin = i
            end = i
        elif ts_tag == 'O':
            end = i-1
            if end >= begin > -1:
                ts_sequence.append((begin, end))
                begin, end = -1, -1
            else:
                begin, end = -1, -1
        elif ts_tag == 'I':
            end = i

    if end >= begin > -1:
        ts_sequence.append((begin, end))

    return ts_sequence

# this function compare the predicted tuples with the ground truth tuples
def match_ts(gold_ts_sequence, pred_ts_sequence):
    hit_count, gold_count, pred_count = 0., 0., 0.
    for t in gold_ts_sequence:
        gold_count += 1
    for t in pred_ts_sequence:
        if t in gold_ts_sequence:
            hit_count += 1
        pred_count += 1

    return hit_count, gold_count, pred_count # number of hits, number of ground truth, number of extracted

def evaluate_program(trainer, batches, args):
    eval_opn_loss, eval_step  = 0., 0
    labels_opn_n, logits_opn_n, hits_opn_n = 0., 0., 0.
    golden_opn, pred_opn = [], []
    for batch in batches:
        loss, pred = trainer.predict(batch)
        eval_opn_loss += loss
        eval_step += 1

        length = batch[7].sum(dim=1).tolist()
        hits_n_t, labels_n_t, logits_n_t, _, _ = evaluate_ts(batch[-1].tolist(), pred, length) # hits_n_t: number of hits, number of ground truth, number of extracted in this batch
        labels_opn_n += labels_n_t
        logits_opn_n += logits_n_t
        hits_opn_n += hits_n_t

    # opn F1
    prec   = hits_opn_n / (logits_opn_n + 1e-10)
    recall = hits_opn_n / (labels_opn_n + 1e-10)
    F1_opn = 2 * prec * recall / (prec + recall + 1e-10)

    return [prec, recall, F1_opn], eval_opn_loss 

def evaluate_program_case_study(trainer, batches, args):
    eval_opn_loss, eval_step  = 0., 0
    labels_opn_n, logits_opn_n, hits_opn_n = 0., 0., 0.
    golden_opn, pred_opn = [], []
    sample_id, golden, pred_span = [], [], []
    for batch in batches:
        loss, pred = trainer.predict(batch)
        eval_opn_loss += loss
        eval_step += 1

        length = batch[7].sum(dim=1).tolist()
        hits_n_t, labels_n_t, logits_n_t, golden_t, pred_t = evaluate_ts(batch[-1].tolist(), pred, length)
        labels_opn_n += labels_n_t
        logits_opn_n += logits_n_t
        hits_opn_n += hits_n_t

        sample_id.extend(batch[0])
        golden.extend(golden_t)
        pred_span.extend(pred_t)

    # opn F1
    prec   = hits_opn_n / (logits_opn_n + 1e-10)
    recall = hits_opn_n / (labels_opn_n + 1e-10)
    F1_opn = 2 * prec * recall / (prec + recall + 1e-10)

    return [prec, recall, F1_opn], [sample_id, golden, pred_span]

