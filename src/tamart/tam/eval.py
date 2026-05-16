import os
import string
import unicodedata

import cv2
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.translate import meteor_score
from rouge import Rouge


if not os.path.exists(os.path.join(os.path.expanduser("~"), 'nltk_data/taggers/averaged_perceptron_tagger.zip')):
    nltk.download('averaged_perceptron_tagger')


_lemmatizer = WordNetLemmatizer()


def get_word_type(word):
    """Categorize a word as 'function', 'noun', or 'others' via NLTK POS tagging."""
    tagged_word = nltk.pos_tag([word])
    pos = tagged_word[0][1]

    if pos in ['CC', 'DT', 'EX', 'MD', 'POS', 'PRP', 'PRP$', 'UH', 'WDT', 'WP', 'WP$', 'WRB']:
        return 'function'
    elif pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        return 'noun'
    else:
        return 'others'


def is_english_punctuation(char):
    return char in string.punctuation


def is_chinese_char_or_punctuation(char):
    for ch in char:
        if 'CJK' in unicodedata.name(ch, ''):
            return True
    return False


def ids_to_word_groups(ids, processor):
    """Decode token ids into grouped words and record the corresponding token indices."""
    txt = processor.batch_decode(ids)[0]
    tokens = processor.tokenizer.tokenize(txt)
    words, tokens_idx = [], []
    for i, _ in enumerate(tokens):
        word = processor.tokenizer.decode(processor.tokenizer.convert_tokens_to_ids(_))
        if i == 0 or is_english_punctuation(word) or is_chinese_char_or_punctuation(word) or word[0] == ' ' or _[0] == '▁':
            words.append(word.replace(' ', ''))
            tokens_idx.append([i])
        else:
            words[-1] += word.replace(' ', '')
            tokens_idx[-1].append(i)
    return words, tokens_idx


def single_words_match(word1, word2):
    a = _lemmatizer.lemmatize(word1.lower().replace('-', ''))
    b = _lemmatizer.lemmatize(word2.lower().replace('-', ''))
    return a == b


def words_match(category_word, target_word):
    """Check if any individual word in category_word matches target_word."""
    tks = category_word.split()
    for tk in tks:
        if single_words_match(tk, target_word):
            return True
    return False


def evaluate(maps, tokens, processor, caption, mask, category):
    """
    Evaluate plausibility (IoU between predicted maps and ground truth masks)
    and NLG metrics for the generated caption.

    Returns a list of [obj_iou, func_iou, rougel, meteor, pre, rec].
    """
    words, tokens_id = ids_to_word_groups(tokens, processor)
    if tokens_id[-1][-1] != (len(maps) - 1):
        return [[], [], [], [], [], []]
    words_label = []
    for word in words:
        word_type = get_word_type(word)
        if word_type == 'noun':
            lb = -1
            for k, v in category.items():
                if words_match(k, word):
                    lb = v
            words_label.append(lb)
        elif word_type == 'function':
            words_label.append(-2)
        else:
            words_label.append(-3)

    if os.path.exists(mask):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros_like(maps[0])
    obj_iou, pre, rec, noun_fg_thresh = [], [], [], []
    for i in range(len(words)):
        if words_label[i] > 0:
            ious, pres, recs, thresh = [], [], [], []
            gt = (mask == words_label[i]).astype('uint8')
            for j in tokens_id[i]:
                map_ = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, pred = cv2.threshold(map_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if gt.sum() != 0:
                    tp = float((gt * pred > 0).sum())
                    ious.append(tp / ((gt + pred / 255) > 0).sum())
                    pres.append(tp / (pred > 0).sum())
                    recs.append(tp / (gt > 0).sum())
                thresh.append(t)

            noun_fg_thresh.append(max(thresh))
            if len(ious) > 0:
                m_iou = max(ious)
                obj_iou.append(max(ious))
                pre.append(pres[ious.index(m_iou)])
                rec.append(recs[ious.index(m_iou)])

            if len(obj_iou) > 1 and words_label[i] > 0 and words_label[i - 1] == words_label[i]:
                select_idx = -1 if obj_iou[-1] > obj_iou[-2] else -2
                obj_iou[-2] = obj_iou[select_idx]
                obj_iou = obj_iou[:-1]
                pre[-2] = pre[select_idx]
                pre = pre[:-1]
                rec[-2] = rec[select_idx]
                rec = rec[:-1]

        elif words_label[i] == -1:
            thresh = []
            for j in tokens_id[i]:
                map_ = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, pred = cv2.threshold(map_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh.append(t)
            noun_fg_thresh.append(max(thresh))

    func_iou = []
    if len(noun_fg_thresh) > 0:
        fg_thresh = sum(noun_fg_thresh) / len(noun_fg_thresh)
        for i in range(len(words)):
            if words_label[i] == -2:
                neg_iou = []
                for j in tokens_id[i]:
                    neg_iou.append(float((maps[j] < fg_thresh).sum()) / maps[j].size)
                func_iou.append(sum(neg_iou) / len(neg_iou))

    output_text = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ref = [str(_).lower().split() for _ in caption]
    hypo = str(output_text[0]).lower().split()
    meteor = [meteor_score.meteor_score(references=ref, hypothesis=hypo)]
    r = Rouge()
    rougel = [max([r.get_scores(output_text[0], _)[0]['rouge-l']['f'] for _ in caption])]

    return [obj_iou, func_iou, rougel, meteor, pre, rec]
