import numpy as np

LABELS = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']


def divide(x, y):
    return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float), where=y != 0)


def compute_p_r_f(preds, labels):
    TP = ((preds == labels) & (preds != 0)).astype(int).sum()
    P_total = (preds != 0).astype(int).sum()
    L_total = (labels != 0).astype(int).sum()
    P  = divide(TP, P_total).mean()
    R  = divide(TP, L_total).mean()
    F1 = divide(2 * P * R, (P + R)).mean()
    return P, R, F1


def eval_hoc(true_list, pred_list, id_list):
    data = {}

    assert len(true_list) == len(pred_list) == len(id_list), \
        f'Gold line no {len(true_list)} vs Prediction line no {len(pred_list)} vs Id line no {len(id_list)}'

    cat = len(LABELS)
    assert cat == len(true_list[0]) == len(pred_list[0])

    for i in range(len(true_list)):
        id = id_list[i]
        key = id.split('_')[0]
        if key not in data:
            data[key] = (set(), set())

        for j in range(cat):
            if true_list[i][j] == 1:
                data[key][0].add(j)
            if pred_list[i][j] == 1:
                data[key][1].add(j)

    print (f"There are {len(data)} documents in the data set")
    # print ('data', data)

    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(LABELS)
        for i in true:
            t[i] = 1

        p = [0] * len(LABELS)
        for i in pred:
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    p, r, f1 = compute_p_r_f(y_pred, y_test)
    return {"precision": p, "recall": r, "F1": f1}
