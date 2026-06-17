"""Levenshtein düzenleme işlemleri (substitution / insertion / deletion).

Tüm metriklerin (WER, CER, cpWER) altında yatan TEK kelime/karakter hizalama
çekirdeği. Önce C-tabanlı hızlı motorları (rapidfuzz, jiwer) dener; yoksa devasa
girdilerde difflib (yaklaşık) ile kilitlenmeyi önler; orta boy girdilerde ise
O(M) bellekli KESİN DP'ye düşer.

Daha önce bu mantık üç ayrı dosyada (tests/evaluator, src/eval/metrics,
chime6_benchmark) kopyalanmış ve sapmıştı; artık tek kaynak burası.
"""

from __future__ import annotations


def levenshtein_sid(ref, hyp) -> tuple[int, int, int]:
    """İki dizi (kelime/karakter listesi) arasındaki (sub, ins, del) sayıları.

    Args:
        ref: Referans token listesi.
        hyp: Hipotez token listesi.

    Returns:
        (substitutions, insertions, deletions)
    """
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return 0, m, 0  # 0 sub, m ins, 0 del
    if m == 0:
        return 0, 0, n  # 0 sub, 0 ins, n del

    # Deneme 1: rapidfuzz (C++ tabanlı, devasa metinlerde milisaniyede çözer)
    try:
        from rapidfuzz.distance import Levenshtein
        ops = Levenshtein.editops(ref, hyp)
        subs = sum(1 for op in ops if op.tag == "replace")
        ins = sum(1 for op in ops if op.tag == "insert")
        dels = sum(1 for op in ops if op.tag == "delete")
        return subs, ins, dels
    except ImportError:
        pass

    # Deneme 2: jiwer (yüksek optimize C-tabanlı motor, genelde kuruludur)
    try:
        import jiwer
        ref_s, hyp_s = " ".join(ref), " ".join(hyp)
        if hasattr(jiwer, "process_words"):          # jiwer >= 3.0
            out = jiwer.process_words(ref_s, hyp_s)
            return out.substitutions, out.insertions, out.deletions
        m = jiwer.compute_measures(ref_s, hyp_s)     # jiwer < 3.0
        return m["substitutions"], m["insertions"], m["deletions"]
    except Exception:
        pass

    # Deneme 3: Arama uzayı devasa ve C-tabanlı motor yoksa, süreci saatlerce
    # kilitlememek için yaklaşık ama hızlı difflib'e (Gestalt, ~%99) düşeriz.
    if n * m > 5_000_000:
        import difflib
        matcher = difflib.SequenceMatcher(None, ref, hyp)
        subs = ins = dels = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                n_ref = i2 - i1
                n_hyp = j2 - j1
                subs += min(n_ref, n_hyp)
                if n_ref > n_hyp:
                    dels += (n_ref - n_hyp)
                elif n_hyp > n_ref:
                    ins += (n_hyp - n_ref)
            elif tag == "insert":
                ins += (j2 - j1)
            elif tag == "delete":
                dels += (i2 - i1)
        return subs, ins, dels

    # Deneme 4: O(M) bellekli KESİN Levenshtein DP (S/I/D takipli).
    # difflib'in aksine gerçek düzenleme mesafesi verir.
    previous_row = [(j, 0, j, 0) for j in range(m + 1)]  # (cost, subs, ins, dels)
    current_row = [(0, 0, 0, 0)] * (m + 1)
    for i in range(1, n + 1):
        current_row[0] = (i, 0, 0, i)
        ref_tok = ref[i - 1]
        for j in range(1, m + 1):
            hyp_tok = hyp[j - 1]
            if ref_tok == hyp_tok:
                current_row[j] = previous_row[j - 1]
            else:
                sub = previous_row[j - 1]
                ins_c = current_row[j - 1]
                dl = previous_row[j]

                c_sub = sub[0] + 1
                c_ins = ins_c[0] + 1
                c_dl = dl[0] + 1

                if c_ins < c_dl:
                    if c_ins < c_sub:
                        current_row[j] = (c_ins, ins_c[1], ins_c[2] + 1, ins_c[3])
                    else:
                        current_row[j] = (c_sub, sub[1] + 1, sub[2], sub[3])
                else:
                    if c_dl < c_sub:
                        current_row[j] = (c_dl, dl[1], dl[2], dl[3] + 1)
                    else:
                        current_row[j] = (c_sub, sub[1] + 1, sub[2], sub[3])
        previous_row, current_row = current_row, previous_row
    _, subs, ins, dels = previous_row[m]
    return subs, ins, dels
