import torch


def raw_frobenius(S_student, S_teacher):
    """Roadmap Sec.1:  L_geom = (1/(N*M)) * || S_theta - S_ref ||_F^2,  S = logit_scale * cos.

    No normalization of S (no softmax): sensitive to absolute similarity values and their per-row offsets.
    Teacher is detached here as a safeguard (callers should also compute it under no_grad).
    Scale note: S has magnitude ~ logit_scale (=100), so the loss scales ~ logit_scale^2; choose lambda on a
    log grid on DEV, do not reuse the lambda of the softmax losses.
    """
    S_t = S_teacher.detach()
    return ((S_student - S_t) ** 2).sum() / S_student.numel()