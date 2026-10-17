import torch.nn.functional as F


def gnsp_cd(S_student, S_teacher):
    """GNSP 'contrastive distillation' (arXiv 2507.19839, Eq. 14-16): row-wise KL + column-wise KL between
    teacher and student distributions of tau-scaled batch similarity matrices:

        L = KL(P_teacher_row || P_student_row) + KL(P_teacher_col || P_student_col),  P = softmax(S)

    S already includes the temperature (S = logit_scale * cos = cos / tau_GNSP with tau_GNSP = 0.01).
    KL direction (teacher || student) follows the Phase 2 request; the fetched excerpt does not state it.
    [VERIFY against GNSP Eq. 14-16 / official code.]  batchmean = mean over rows / columns.
    """
    St = S_teacher.detach()
    row = F.kl_div(F.log_softmax(S_student, dim=1), F.softmax(St, dim=1), reduction="batchmean")
    col = F.kl_div(F.log_softmax(S_student.T, dim=1), F.softmax(St.T, dim=1), reduction="batchmean")
    return row + col