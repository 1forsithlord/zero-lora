import torch
import torch.nn.functional as F


def zscl_ce(S_student, S_teacher, orientation="teacher_target"):
    """ZSCL-style similarity-distribution distillation (arXiv 2303.06628 Sec. 3.2, Eq. 3-4), both directions.

    teacher_target (default, standard soft-target CE):  CE = - sum_j p_teacher_j * log p_student_j,
        averaged over rows (image->text) plus the same over columns (text->image).
    student_target (the literal reading of the extracted paper text, - sum_j p_j log p_bar_j with p = student):
        provided ONLY for comparison; it is a different objective (it is minimized by sharpening, not matching).
        [VERIFY against the official ZSCL code before choosing; the paper's text direction term has no explicit
         equation, so the column term here is an assumption.]
    Value is NOT zero when student == teacher (it equals the teacher entropy); see teacher_entropy().
    Its gradient w.r.t. the student equals that of the row/col KL in gnsp_cd (CE = KL + H(teacher), H constant).
    """
    S_t = S_teacher.detach()

    def one_dir(Ss, St):
        if orientation == "teacher_target":
            return -(F.softmax(St, dim=1) * F.log_softmax(Ss, dim=1)).sum(dim=1).mean()
        if orientation == "student_target":
            return -(F.softmax(Ss, dim=1) * F.log_softmax(St, dim=1)).sum(dim=1).mean()
        raise ValueError(orientation)

    return one_dir(S_student, S_t) + one_dir(S_student.T, S_t.T)


def teacher_entropy(S_teacher):
    """Row + column entropy of the teacher distributions = the value zscl_ce takes at student == teacher."""
    S = S_teacher.detach()

    def H(X):
        lp = F.log_softmax(X, dim=1)
        return -(lp.exp() * lp).sum(dim=1).mean()

    return H(S) + H(S.T)