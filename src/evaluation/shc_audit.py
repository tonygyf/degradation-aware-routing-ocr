"""Population reweighting of stratified operational-flag human audits (Supplementary Section 6)."""


def reweighted_flag_audit(
    positive_population, negative_population, positive_review, negative_review=None
):
    def count(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Audit counts must be nonnegative integers")
        return value

    positive_population, negative_population = (
        count(positive_population),
        count(negative_population),
    )
    total = positive_population + negative_population
    tp, fp = count(positive_review["true_positive"]), count(positive_review["false_positive"])
    if not total or not 0 < tp + fp <= positive_population:
        raise ValueError("Positive review must sample the auto-positive population")
    precision = tp / (tp + fp)
    tp_hat = precision * positive_population
    fn_hat = None
    if negative_review is not None:
        fn, tn = count(negative_review["false_negative"]), count(negative_review["true_negative"])
        if not 0 < fn + tn <= negative_population:
            raise ValueError("Negative review must sample the auto-negative population")
        fn_hat = fn / (fn + tn) * negative_population
    elif negative_population == 0:
        fn_hat = 0.0
    return {
        "population": total,
        "auto_positive_population": positive_population,
        "auto_negative_population": negative_population,
        "flag_precision": precision,
        "estimated_true_positive_count": tp_hat,
        "estimated_false_negative_count": fn_hat,
        "population_reweighted_recall": tp_hat / (tp_hat + fn_hat)
        if fn_hat is not None and tp_hat + fn_hat > 0
        else None,
        "precision_corrected_flag_rate": tp_hat / total,
        "audit_adjusted_true_rate": (tp_hat + fn_hat) / total if fn_hat is not None else None,
        "note": (
            "Stratified population reweighting, not unweighted pooled audit accuracy. "
            "Without a negative audit, recall is not inferred."
        ),
    }
