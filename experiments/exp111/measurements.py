"""Quantify snnsim--Brian2 discrepancies for exp111 comparisons."""

from __future__ import annotations


def evaluate(comparisons):
    results = []
    for comparison in comparisons:
        measured = {key: value for key, value in comparison.items() if key != "condition"}
        rows = []
        for row in comparison["series"]:
            snnsim = float(row["snnsim"])
            brian2 = float(row["brian2"])
            difference = brian2 - snnsim
            scale = max(abs(snnsim), abs(brian2))
            rows.append(
                {
                    **row,
                    "brian2_minus_snnsim": difference,
                    "absolute_difference": abs(difference),
                    "relative_absolute_difference": (
                        abs(difference) / scale if scale else 0.0
                    ),
                }
            )
        results.append(
            {
                **measured,
                "series": rows,
                "maximum_absolute_difference": max(
                    row["absolute_difference"] for row in rows
                ),
            }
        )
    return {
        "schema": "exp111.analysis/v2",
        "tests": results,
        "summary": {"total": len(results)},
    }
