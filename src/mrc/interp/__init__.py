"""Interpretability: which features the fitted hazard model actually uses.

Three attribution views live in `importance`: permutation importance (shuffle one
column, keep its marginal), LightGBM SHAP contributions (additive within a tree),
and feature-family ablation (zero a family, refit, score the curve). Only the
validation block is ever scored.
"""
