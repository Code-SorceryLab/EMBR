"""The EMBR evaluation harness: baselines, metrics, scenarios, attacks, and the runner.

Everything in here measures the system in `embr/`; nothing in `embr/` may import from
here. Scoring variants are weight maps over the same CompositeScorer, never copies.
"""
