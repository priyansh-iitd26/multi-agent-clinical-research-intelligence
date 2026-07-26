# fetches research papers from PubMed that reference a specific clinical trial ID
# This is our second data source.
# What it does:
# Takes a NCT ID (e.g. NCT09664) (unique ID for every trial)
# searches PubMed for papers that reference that particular trial
# fetch the full abstract and metadata for each paper
# returns the raw paper records - no cleaning happens here

# why do we even need PubMed in the first place ?
# clinicaltrials.gov tells us what a study promised to measure
# PubMed tells us what researchers actually published in a partiular study
# gap between these two things is where signals live