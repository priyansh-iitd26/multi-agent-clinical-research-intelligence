# saves and loads the documents to and from the Google Cloud Storage (bucket)
# this is a permanent storage layer for all the raw and parsed docs
# saves the raw API responses to GCS bucket as JSON
# saves parsed study and paper records to GCS as JSON
# loads the docs back from GCS when the agent(s) need them
# lists available docs by prefix - useful for batch processing

# why we save the raw data first ?
# if parser has a bug, the raw originals are safe in GCS bucket
# we can re-parse them anytime without re-fetching from the API
# this is called raw zone - processed zone pattern typically used in data engineering

# folder structure expected in the GCS bucket
# raw/studies/NCT0089870.json ---> exactly what the API (ClinicalTrials.gov) returned
# raw/papers/9897948.json ---> exactly what PubMed returned
# processed/studies/NCT0089870.json ---> parsed study as JSON
# processed/papers/9897948.json ---> parsed paper as JSON