# run the entire ingestion pipeline
# run this file to fetch studies, fetch papers, parse everything
# save it to the Google Cloud Storage

# fetches studies from clinicaltrials.gov
# save raw study data to GCS bucket
# parse the raw studies to clean, structured records
# save parsed study records to GCS bucket

# for each research study, fetches related research papers from PubMed
# saves both raw and parsed papers to GCS bucket