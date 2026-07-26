# it is entry point to all the data
# fetches all the clinical study records from clinicaltrials.gov
# connect to the clinicaltrials.gov public API - No API key
# search for the studies by condition, intervention or sponsor
# fetches the full study details for each result
# handles pagination - returns 100 results per page
# handles rate limiting and retries automatically
# async - concurrency method