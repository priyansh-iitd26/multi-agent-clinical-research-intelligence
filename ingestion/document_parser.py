# cleans up the raw, messy data that comes back from
# ClinicalTrials.gov and PubMed APIs
# the APIs return deeply nested, inconsistent JSON
# this file extracts only the fields we actually need and puts them into
# clean, typed Python objects
# everything AFTER this file - chunking, embedding, agents,etc. 
# only ever sees the clean version 
# they never have to deal with the API's confusing nested structure
# this matters because if ClinicalTrials.gov changes their API
# tomorrow, we only need to fix THIS ONE FILE 
# nothing else in the entire system needs to change

