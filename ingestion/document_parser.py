# transforms the raw API responses into clean, structured internal responses
# takes raw study dicts from PubMedClient
# extracts only those fields which are required by the agent
# normalize the inconsistent data - missing fields, null values
# returns clean pydantic models ready for storage and agent use