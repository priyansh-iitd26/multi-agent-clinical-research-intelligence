# single source file for all the configuration 
# without centralized settings, developers use os.getenv() is scattered across many files
# for example, if the environment variable name changes

# pydantic settings
# all environment variables are defined in one place - settings.py
# if my OPENAI_API_KEY is missing from .env, I will get an error immediately when the app starts
# NOT after 1st LLM call is invoked - fail first technique

from pydantic_settings import BaseSettings, SettingsConfigDict
# BaseSettings: it knows how to read values from environment variables and .env file
# pydantic BaseModel reads from the Python dict()
# BaseSettinsg is going to read from environment
# SettingsConfigDict: how we configure the base settings behavior

from pydantic import Field
# Field: adds metadata to each setting

class Settings(BaseSettings):
    """
    all configs defined at one place only
    case insensitive: openai_api_key = OPENAI_API_KEY
    """
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
        extra = "ignore",
    )

    openai_api_key : str = Field(
        ...,
        description = "OpenAI API Key for LLM and Embeddings"
    )

    openai_embedding_model : str = Field(
        default = "text-embedding-3-large",
        description = "OpenAI Model used to generate the vector embeddings"
    )

    # agents would be using GPT-4o
    openai_chat_model : str = Field(
        default = "gpt-4o",
        description = "OpenAI model used for Agent Reasoning"
    )

    langsmith_api_key : str = Field(
        ...,
        description = "Langsmith API Key for Agent Tracing"
    )

    langsmith_project : str = Field(
        default = "clinical_trial_intelligence",
        description = "Langsmith project name"
    )

    langsmith_tracing_v2 : bool = Field(
        default = True,
        description = "Enable Langsmith for all the agent runs"
    )

    gcp_project_id : str = Field(
        ...,
        description = "GCP Project ID"
    )

    gcp_region : str = Field(
        default = "us-central1",
        description = "GCP region for all the cloud resources"
    )

    gcs_bucket_name : str = Field(
        ...,
        description = "Google Cloud Storage bucket name"
    )

    db_host : str = Field(
        ...,
        description = "Cloud SQL Host IP (local) or Socket Path (Cloud Run)"
    )

    db_port : int = Field(
        default = 5432, # standard PostgreSQL port
        description = "PostgreSQL port"
    )

    db_name : str = Field(
        default = "clinical_trial_db",
        description = "PostgreSQL database name"
    )

    db_user : str = Field(
        ...,
        description = "PostgreSQL database user"
    )

    db_password : str = Field(
        ...,
        description = "PostgreSQL database password"
    )

    clinical_trials_base_url : str = Field(
        default = "https://clinicaltrials.gov/api/v2",
        description = "ClinicalTrials.gov API V2 Base URL"
    )

    clinical_trials_page_size : int = Field(
        default = 100, # how many studies requested per API call - setting to 100 for rate limits
        description = "Number of studies to fetch per API page"
    )

    pubmed_base_url : str = Field(
        default = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description = "PubMed eutils API Base URL"
    )

    api_host : str = Field(
        default = "0.0.0.0",
        description = "FastAPI host address"
    )

    api_port : int = Field(
        default = 8000,
        description = "FastAPI port"
    )

    api_env : str = Field(
        default = "development", # can be 'development' or 'production'
        description = "Environment name: development or production"
    )

    @property
    def database_url(self) -> str :
        """
        build the full async postgreSQL connection string from parts
        we use asyncpg as the async postgreSQL driver
        asyncpg is used to handle concurrency
        asyncpg requires the connection string to start with:
        postgresql+asyncpg://

        returns :
        str : full connection url ready for asyncpg.create_pool()
        example : postgresql+asyncpg://mosaic_user:password@35.232.74.203:5432/mosaic
        """
        return (
            f"postgresql+asyncpg://"
            f"{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}"
            f"/{self.db_name}"
        )
    
    @property
    def is_production(self) -> bool :
        """
        returns True if the app is running in production
        """
        return self.api_env.lower() == "production"

