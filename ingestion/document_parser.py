# cleans up the raw, messy data that comes back from
# ClinicalTrials.gov and PubMed APIs
# the APIs return deeply nested, complex and inconsistent JSON
# this file extracts only the fields we actually need and puts them into
# clean, typed Python objects
# everything AFTER this file - chunking, embedding, agents,etc. 
# only ever sees the clean version 
# they never have to deal with the API's confusing nested structure
# this matters because if ClinicalTrials.gov changes their API
# tomorrow, we only need to fix THIS ONE FILE 
# nothing else in the entire system needs to change

from datetime import datetime, UTC
from typing import Any
from pydantic import BaseModel, Field
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# internal data schemas
class ParsedStudy(BaseModel):

    nct_id : str
    title : str
    sponsor : str
    phase : str
    status : str
    conditions : list[str] = Field(default_factory=list)
    interventions : list[str] = Field(default_factory=list)
    primary_outcome : str
    secondary_outcomes : list[str] = Field(default_factory=list)
    start_date : str
    completion_date : str
    results_posted : bool
    enrollment : int = Field(ge=0)
    protocol_amendments : list[dict[str, Any]] = Field(default_factory=list)
    raw_data : dict[str, Any]
    parsed_at : datetime

class ParsedPaper(BaseModel):

    pmid : str
    title : str
    abstract : str
    journal : str
    pub_date : str
    authors : list[str] = Field(default_factory=list)
    nct_ids_referenced : list[str] = Field(default_factory=list)
    source : str = Field(default="pubmed")
    word_count : int = Field(ge=0)
    parsed_at : datetime

# parser class
# messy raw data -> clean ParsedStudy and ParsedPapers
# stateless -> doesn't remember anything between calls
# one document parser can be re-used

class DocumentParser:
    """
    converts raw API data into clean ParsedStudy and ParsedPaper objects
    usage:
        parser = DocumentParser()
        study = parser.parse_study(raw_study_dict)
        paper = parser.parse_paper(raw_paper_dict)
    """

    # parse one study
    def parse_study(self, raw_study_dict : dict[str, Any]) -> ParsedStudy | None:
        """
        cleans one clinicaltrials.gov study record
        args:
           one raw study dict, exactly as the API returned
        returns:
            a ParsedStudy object if everything worked
            None if something essential was missing or broken
        """
        try:
            # the API wraps almost everything inside "protocolSection"
            # opening each one we need and store it in a short variable
            # so the rest of this method stays readable

            protocol = raw_study_dict.get("protocolSection", {})

            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            design_module = protocol.get("designModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            interventions_mod = protocol.get("armsInterventionsModule", {})

            results_section = raw_study_dict.get("resultsSection", {})
            has_results = bool(results_section)

            # extract the nct_id first
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                logger.warning("study is missing its NCT ID — skipping...")
                return None

            # extract title
            title = (
                id_module.get("officialTitle")
                or id_module.get("briefTitle", "")
                or ""
            )
            # "or" chain tries each option in order until one of them is non-empty

            # extract sponsor
            sponsor = (
                sponsor_module
                .get("leadSponsor", {})
                .get("name", "Unknown Sponsor")
            )
            # chain of .get() calls - each one safely returns an empty
            # dict {} if the key is missing, so the NEXT .get() never
            # crashes trying to call .get() on something that is None

            phase = design_module.get("phases", ["N/A"])
            phase = phase[0] if phase else "N/A"
            # "phases" comes back as a list
            # sometimes a study lists two phases together like ["PHASE1", "PHASE2"]
            # we take the first one as our single phase value

            # extract status, conditions and interventions
            status = status_module.get("overallStatus", "UNKNOWN")
            conditions = conditions_module.get("conditions", [])
            interventions = [
                i.get("name", "")
                for i in interventions_mod.get("interventions", [])
                if i.get("name")
            ]

            # extract primary outcomes
            primary_outcomes_list = outcomes_module.get("primaryOutcomes", [])
            primary_outcome = (
                primary_outcomes_list[0].get("measure", "")
                if primary_outcomes_list
                else ""
            )
            # a study can technically list more than one primary outcome,
            # but in practice the first one is the main one

            # extract secondary outcomes
            secondary_outcomes = [
                i.get("measure", "")
                for i in outcomes_module.get("secondaryOutcomes", [])
                if i.get("measure")
            ]

            # extract the start and completion dates
            start_date = (
                status_module
                .get("startDateStruct", {})
                .get("date", "")
            )
            
            completion_date = (
                status_module
                .get("primaryCompletionDateStruct", {})
                .get("date", "")
                or status_module
                .get("completionDateStruct", {})
                .get("date", "")
            )

            # extract enrollment number
            enrollment_info = design_module.get("enrollmentInfo", {})
            num_enrollments = enrollment_info.get("count", 0) # number of enrollments
            try:
                num_enrollments = int(num_enrollments)
            except (ValueError, TypeError):
                num_enrollments = 0

            # extract protocol amendments
            annotations = raw_study_dict.get("annotationSection", {})
            amendment_module = annotations.get("annotationModule", {})
            amendments = amendment_module.get("unpostedAnnotation", {})

            protocol_amendments = []

            if amendments:
                protocol_amendments = [
                    {
                        "date" :        amendments.get("unpostedResponsibleParty", ""),
                        "description" : str(amendments)
                    }
                ]
            # amendment data from the API is structured inconsistently

            return ParsedStudy(
                nct_id=nct_id,
                title=title,
                sponsor=sponsor,
                phase=phase,
                status=status,
                conditions=conditions,
                interventions=interventions,
                primary_outcome=primary_outcome,
                secondary_outcomes=secondary_outcomes,
                start_date=start_date,
                completion_date=completion_date,
                results_posted=has_results,
                enrollment=num_enrollments,
                protocol_amendments=protocol_amendments,
                raw_data=raw_study_dict,
                parsed_at=datetime.now(UTC)
            )

        except Exception as e:
            # trying to get the NCT ID for the error log even though
            # parsing failed, so we know WHICH study had the problem
            nct_id = raw_study_dict.get("protocolSection", {}).get(
                "identificationModule", {}
            ).get("nctId", "UNKNOWN")

            logger.error(
                f"Failed to parse study | nct_id={nct_id} | error={e}"
            )
            return None

    # parse multiple studies
    def parse_studies(self, raw_studies : list[dict[str, Any]]) -> list[ParsedStudy]:
        """
        parses the whole list of raw studies in one go
        basically loops using the function call parse_study()
        any study that fails to parse is skipped - not fatal
        """
        parsed = []
        num_failed = 0

        for raw_study in raw_studies:
            study = self.parse_study(raw_study)

            if(study):
                parsed.append(study)
            else:
                num_failed += 1

        logger.info(
            f"Parsed studies | "
            f"success={len(parsed)} | "
            f"failed={num_failed} | "
            f"total={len(raw_studies)}"
        )

        return parsed

    # parse one paper
    def parse_paper(self, raw: dict[str, Any]) -> ParsedPaper | None:
        """
        cleans one raw PubMed paper record
        pubMed papers are simpler than studies - the pubmed_client.py
        file already flattened the XML into a reasonably clean dict
        this method does the final cleanup and builds the typed and
        validated object
        """
        try:
            abstract = raw.get("abstract", "")
            word_count = len(abstract.split()) if abstract else 0

            return ParsedPaper(
                pmid=raw.get("pmid", ""),
                title=raw.get("title", ""),
                abstract=abstract,
                journal=raw.get("journal", ""),
                pub_date=raw.get("pub_date", ""),
                authors=raw.get("authors", []),
                nct_ids_referenced=raw.get("nct_ids_referenced", []),
                source="pubmed",
                word_count=word_count,
                parsed_at=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(
                f"Failed to parse paper | "
                f"pmid={raw.get('pmid', 'UNKNOWN')} | "
                f"error={e}"
            )
            return None

    # parse multiple papers
    def parse_papers(self, raw_papers: list[dict[str, Any]]) -> list[ParsedPaper]:

        parsed = []
        num_failed = 0

        for raw in raw_papers:
            paper = self.parse_paper(raw)
            if paper:
                parsed.append(paper)
            else:
                num_failed += 1

        logger.info(
            f"Parsed papers | "
            f"success = {len(parsed)} | "
            f"failed = {num_failed} | "
            f"total = {len(raw_papers)}"
        )

        return parsed