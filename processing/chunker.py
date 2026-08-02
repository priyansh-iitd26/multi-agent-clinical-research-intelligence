# this file takes a long study document and breaks it into smaller,
# overlapping pieces of text called chunks
# each chunk is then sent to OpenAI in the next step (embedder.py)
# to get its vector embedding

# why do we need to chunk at all ?
# OpenAI's embedding model has a token limit — it cannot process
# an entire study document in one shot 
# a single clinical trial record can easily be 2000-5000 words long 
# we must break it into smaller pieces first

# but there is a much deeper reason too
# if we embed the WHOLE document as one giant chunk, 
# the embedding becomes a blurry average of everything in it 
# when an agent searches for "sponsor never posted results", 
# a whole-document embedding might miss that signal because
# it is diluted by all the other content

# NOTE: smaller, focused chunks -> sharper, more precise embeddings
# -> agents find exactly what they are looking for

# what is TextChunk ?
# it is a small python dataclass — a lightweight container
# that holds one chunk of text plus metadata about it:
# which study it came from, which position it is in the
# document, and what type of content it contains

from dataclasses import dataclass
# dataclass is a python decorator that automatically generates
# common methods like __init__ and __repr__ for a class
# instead of writing a __init__ method ourselves
# we just decorate the class with @dataclass
# python handles all the boilerplate code automatically

from typing import Any
from ingestion.document_parser import ParsedStudy, ParsedPaper
# we import clean data models from document_parser.py
# intuitive: chunker always works with parsed data

from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration
# control knobs for chunking

CHUNK_SIZE = 500
# max number of WORDS per chunk
# we measure in words (not characters or tokens) because 
# words are easier to reason about
# "this chunk is about 500 words" is intuitive 
# characters and tokens are less human-friendly

OVERLAP_SIZE = 50
# tells how many words to REPEAT between consecutive chunks
# the last 50 words of chunk 1 become the first 50 words of chunk 2
# to ensure that no important sentence gets split across a boundary
# 50 words is roughly 2-3 sentences - enough context to preserve
# meaning at the edges without wasting too much space on repetition

# TextChunk dataclass
# **output format of the chunker
# **every chunk produced is a TextChunk object
# **embedder.py receives a list of TextChunks and embeds each one

@dataclass
class TextChunk:
    """
    one chunk of text from a study or paper, ready to be embedded

    fields:
        chunk_id: unique identifier for this specific chunk
                  format: NCT_ID_chunk_0, NCT_ID_chunk_1, etc.

        unique_id: which study/paper this chunk belongs to

        chunk_text: the actual text content of this chunk

        chunk_index: position of this chunk in the document
                     0 = first chunk, 1 = second chunk, etc.

        source:      where this chunk came from
                     "study" -> from a ClinicalTrials.gov record
                     "paper" -> from a PubMed research paper

        word_count:  number of words there are in this chunk
    """
    chunk_id : str
    unique_id : str
    chunk_text : str
    chunk_index : int
    source : str
    word_count : int

# chunker class
class Chunker:
    """
    splits clean study and paper documents into overlapping text chunks
    """
    # chunk one study
    def chunk_study(self, parsed_study: ParsedStudy) -> list[TextChunk]:
        """
        takes one parsed study from GCS bucket and returns a list of TextChunks
        these list of TextChunks are ready to be embedded

        first we BUILD the full text by combining all the study's
        important fields into one long string, with clear labels
        so the embedding model knows what each section means
        then we SPLIT that long string into overlapping chunks

        reason:
            clinical trials data is stored as structured fields, 
            but embedding models are trained on natural language 
            so instead of embedding each field independently, 
            we first combine all important fields into a single readable document with labels
            this gives the model the complete context - for example, it understands that a 
            particular intervention belongs to a specific disease and trial phase
            once the document is created, we split it into overlapping chunks 
            because embedding models have token limits
        """
        # step 1 : combining all the study's fields into one labelled text block

        full_text = self._build_full_study_text(parsed_study)

        # step 2 : split the full text block into overlapping chunks

        chunks = self._split_into_chunks(
            text = full_text,
            unique_id = parsed_study.nct_id,
            source = "study"
        )

        logger.info(
            f"Chunked study | "
            f"nct_id = {parsed_study.nct_id} | "
            f"chunks_produced = {len(chunks)}"
        )

        return chunks

    # chunk one paper
    def chunk_paper(self, parsed_paper: ParsedPaper) -> list[TextChunk]:
        """
        takes one parsed paper from GCS bucket and returns a list of TextChunks
        these list of TextChunks are ready to be embedded

        first we BUILD the full text by combining all the paper's
        important fields into one long string, with clear labels
        so the embedding model knows what each section means
        then we SPLIT that long string into overlapping chunks
        (same strategy as chunk_study)
        """
        full_text = self._build_full_paper_text(parsed_paper)

        chunks = self._split_into_chunks(
            text = full_text,
            unique_id = parsed_paper.pmid,
            source = "paper"
        )

        logger.info(
            f"Chunked paper | "
            f"pmid = {parsed_paper.pmid} | "
            f"chunks_produced = {len(chunks)}"
        )

        return chunks

    # chunk multiple studies
    def chunk_studies(self, studies: list[ParsedStudy]) -> list[TextChunk]:
        """
        chunks a whole list of studies in one single function call
        returns ALL chunks from every study in studies[] as one flat list.
        """
        all_chunks : list[TextChunk] = []

        for study in studies:
            chunks = self.chunk_study(study)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked all studies in input | "
            f"studies = {len(studies)} | "
            f"total_chunks = {len(all_chunks)}"
        )

        return all_chunks

    # chunk multiple papers
    def chunk_papers(self, papers: list[ParsedPaper]) -> list[TextChunk]:
        """
        chunks a whole list of papers in one single function call
        returns ALL chunks from every paper in papers[] as one flat list.
        """
        all_chunks : list[TextChunk] = []

        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked all papers in input | "
            f"papers = {len(papers)} | "
            f"total_chunks = {len(all_chunks)}"
        )

        return all_chunks

    # private method : build_full_study_text
    def _build_full_study_text(self, study: ParsedStudy) -> str:
        """
        combines all a study's fields into one labelled text block

        WHY label each field ?
        when OpenAI embeds this text, the labels help the model
        understand what it is reading 
        "SPONSOR: Novo Nordisk" is much more informative than just "Novo Nordisk"
        there with no context, labels make embeddings more precise

        WHY not just embed the json ?
        json has lots of noise - curly braces, quotes, etc.
        embedding models are trained on natural languages and
        perform much better on natural language
        plain labelled text is cleaner and produces better embeddings.

        args:
            study: the ParsedStudy to convert to text

        returns:
            one long string containing all the study's key fields
            each clearly labelled on its own line
        """
        # building sections as a list first
        # will join them at end
        # much cleaner than string concatenation with +=
        # and avoids creating many intermediate string objects in memory
        sections = []

        # title and identification
        # these two fields are most important identifiers
        sections.append(f"NCT ID: {study.nct_id}")
        sections.append(f"TITLE: {study.title}")

        # sponsor
        # important for our Track Record and Pattern Finder agents -
        # they reason about sponsors across many studies
        # labelling will clearly help
        sections.append(f"SPONSOR: {study.sponsor}")

        # phase and status
        # status (COMPLETING, RECRUITING, etc.) is what our
        # Missing Results agent cares about most -
        # COMPLETED + no results posted is the signal it looks for
        sections.append(f"PHASE: {study.phase}")
        sections.append(f"STATUS: {study.status}")

        # conditions and interventions
        if study.conditions:
            sections.append(f"CONDITIONS: {','.join(study.conditions)}")

        if study.interventions:
            sections.append(f"INTERVENTIONS: {','.join(study.interventions)}")

        # outcomes
        # this is what the Broken Promises agent compares against
        # the actual results - if what was promised here does not
        # match what was measured, that is outcome switching -> undesirable
        if study.primary_outcome:
            sections.append(f"PRIMARY_OUTCOME: {study.primary_outcome}")

        if study.secondary_outcomes:
            sections.append(f"SECONDARY_OUTCOMES: {';'.join(study.secondary_outcomes)}")
            # using ; to join as a secondary outcome can be a long sentence with commas too
        
        # dates
        if study.start_date:
            sections.append(f"START DATE: {study.start_date}")

        if study.completion_date:
            sections.append(f"COMPLETION DATE: {study.completion_date}")

        # results posted
        # this is a boolean in our data model (see lines 93 and 203 in document_parser.py)
        # but we convert it to plain english — "YES" or "NO" — because language
        # models understand natural language better than True/False
        # "RESULTS POSTED: NO" is a very strong signal for our
        # Missing Results agent to pick up on
        sections.append(
            f"RESULTS POSTED: {'YES' if study.results_posted else 'NO'}"
        )

        # enrollment
        if study.enrollment:
            sections.append(f"ENROLLMENT: {study.enrollment} participants")

        # amendments
        if study.protocol_amendments:
            # we just note the count of amendments made rather than
            # the full details 
            # too much amendment details would bloat the text and 
            # dilute the more important outcome signals
            sections.append(
                f"PROTOCOL AMENDMENTS: "
                f"{len(study.protocol_amendments)} amendment(s) filed"
            )

        return '\n'.join(sections)

    # private method : build_full_paper_text
    def _build_full_paper_text(self, paper: ParsedPaper) -> str:
        """
        combines all a paper's fields into one labelled text block
        similar in logic to build_full_study_text
        """
        sections = []

        sections.append(f"PMID: {paper.pmid}")
        sections.append(f"TITLE: {paper.title}")

        if paper.journal:
            sections.append(f"JOURNAL: {paper.journal}")

        if paper.pub_date:
            sections.append(f"PUBLICATION_DATE: {paper.pub_date}")

        if paper.authors:
            # we only include the first 5 authors to keep the text focused 
            # a paper with 15 authors does not need all 15 listed - 
            # the first 5 are enough to identify it
            sections.append(f"AUTHORS: {','.join(paper.authors[:5])}")

        if paper.nct_ids_referenced:
            # which clinical trials this paper discusses/references to
            # our Side Effect Checker agent uses this to link
            # papers back to their corresponding trial filings
            sections.append(
                f"CLINICAL TRIALS REFERENCED: "
                f"{', '.join(paper.nct_ids_referenced)}"
            )

        if paper.abstract:
            sections.append(f"ABSTRACT: {paper.abstract}")

        return '\n'.join(sections)

    # private method : split_into_chunks
    def _split_into_chunks(self, text : str, unique_id : str, source: str) -> list[TextChunk]:
        """
        the core splitting algorithm 
        splits one long text into overlapping chunks of CHUNK_SIZE words each

        working:
        1. splits the full text into individual words
        2. used a sliding window of CHUNK_SIZE words
        3. slide forwards by (CHUNK_SIZE - OVERLAP_SIZE) words each time
        4. this creates the overlap
        """
        # NOTE:
        # considered using LangChain's RecursiveCharacterTextSplitter and SemanticChunker
        # but the input isn't free-form text—it's structured clinical trial metadata
        # after serializing the fields into a labeled text doc, 
        # a deterministic sliding-window chunker is sufficient
        # it produces predictable chunk sizes, preserves context through overlap, 
        # has linear runtime complexity, avoids unnecessary dependencies, 
        # and is easy to test and reproduce
        # if we were working with unstructured documents like PDFs or long reports, 
        # we'd prefer RecursiveCharacterTextSplitter, and for documents with frequent topic shifts, 
        # we'd consider semantic chunking
        
        words = text.split()

        if not words:
            logger.warning(
                f"Empty text - no chunks produced | "
                f"unique_id = {unique_id} | source = {source}"
            )

        chunks : list[TextChunk] = []
        chunk_index = 0

        step = CHUNK_SIZE - OVERLAP_SIZE

        for start in range(0, len(words), step):

            end = start + CHUNK_SIZE
            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = "".join(chunk_words)

            chunk = TextChunk(
                chunk_id = f"{unique_id}_chunk_{chunk_index}",
                unique_id = unique_id,
                chunk_text = chunk_text,
                chunk_index = chunk_index,
                source = source,
                word_count = len(chunk_words)
            )
            # example: "NCT04788680_chunk_0", "NCT04788680_chunk_1",...

            chunks.append(chunk)
            chunk_index += 1

            logger.info(
                f"Split complete | "
                f"unique_id = {unique_id} | "
                f"total_words = {len(words)} | "
                f"chunks = {len(chunks)} | "
                f"chunk_size = {CHUNK_SIZE} | "
                f"overlap = {OVERLAP_SIZE}"
            )

        return chunks