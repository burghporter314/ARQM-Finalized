import fitz
import nltk
import spacy
from nltk import sent_tokenize
from spacy.matcher import Matcher
from app.startup.model_loader import generative_identification_model, generative_quality_model

nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)

# === PATTERN 1: Actor + Modal + Verb ===
pattern_modal_active = [
    {"POS": "DET", "OP": "?"},
    {"LOWER": {"IN": ["system", "user", "application", "software"]}},
    {"LOWER": {"IN": ["shall", "must", "should", "will", "can", "may"]}},
    {"POS": "VERB", "OP": "+"}
]

# # === PATTERN 2: Actor + Modal + Passive (be + VERB) ===
# pattern_modal_passive = [
#     {"POS": "DET", "OP": "?"},
#     {"LOWER": {"IN": ["system", "user", "application", "software"]}},
#     {"LOWER": {"IN": ["shall", "must", "should", "will", "can", "may"]}},
#     {"LOWER": "be"},
#     {"POS": "VERB"}
# ]
#
# # === PATTERN 3: Imperative verb + noun ===
# pattern_imperative = [
#     {"POS": "VERB"},
#     {"POS": "DET", "OP": "?"},
#     {"POS": "NOUN", "OP": "+"}
# ]

# Add patterns to matcher
matcher.add("REQUIREMENT_MODAL_ACTIVE", [pattern_modal_active])
# matcher.add("REQUIREMENT_MODAL_PASSIVE", [pattern_modal_passive])
# matcher.add("REQUIREMENT_IMPERATIVE", [pattern_imperative])

class RequirementQualityAnalyzer:

    def __init__(self, content = None, file=None):
        """
        Initializes a RequirementIdentifier object

        Parameters:
        - content (str): The content to find requirements within
        - file: The file to extract content and find requirements within
        """

        self.content = content
        self.file = file

        if(file):
            file_bytes = file.read()
            document = fitz.open(stream=file_bytes, filetype="pdf")

        else:
            document = fitz.open(stream=content, filetype="pdf")

        text = ""

        for page in document:
            text += page.get_text()

        self.content = text

    def _get_sentences_from_content(self):
        # doc = nlp(self.content)
        # matches = matcher(doc)
        #
        # matched_sentences = set()
        # for _, start, end in matches:
        #     matched_sentences.add(doc[start].sent.text.strip())
        #
        # return list(matched_sentences)
        return sent_tokenize(self.content)

    def get_requirement_quality(self, auto_combine=False):
        """
        Gets all the requirements related to the instance content
        """

        sentences = self._get_sentences_from_content()

        requirements_to_analyze = []
        for sentence in sentences:

            doc = nlp(sentence)
            matches = sorted(matcher(doc), key=lambda x: x[1])

            chunks = []
            for i in range(len(matches) - 1):
                start_i = matches[i][1]
                start_next = matches[i + 1][1]
                span = doc[start_i:start_next].text.strip()
                chunks.append(span)

            if matches:
                start_last = matches[-1][1]
                span = doc[start_last:].text.strip()
                chunks.append(span)

            if len(chunks) > 0:
                for sub_sentence in chunks:
                    if(len(sub_sentence.strip().split()) > 3):
                        requirements_to_analyze.append(sub_sentence.replace("\n", ""))
            else:
                if(len(sentence) > 40):
                    requirements_to_analyze.append(sentence.replace("\n", ""))

        requirement_flags = generative_identification_model.getIfRequirementFewShot(requirements_to_analyze)
        original_req_indices = [i for i, flag in enumerate(requirement_flags) if flag]
        requirement_array = [requirements_to_analyze[i] for i in original_req_indices]

        result = generative_quality_model.getOverallQuality(requirement_array)

        generative_quality_model.getExcelResult(requirement_flags, requirements_to_analyze)

        return {}