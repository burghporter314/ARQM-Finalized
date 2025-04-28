import fitz
import nltk
from nltk.tokenize import sent_tokenize

from app.startup.model_loader import generative_identification_model, generative_quality_model

# Download required dictionary for tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

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
        return sent_tokenize(self.content)

    def get_requirement_quality(self, auto_combine=False):
        """
        Gets all the requirements related to the instance content
        """

        assoc_content_sentences = self._get_sentences_from_content()

        # TODO break sentences apart further by new line characters
        sentences = self._get_sentences_from_content()

        requirements_to_analyze = []
        for sentence in sentences:
            # if(sentence.count('\n') > 3):
            #     for sub_sentence in sentence.split("\n"):
            #         if(len(sub_sentence) > 20):
            #             requirements_to_analyze.append(sub_sentence.replace("\n", ""))
            #
            # else:
            if(len(sentence) > 20):
                requirements_to_analyze.append(sentence.replace("\n", ""))

        requirement_indexes = generative_identification_model.getIfRequirementFewShot(requirements_to_analyze)

        result = [req for req, flag in zip(requirements_to_analyze, requirement_indexes) if flag]

        result = generative_quality_model.getOverallQuality(result)

        filtered_results = {}

        return filtered_results