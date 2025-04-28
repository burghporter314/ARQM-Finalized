from transformers import AutoTokenizer, BertForSequenceClassification, BitsAndBytesConfig
import torch
label_columns = ["result_binary_ambiguity", "result_binary_feasibility", "result_binary_singularity",
                 "result_binary_verifiability"]

import shap
import re
from scipy.ndimage import gaussian_filter1d

from reportlab.lib import colors

label_columns = ["result_binary_ambiguity", "result_binary_feasibility", "result_binary_singularity",
                 "result_binary_verifiability"]

LABEL_MAPPING = {
    "result_binary_singularity": "Singularity",
    "result_binary_feasibility": "Feasibility",
    "result_binary_ambiguity": "Ambiguity",
    "result_binary_verifiability": "Verifiability"
}

class BertWrapper:

    def __init__(self, violation_threshold=0.5, max_n_grams=5, display_ngram_summary=True, explainer_max_evals=500, model_path='app/models/requirement_quality/bert-requirement-classifier'):

        self.violation_threshold=violation_threshold
        self.max_n_grams=max_n_grams
        self.model_path = model_path
        self.display_ngram_summary = display_ngram_summary


        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.float16
        )

        self.model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.explainer = shap.Explainer(self.__predict_function__, self.tokenizer, batch_size=32, max_evals=explainer_max_evals, algorithm="partition", n_jobs=-1)

        self.explainer_max_evals = explainer_max_evals


    def __predict_function__(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=32)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.sigmoid(logits).cpu().numpy()


    def __apply_gaussian_smoothing__(self, impacts, sigma=1.2):
        return gaussian_filter1d(impacts, sigma=sigma)

    def __get_gradient_color__(self, weighted_val, max_val):
        if max_val == 0:
            return colors.white

        # Normalize between 0 (no impact) and 1 (strong negative impact)
        norm_val = abs(weighted_val) / max_val
        norm_val = max(0, min(1, norm_val))  # Clamp to [0, 1]

        # White → Red gradient
        r = 255
        g = int(255 * (1 - norm_val))
        b = int(255 * (1 - norm_val))

        # Optional: pastel soften
        pastel_factor = 0.4
        r = int(pastel_factor * 255 + (1 - pastel_factor) * r)
        g = int(pastel_factor * 255 + (1 - pastel_factor) * g)
        b = int(pastel_factor * 255 + (1 - pastel_factor) * b)

        return colors.Color(r / 255, g / 255, b / 255)

    def __weight_ngrams_for_visualization__(self, ngram_summary, label_name):
        weighted_ngrams = []
        is_ambiguity = label_name.lower() == "ambiguity"

        for n, ngram, score in ngram_summary:

            weight = n
            adjusted_score = score * weight

            if is_ambiguity:
                adjusted_score *= -1

            weighted_ngrams.append((n, ngram, adjusted_score))

        return weighted_ngrams


    def __get_top_ngrams__(self, words, shap_scores, n=2, top_k=3):
        """Extracts top n-grams with highest SHAP importance."""
        if len(words) < n:
            return []  # Skip if text is too short for n-grams
        ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        ngram_scores = [sum(shap_scores[i:i + n]) for i in range(len(shap_scores) - n + 1)]
        return sorted(zip(ngrams, ngram_scores), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    def __get_top_ngrams_by_size__(self, ngram_summary, label_name, top_n=5):
        is_ambiguity = label_name.lower() == "ambiguity"
        ngram_groups = {}

        for n, ngram, score in ngram_summary:
            if n not in ngram_groups:
                ngram_groups[n] = []
            ngram_groups[n].append((ngram, score))

        for n in ngram_groups:
            sorted_ngrams = sorted(
                ngram_groups[n],
                key=lambda x: x[1],
                reverse=is_ambiguity
            )
            ngram_groups[n] = sorted_ngrams[:top_n]

        return ngram_groups

    def __visualize_explanations_to_pdf__(self, visualization_data, filename="ARQM_Report.pdf", filter_labels=None,
                                          show_average=True):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from collections import defaultdict, Counter
        from datetime import datetime
        import statistics

        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        x_start, y_start = 50, height - 50
        max_line_width = width - 100
        line_height = 18
        min_y_threshold = 60
        indent_offset = 20
        sub_indent_offset = 40

        req_to_impacts = defaultdict(list)
        req_to_weights = defaultdict(list)
        req_to_words = {}
        visualization_by_req = defaultdict(list)
        label_counter = Counter()
        severity_scores = []

        for item in visualization_data:
            original_req_text, req_text, words, impacts, ngram_summary, req_index, label_key, certainty = item
            label_name = LABEL_MAPPING.get(label_key, label_key)
            is_ambiguity = label_name.lower() == "ambiguity"

            if filter_labels and label_name not in filter_labels:
                continue
            if is_ambiguity:
                certainty = 100 - certainty
            if certainty > 50:
                continue

            label_counter[label_name] += 1

            top_ngrams_by_size = self.__get_top_ngrams_by_size__(ngram_summary, label_name, top_n=3)
            weighted_ngrams = self.__weight_ngrams_for_visualization__(ngram_summary, label_name)

            word_impact_map = defaultdict(float)
            for n, ngram, score in weighted_ngrams:
                for token in ngram.split():
                    word_impact_map[token] += score

            weighted_impacts = [word_impact_map[word.strip()] for word in words]
            smoothed_impacts = self.__apply_gaussian_smoothing__(weighted_impacts, sigma=1.2)
            abs_max = max(abs(min(smoothed_impacts)), abs(max(smoothed_impacts))) or 1.0
            normalized_impacts = [v / abs_max for v in smoothed_impacts]
            if is_ambiguity:
                normalized_impacts = [-v for v in normalized_impacts]

            severity = statistics.mean(abs(v) for v in normalized_impacts) * (certainty / 100.0)
            severity_scores.append(severity)

            req_to_impacts[req_index].append(normalized_impacts)
            req_to_weights[req_index].append((100 - certainty) / 100.0)
            req_to_words[req_index] = words
            visualization_by_req[req_index].append(
                (original_req_text, req_text, label_name, words, normalized_impacts, top_ngrams_by_size, certainty))

        x, y = x_start, y_start
        c.setFont("Helvetica-Bold", 18)
        c.drawString(x, y, "ARQM Quality Report")
        y -= 25
        c.setFont("Helvetica", 14)
        c.drawString(x, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 20
        c.setFont("Helvetica", 12)
        for label, count in label_counter.items():
            c.drawString(x, y, f"{label}: {count} violations")
            y -= 18
        y -= 10
        if severity_scores:
            avg_severity = sum(severity_scores) / len(severity_scores)
            overall_score = avg_severity * 100
            c.drawString(x, y, f"Overall Quality Score: {overall_score:.2f}%")
        else:
            c.drawString(x, y, f"Overall Quality Score: 100.00% (No violations found)")
        y -= 40

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, "Legend (Impact Severity)")
        y -= 18

        legend_x = 50
        legend_y = y - 10
        legend_width = 150
        legend_height = 15
        legend_steps = 100

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)

        max_val = 1
        for i in range(legend_steps):
            norm_val = i / (legend_steps - 1)
            weighted_val = norm_val * max_val
            color = self.__get_gradient_color__(weighted_val, max_val)

            bar_x = legend_x + (i * legend_width / legend_steps)
            c.setFillColor(color)
            c.rect(bar_x, legend_y, legend_width / legend_steps, legend_height, fill=1, stroke=0)

        # Add labels under the bar
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.black)
        c.drawString(legend_x - 2, legend_y - 10, "None")
        c.drawRightString(legend_x + legend_width + 2, legend_y - 10, "High")
        c.drawCentredString(legend_x + legend_width / 2, legend_y - 10, "Moderate")
        y = legend_y - 30

        c.showPage()

        x, y = x_start, y_start
        for req_index in sorted(visualization_by_req.keys()):
            words = req_to_words[req_index]
            impacts_list = req_to_impacts[req_index]
            weights = req_to_weights[req_index]
            violations = visualization_by_req[req_index]

            if y < min_y_threshold + line_height * 4:
                c.showPage()
                x, y = x_start, y_start

            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_start, y,f"Req {req_index + 1} ({len(violations)} {'violation' if len(violations) == 1 else 'violations'})")
            y -= 20

            if show_average and impacts_list:
                total_weight = sum(weights) or 1.0
                weighted_avg = [
                    sum(val * weight for val, weight in zip(vals, weights)) / total_weight
                    for vals in zip(*impacts_list)
                ]
                if y < min_y_threshold + line_height * 4:
                    c.showPage()
                    x, y = x_start, y_start

                c.setFont("Helvetica-Bold", 11)
                c.drawString(x_start + indent_offset, y, f"Violation: Average Impact")
                y -= 20
                c.setFont("Helvetica", 11)
                x = x_start + sub_indent_offset

                original_words = violations[0][0].split()
                req_words = violations[0][1].split()

                cut_words = original_words[:len(original_words) - len(req_words)]
                word_text = (" ".join(cut_words)) + " "

                word_width = c.stringWidth(word_text, "Helvetica", 11)
                if x + word_width > width - indent_offset:
                    x = x_start + sub_indent_offset
                    y -= line_height
                if y < min_y_threshold:
                    c.showPage()
                    x, y = x_start, y_start

                color = colors.white
                c.setFillColor(color)
                c.rect(x - 1, y - 3, word_width + 2, line_height - 4, fill=1, stroke=0)
                c.setFillColor(colors.black)
                c.drawString(x, y, word_text)
                x += word_width

                for idx, (word, val) in enumerate(zip(words, weighted_avg)):
                    word_text = word + " "
                    word_width = c.stringWidth(word_text, "Helvetica", 11)
                    if x + word_width > width - indent_offset:
                        x = x_start + sub_indent_offset
                        y -= line_height
                    if y < min_y_threshold:
                        c.showPage()
                        x, y = x_start, y_start

                    color = self.__get_gradient_color__(val, 1)
                    c.setFillColor(color)
                    c.rect(x - 1, y - 3, word_width + 2, line_height - 4, fill=1, stroke=0)
                    c.setFillColor(colors.black)
                    c.drawString(x, y, word)
                    x += word_width
                y -= 25

            for idx, (original_req_text, req_text, label_name, words, normalized_impacts, top_ngrams_by_size, certainty) in enumerate(violations):
                if y < min_y_threshold + line_height * 4:
                    c.showPage()
                    x, y = x_start, y_start


                x = x_start + indent_offset
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x, y, f"Violation: {label_name}    Certainty: {100 - certainty:.1f}%")
                y -= 20
                c.setFont("Helvetica", 11)
                x = x_start + sub_indent_offset


                original_words = original_req_text.split()
                req_words = req_text.split()

                cut_words = original_words[:len(original_words) - len(req_words)]
                word_text = (" ".join(cut_words)) + " "

                word_width = c.stringWidth(word_text, "Helvetica", 11)
                if x + word_width > width - indent_offset:
                    x = x_start + sub_indent_offset
                    y -= line_height
                if y < min_y_threshold:
                    c.showPage()
                    x, y = x_start, y_start

                color = colors.white
                c.setFillColor(color)
                c.rect(x - 1, y - 3, word_width + 2, line_height - 4, fill=1, stroke=0)
                c.setFillColor(colors.black)
                c.drawString(x, y, word_text)
                x += word_width


                for idx2, (word, val) in enumerate(zip(words, normalized_impacts)):
                    word_text = word + " "
                    word_width = c.stringWidth(word_text, "Helvetica", 11)
                    if x + word_width > width - indent_offset:
                        x = x_start + sub_indent_offset
                        y -= line_height
                    if y < min_y_threshold:
                        c.showPage()
                        x, y = x_start, y_start

                    color = self.__get_gradient_color__(val, 1)

                    c.setFillColor(color)
                    c.rect(x - 1, y - 3, word_width + 2, line_height - 4, fill=1, stroke=0)
                    c.setFillColor(colors.black)
                    c.drawString(x, y, word)
                    x += word_width
                y -= 15

                c.setFont("Helvetica-Oblique", 10)
                if self.display_ngram_summary:
                    for n, ngrams in top_ngrams_by_size.items():
                        y -= 10
                        c.drawString(x_start + sub_indent_offset, y, f"Top {n}-grams:")
                        y -= 15
                        for ngram, score in ngrams:
                            if y < min_y_threshold:
                                c.showPage()
                                x, y = x_start, y_start
                            c.drawString(x_start + sub_indent_offset, y, f"'{ngram}' - SHAP Impact: {abs(score):.4f}")
                            y -= 15
                        y -= 10
                y -= 20

        c.save()

    def predict_requirement_v2(self, texts, word_offset_percent=0.2):

        truncated_texts = []
        for text in texts:
            words = text.split()
            cutoff = max(1, int(len(words) * word_offset_percent))
            truncated_text = " ".join(words[cutoff:])
            truncated_texts.append(truncated_text)

        inputs = self.tokenizer(
            truncated_texts,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()

        explain_indices = []
        for i in range(len(truncated_texts)):
            p = probs[i]
            if p[0] > self.violation_threshold or any(p[j] < 1 - self.violation_threshold for j in [1, 2, 3]):
                explain_indices.append(i)

        shap_values = self.explainer([truncated_texts[i] for i in explain_indices]) if explain_indices else None

        visualization_data = []

        for i in range(len(truncated_texts)):
            if i in explain_indices:
                sv_idx = explain_indices.index(i)
                words = shap_values.data[sv_idx]

                for j, label in enumerate(label_columns):
                    p = probs[i][j]
                    if (j == 0 and p > self.violation_threshold) or (j in [1, 2, 3] and p < 1 - self.violation_threshold):
                        shap_scores = shap_values.values[sv_idx][:, j]
                        impacts = [0.0] * len(words)
                        ngram_summary = []

                        words = list(map(str, shap_values.data[sv_idx]))
                        for n in range(1, self.max_n_grams + 1):
                            top_ngrams = self.__get_top_ngrams__(words, shap_scores, n=n)
                            for ngram, score in top_ngrams:
                                ngram_summary.append((n, ngram, score))
                                ngram_tokens = ngram.split()
                                for k in range(len(words) - len(ngram_tokens) + 1):
                                    if words[k:k + len(ngram_tokens)] == ngram_tokens:
                                        for t in range(len(ngram_tokens)):
                                            impacts[k + t] += score

                        # Add to visualization list
                        certainty = probs[i][j] * 100
                        visualization_data.append((texts[i], truncated_texts[i], words, impacts, ngram_summary, i, label, certainty))

        if visualization_data:
            self.__visualize_explanations_to_pdf__(visualization_data)