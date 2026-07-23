# -------------------------------------------------------
# Helpers — export, history, sample data, quality metadata
# -------------------------------------------------------

import pandas as pd
from datetime import datetime


# -------------------------------------------------------
# Sample wine descriptions for the preset buttons
# -------------------------------------------------------

EXAMPLE_DESCRIPTIONS = {
    "high": (
        "This is an extraordinary wine with remarkable depth and complexity. "
        "Layers of dark fruit, including blackberry, cassis, and plum, are "
        "complemented by nuances of cedar, vanilla, and dark chocolate from "
        "expert oak aging. The tannins are polished and velvety, supporting "
        "a full-bodied structure that leads to an exceptionally long, "
        "elegant finish. Superb balance and outstanding craftsmanship."
    ),
    "medium": (
        "A pleasant and approachable wine with fresh fruit aromas of cherry "
        "and raspberry. The palate offers moderate acidity with soft tannins "
        "and a clean finish. Notes of vanilla and light spice add interest. "
        "Well-made and enjoyable, this wine pairs nicely with everyday meals."
    ),
    "low": (
        "A simple wine with muted aromas and a thin palate. Light fruit "
        "flavors are overpowered by noticeable acidity and a somewhat harsh "
        "finish. The wine lacks complexity and depth, with a short, "
        "unremarkable aftertaste. Basic and straightforward."
    ),
}


# -------------------------------------------------------
# Quality class metadata (no emojis — clean labels)
# -------------------------------------------------------

QUALITY_META = {
    "High": {
        "color": "#22c55e",
        "css_class": "high",
        "label": "High Quality",
        "insight": (
            "The model identifies this description as indicative of a "
            "<b>premium wine</b> - the language suggests complex flavour "
            "structure, refined tannins, and an extended finish."
        ),
    },
    "Medium": {
        "color": "#eab308",
        "css_class": "medium",
        "label": "Medium Quality",
        "insight": (
            "The model classifies this description as <b>good quality</b> - "
            "the text conveys approachable characteristics with balanced "
            "acidity and a pleasant, if straightforward, profile."
        ),
    },
    "Low": {
        "color": "#ef4444",
        "css_class": "low",
        "label": "Low Quality",
        "insight": (
            "The model rates this description as **basic quality** - "
            "simpler vocabulary and references to thin body or harsh "
            "notes are typical markers for this class."
        ),
    },
}


def get_quality_meta(prediction: str) -> dict:
    """Return metadata dict for a predicted quality class."""
    return QUALITY_META.get(prediction, QUALITY_META["Medium"])


# -------------------------------------------------------
# Prediction history
# -------------------------------------------------------

def add_to_history(history: list, text: str, prediction: str,
                   confidence: float) -> None:
    """Append a prediction record to the session history list."""
    history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "input": (text[:72] + "...") if len(text) > 72 else text,
        "prediction": prediction,
        "confidence": f"{confidence:.1%}",
    })


def history_to_dataframe(history: list) -> pd.DataFrame:
    """Convert history list to a display-ready DataFrame."""
    if not history:
        return pd.DataFrame(
            columns=["Time", "Input", "Prediction", "Confidence"]
        )
    df = pd.DataFrame(history)
    df.columns = ["Time", "Input", "Prediction", "Confidence"]
    df.index = range(1, len(df) + 1)
    df.index.name = "#"
    return df


# -------------------------------------------------------
# CSV export
# -------------------------------------------------------

def export_csv(history: list) -> bytes:
    """Return prediction history as CSV bytes."""
    return history_to_dataframe(history).to_csv(index=True).encode("utf-8")


# -------------------------------------------------------
# PDF report
# -------------------------------------------------------

def generate_pdf(prediction_data: dict, steps: dict = None) -> bytes:
    """Generate a single-page PDF report for the latest prediction."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "Wine Quality - Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # Divider
    pdf.set_draw_color(139, 92, 246)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Prediction
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Prediction Result", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Predicted Class:", 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, str(prediction_data.get("prediction", "N/A")), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(55, 7, "Confidence:", 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f'{prediction_data.get("confidence", 0):.1%}', ln=True)
    pdf.ln(4)

    # Probabilities
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, "Class Probabilities", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for cls, prob in prediction_data.get("probabilities", {}).items():
        pdf.cell(55, 7, f"{cls}:", 0)
        pdf.cell(0, 7, f"{prob:.1%}", ln=True)

    # NLP section
    if steps:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, "NLP Processing Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        raw = steps.get("raw_text", "")[:120]
        pdf.cell(0, 6, f"Input: {raw}{'...' if len(steps.get('raw_text',''))>120 else ''}", ln=True)
        pdf.cell(0, 6, f"Processed: {steps.get('final_text', '')[:120]}", ln=True)
        pdf.cell(0, 6, f"Tokens: {steps.get('token_count', 0)} | Unique: {steps.get('unique_token_count', 0)}", ln=True)

    # Footer
    pdf.ln(12)
    pdf.set_draw_color(139, 92, 246)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 6, "Wine Quality AI  |  NLP + TF-IDF + Decision Tree", ln=True, align="C")

    return bytes(pdf.output())
