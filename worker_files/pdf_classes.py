"""
OOP Base and Specialized PDF Report Generators for FAD Analysis
Provides clean, reusable report abstractions using FPDF2 for all experimental,
background, characterization, and calibration reports.
Author: Ben Auer
"""
from pathlib import Path
from datetime import date
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF, XPos, YPos, enums


def safe_str(text):
    """
    Sanitizes strings for FPDF latin-1 encoding, replacing common Unicode
    symbols with ASCII or Latin-1 equivalents.
    """
    if text is None:
        return ""
    replacements = {
        "•": "-",
        "°": " deg ",
        "±": "+/-",
        "—": "-",
        "–": "-",
        "µ": "u",
        "μ": "u",
        "Ω": " Ohm",
        "π": "pi",
        "²": "2",
        "³": "3",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"'
    }
    s = str(text)
    for key, val in replacements.items():
        s = s.replace(key, val)
    return s.encode("latin-1", "ignore").decode("latin-1")


class BasePDF(FPDF):
    """
    Core FPDF wrapper providing consistent headers, footers, margins,
    and metadata across all lab reports.
    """
    def __init__(self, title="Report", author="Auer", report_date=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = title
        self.report_author = author
        self.report_date = report_date or date.today().strftime("%m. %d. %Y.")
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, self.report_title, align='C')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()} | {self.report_author} | {self.report_date}', align='R', border=0)


class BaseReport:
    """
    Base Report Controller that encapsulates PDF layout generation,
    table formatting, figure embedding, and file exports.
    """
    def __init__(self, title="Lab Report", author="Auer", report_date=None, target_dir=None):
        self.title = title
        self.author = author
        self.report_date = report_date or date.today().strftime("%m. %d. %Y.")
        self.target_dir = Path(target_dir) if target_dir else Path.cwd()
        
        self.plots_dir = self.target_dir / "plots"
        self.data_dir = self.target_dir / "data"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.pdf = BasePDF(title=self.title, author=self.author, report_date=self.report_date)

    def add_page(self):
        """Adds a new page to the document."""
        self.pdf.add_page()

    def add_title_block(self, title_text, subtitle=None):
        """Renders a section title with optional subtitle."""
        self.pdf.set_font('Times', 'B', 14)
        self.pdf.cell(0, 8, safe_str(title_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if subtitle:
            self.pdf.set_font('Times', 'I', 9.5)
            self.pdf.cell(0, 5, safe_str(subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(2)

    def add_key_value_pairs(self, kv_dict, font_size=11, label_bold=True):
        """
        Renders a series of aligned key-value metadata rows.
        """
        for key, val in kv_dict.items():
            k_str = f"{key}: "
            v_str = f"{val}"
            
            self.pdf.set_font('Times', 'B' if label_bold else '', font_size)
            k_width = self.pdf.get_string_width(k_str) + 2
            self.pdf.cell(k_width, 6.5, safe_str(k_str), new_x=XPos.RIGHT, new_y=YPos.LAST)
            
            self.pdf.set_font('Times', '', font_size)
            self.pdf.cell(0, 6.5, safe_str(v_str), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def add_text_section(self, label, content, line_height=6):
        """Renders notes, analysis, or narrative paragraphs."""
        self.pdf.set_font('Times', 'B', 12)
        self.pdf.write(line_height, safe_str(f"{label}: "))
        self.pdf.set_font('Times', '', 12)
        self.pdf.write(line_height, safe_str(f"{content}\n\n"))

    def add_plot_figure(self, fig, filename, title=None, width=180, ln_after=105, close_fig=True, dpi=300):
        """
        Saves a matplotlib figure to plots_dir and embeds it into the PDF.
        """
        plot_path = self.plots_dir / filename
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        if close_fig:
            plt.close(fig)

        if title:
            self.pdf.set_font("Times", "B", 12)
            self.pdf.cell(0, 8, safe_str(title), new_x=XPos.RIGHT, new_y=YPos.NEXT)

        self.pdf.image(str(plot_path), enums.Align.C, y=self.pdf.get_y(), w=width)
        if ln_after > 0:
            self.pdf.ln(ln_after)
        return plot_path

    def add_plot_image(self, image_path, title=None, width=180, ln_after=105):
        """Embeds an existing image file from disk."""
        if title:
            self.pdf.set_font("Times", "B", 12)
            self.pdf.cell(0, 8, safe_str(title), new_x=XPos.RIGHT, new_y=YPos.NEXT)

        self.pdf.image(str(image_path), enums.Align.C, y=self.pdf.get_y(), w=width)
        if ln_after > 0:
            self.pdf.ln(ln_after)

    def add_table(self, df, col_widths, headers=None, alignments=None, row_height=6.0, font_size=8.5, max_y=265):
        """
        Renders a pandas DataFrame as a structured, bordered table with automatic
        pagination and header repetition.
        """
        if df is None or df.empty:
            return

        headers = headers or list(df.columns)
        alignments = alignments or ['C'] * len(headers)

        def print_header():
            self.pdf.set_font('Times', 'B', font_size)
            for w, h, a in zip(col_widths, headers, alignments):
                self.pdf.cell(w, row_height, safe_str(str(h)), border=1, align=a)
            self.pdf.ln()

        print_header()
        self.pdf.set_font('Times', '', font_size)

        for _, row in df.iterrows():
            if self.pdf.get_y() > max_y:
                self.pdf.add_page()
                print_header()
                self.pdf.set_font('Times', '', font_size)

            for i, val in enumerate(row):
                w = col_widths[i]
                a = alignments[i] if i < len(alignments) else 'C'
                val_str = f"{val:.4f}" if isinstance(val, float) and abs(val) < 1 else (f"{val:.2f}" if isinstance(val, float) else str(val))
                self.pdf.cell(w, row_height, safe_str(val_str), border=1, align=a)
            self.pdf.ln()
        self.pdf.ln(3)

    def add_metrics_card(self, title, metrics_dict, model_name=None, font_size=10):
        """
        Renders a structured key-value parameter box (e.g. for circuit fit metrics).
        """
        if title:
            self.pdf.set_font('Times', 'B', font_size + 1)
            self.pdf.cell(0, 6, safe_str(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if model_name:
            self.pdf.set_font('Times', 'I', font_size - 0.5)
            self.pdf.cell(0, 5, safe_str(f"Model: {model_name}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.pdf.set_font('Times', '', font_size)
        for k, v in metrics_dict.items():
            self.pdf.cell(0, 5, safe_str(f"{k}: {v}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(3)

    def save(self, output_filename=None):
        """Exports the generated PDF report to disk."""
        if output_filename:
            output_path = self.target_dir / output_filename
        else:
            output_path = self.target_dir / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', self.title.lower())}.pdf"

        self.pdf.output(str(output_path))
        print(f"Report successfully saved at: {output_path}")
        return output_path


# ==============================================================================
# SPECIALIZED REPORT SUBCLASSES
# ==============================================================================

class ReactionReport(BaseReport):
    """Specialized report generator for Formic Acid Decomposition Reaction Trials."""
    def __init__(self, target_dir, trial_info=None, gc_method_info=None):
        super().__init__(
            title="Formic Acid Decomposition Trial Report",
            author="Auer",
            target_dir=target_dir
        )
        self.trial_info = trial_info or {}
        self.gc_method_info = gc_method_info or {}

    def build_metadata_page(self, notes="", analysis="", pre_rxn_cap=0.0, rate_fig=None):
        """Builds Page 1 with trial metadata, rate plot, and notes/analysis."""
        self.add_page()
        
        kv = {
            "Device": self.trial_info.get("Device", "N/A"),
            "Date": self.trial_info.get("Date", "N/A"),
            "Reactor Temperature": f"{self.trial_info.get('T reactor (deg C)', 'N/A')} deg C",
            "% FA in vapor stream": f"{self.trial_info.get('Percent (%) FA in vapor stream', 'N/A')}%",
            "Pre-rxn Capacitance (CV, N2)": f"{pre_rxn_cap:.2f} uF"
        }
        self.add_key_value_pairs(kv)

        if rate_fig is not None:
            self.add_plot_figure(rate_fig, filename="rate_data.png", title="Reaction Kinetics Plot", width=180, ln_after=105)

        if notes:
            self.add_text_section("Reaction Notes", notes)
        if analysis:
            self.add_text_section("Analysis", analysis)


class CharacterizationReport(BaseReport):
    """Specialized report generator for Electrochemical Characterizations (EIS & CV)."""
    def __init__(self, target_dir, folder_title=""):
        display_title = f"Electrochemical Characterization Report - {folder_title}" if folder_title else "Electrochemical Characterization Report"
        super().__init__(
            title=display_title,
            author="Auer",
            target_dir=target_dir
        )
        self.folder_title = folder_title


class BackgroundReport(BaseReport):
    """Specialized report generator for Formic Acid Decomposition Background Testing."""
    def __init__(self, target_dir):
        super().__init__(
            title="Formic Acid Decomposition Background Report",
            author="Auer",
            target_dir=target_dir
        )


class CalibrationReport(BaseReport):
    """Specialized report generator for GC & Sensor Calibration Reports."""
    def __init__(self, target_dir):
        super().__init__(
            title="Chromatography & Sensor Calibration Report",
            author="Auer",
            target_dir=target_dir
        )


class RTDReport(BaseReport):
    """Specialized report generator for Residence Time Distribution (RTD) Reports."""
    def __init__(self, target_dir):
        super().__init__(
            title="Residence Time Distribution (RTD) Report",
            author="Auer",
            target_dir=target_dir
        )

