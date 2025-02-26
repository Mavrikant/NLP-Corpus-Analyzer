#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLP Corpus Analyzer - A tool for analyzing text corpora with NLP techniques.
This module provides tools for text analysis including unigram and bigram probability calculations
with optional smoothing features.
"""
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import nltk
from nltk import bigrams
from nltk.lm.preprocessing import pad_both_ends

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_SUPPORTED = True
except ImportError:
    DRAG_DROP_SUPPORTED = False


class TreeviewWithScroll(ttk.Frame):
    """
    Custom Treeview widget with integrated vertical and horizontal scrollbars.
    Provides built-in sorting functionality for each column.
    """
    def __init__(self, parent, columns, show="headings"):
        super().__init__(parent)
        # Create Treeview
        self.tree = ttk.Treeview(self, columns=columns, show=show)
        # Add scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        # Grid layout
        self.tree.grid(column=0, row=0, sticky="nsew")
        vsb.grid(column=1, row=0, sticky="ns")
        hsb.grid(column=0, row=1, sticky="ew")
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Setup sorting
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, anchor="center")
        self.sort_reverse = {}
        for col in columns:
            self.sort_reverse[col] = False

    def sort_by(self, col):
        """Sort treeview data by the specified column."""
        items = [
            (self.tree.set(item, col), item) for item in self.tree.get_children("")
        ]
        # Convert string numbers to float for proper numerical sorting
        try:
            items = [(float(key), item) for key, item in items]
            items.sort(reverse=self.sort_reverse[col])
        except ValueError:
            # String values - reset sort state and sort ascending first time
            self.sort_reverse[col] = False
            items.sort(key=lambda x: str(x[0]))
        # Rearrange items in sorted positions
        for index, (_, item) in enumerate(items):
            self.tree.move(item, "", index)
        # Toggle sort direction for next time
        self.sort_reverse[col] = not self.sort_reverse[col]


class CorpusAnalyzer:
    """
    Main class for corpus analysis with NLP techniques.

    This class handles text analysis using unigram and bigram models.
    It provides a GUI for visualizing the analysis results including:
    - Basic corpus statistics
    - Sentence tokenization
    - Unigram probabilities (with and without smoothing)
    - Bigram probabilities (with and without smoothing)
    - Bigram probability matrix visualization
    - Sentence probability calculation
    """
    def __init__(self):
        # Constants
        self.smoothing_k = 0.5

        # Initialize NLTK
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to download NLTK data: {str(e)}\nPlease check your internet connection.")
            raise

        # Create tokenizer once
        self.word_tokenizer = nltk.RegexpTokenizer(r"\w+")

        # Create root window first
        if DRAG_DROP_SUPPORTED:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("NLP Corpus Analyzer")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        # Data structures
        self.sentences = []
        self.all_words = []
        self.unique_words = set()
        self.word_occurrence = Counter()
        self.all_bigrams = []
        self.unique_bigrams = set()
        self.bigram_occurrence = Counter()
        self._denominator_cache = {}

        # UI state - now created after root window
        self.filename = tk.StringVar()
        self.num_sentences = tk.StringVar()
        self.num_all_words = tk.StringVar()
        self.num_unique_words = tk.StringVar()
        self.status_text = tk.StringVar()  # Add status text variable

        # Initialize UI components that will be created later
        self.status_bar = None
        self.tab_parent = None
        self.tab1 = None
        self.tab2 = None
        self.tab3 = None
        self.tab4 = None
        self.tab5 = None
        self.tab6 = None
        self.tab7 = None
        self.tab8 = None
        self.tab2_st = None
        self.tab3_tree = None
        self.tab4_tree = None
        self.tab5_tree = None
        self.tab6_tree = None
        self.tab7_tree = None
        self.entry = None
        self.tab8_result_label = None
        self.tab8_details_label = None

        self.setup_ui()

        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set up drag and drop only if supported
        if DRAG_DROP_SUPPORTED:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def setup_ui(self):
        """Set up the user interface components."""
        self.create_main_layout()
        self.create_tabs()

        # Create status bar
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_text, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(
            column=0, row=3, sticky=tk.EW, columnspan=2, padx=5, pady=2
        )

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.select_file())
        self.root.bind('<Control-r>', lambda e: self.analyze_button_click())
        self.root.bind('<Control-w>', lambda e: self.on_closing())
        self.root.bind('<Escape>', lambda e: self.clear_data())

    def create_main_layout(self):
        """Create the main application layout with buttons and labels."""
        # Create a custom style for the ANALYZE button
        style = ttk.Style()
        style.configure(
            "Analyze.TButton",
            font=("Arial", 10, "bold"),
            padding=5,
            background="#4CAF50",
        )

        open_button = ttk.Button(self.root, text="Open File", command=self.select_file)
        run_button = ttk.Button(
            self.root,
            text="🔍 ANALYZE",
            command=self.analyze_button_click,
            style="Analyze.TButton",
        )

        filename_label = ttk.Label(self.root, text="Selected Filename")
        filename_value = ttk.Label(self.root, textvariable=self.filename)

        open_button.grid(column=0, row=0, sticky=tk.NW, padx=5, pady=5)
        run_button.grid(column=1, row=0, sticky=tk.NW, padx=5, pady=5)
        filename_label.grid(column=0, row=1, sticky=tk.NW, padx=5, pady=5)
        filename_value.grid(column=1, row=1, sticky=tk.NW, padx=5, pady=5)

    def analyze_button_click(self):
        """Handle the analyze button click event, updating button state during analysis."""
        # Get the analyze button widget
        run_button = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Button) and child["text"] == "🔍 ANALYZE":
                run_button = child
                break

        if run_button:
            # Temporarily disable the button and change its text
            run_button["state"] = "disabled"
            run_button["text"] = "⏳ ANALYZING..."
            self.root.update_idletasks()  # Force update of the button appearance

            # Perform the analysis
            self.analyze_file()

            # Restore the button after analysis
            run_button["state"] = "normal"
            run_button["text"] = "🔍 ANALYZE"
            self.root.update_idletasks()  # Update the button appearance again

    def select_file(self):
        """Open a file dialog to select a corpus file for analysis."""
        filetypes = (("text files", "*.txt"), ("All files", "*.*"))
        self.filename.set(
            filedialog.askopenfilename(
                title="Open a file", initialdir="/", filetypes=filetypes
            )
        )

    def on_closing(self):
        """Handle cleanup when the window is closed"""
        try:
            # Clear any temporary files or resources
            self.clear_data()
            # Destroy the window
            self.root.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()

    def handle_drop(self, event):
        """Handle drag and drop of files"""
        try:
            file_path = event.data
            if file_path.lower().endswith('.txt'):
                self.filename.set(file_path)
                self.analyze_file()
            else:
                tk.messagebox.showerror("Error", "Please drop a .txt file")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    def analyze_file(self):
        """Process the selected corpus file and perform text analysis."""
        if not self.filename.get():
            tk.messagebox.showerror("Error", "Please select a file first")
            return

        try:
            import time

            start_time = time.time()

            self.status_text.set("Clearing previous data...")
            self.clear_data()

            self.status_text.set("Reading file...")
            file_path = Path(self.filename.get())
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            try:
                text = file_path.read_text(encoding="utf-8").strip().lower()
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                encodings = ['latin1', 'cp1252', 'ascii']
                for encoding in encodings:
                    try:
                        text = file_path.read_text(encoding=encoding).strip().lower()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("Failed to decode file with any known encoding")

            text = re.sub(r"\n\s+", " ", text)

            if not text.strip():
                raise ValueError("File is empty or contains no valid text")

            self.status_text.set("Processing text...")
            self.process_text(text)

            self.status_text.set("Calculating statistics...")
            self.calculate_statistics()

            self.status_text.set("Updating interface...")
            self.update_all_tabs()

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.status_text.set(f"Analysis completed in {elapsed_time:.2f} seconds")

        except Exception as e:
            error_msg = str(e)
            tk.messagebox.showerror("Error", f"Failed to analyze file: {error_msg}")
            self.status_text.set("Analysis failed")

    def process_text(self, text):
        """Process raw text to extract words, sentences and calculate initial statistics."""
        # Convert to lowercase before tokenization
        text = text.lower()
        self.sentences = nltk.sent_tokenize(text, language="english")
        words = self.word_tokenizer.tokenize(text)

        # Add special tokens after word tokenization
        special_tokens = ["<s>", "</s>"] * len(self.sentences)
        self.all_words = words + special_tokens

        # Update word occurrences
        self.word_occurrence = Counter(self.all_words)

        # Include all tokenized words in unique words, not just alphabetic ones
        self.unique_words = set(self.all_words)

        # Initialize denominator cache
        total_words = len(self.all_words)
        vocab_size = len(self.unique_words)
        self._denominator_cache = {
            "unigram": total_words + self.smoothing_k * vocab_size,
            "bigram_base": self.smoothing_k * (vocab_size - 2),
        }

        self.process_bigrams()

    def process_bigrams(self):
        """Extract and process bigrams from the sentences."""
        self.all_bigrams = []
        for sentence in self.sentences:
            words = self.word_tokenizer.tokenize(sentence)
            # Add <s> and </s> tokens for each sentence
            padded = list(pad_both_ends(words, n=2))
            self.all_bigrams.extend(list(bigrams(padded)))
        self.unique_bigrams = set(self.all_bigrams)
        self.bigram_occurrence = Counter(self.all_bigrams)

    def calculate_statistics(self):
        """Calculate corpus statistics and precompute smoothing cache values."""
        # Use Counter for efficient counting
        self.word_occurrence = Counter(self.all_words)
        self.bigram_occurrence = Counter(self.all_bigrams)

        # Pre-calculate denominators for smoothing
        total_words = len(self.all_words)
        vocab_size = len(self.unique_words)
        self._denominator_cache = {
            "unigram": total_words + self.smoothing_k * vocab_size,
            "bigram_base": self.smoothing_k * (vocab_size - 2),
        }

    def update_all_tabs(self):
        """Update all UI tabs with current analysis data."""
        self.update_tab1()
        self.update_tab2()
        self.update_tab3()
        self.update_tab4()
        self.update_tab5()
        self.update_tab6()
        self.update_tab7()

    def update_tab1(self):
        """Update the statistics tab with corpus metrics."""
        self.num_sentences.set(str(len(self.sentences)))
        self.num_all_words.set(str(len(self.all_words)))
        self.num_unique_words.set(str(len(self.unique_words)))

    def update_tab2(self):
        """Update the sentences tab with tokenized sentences."""
        self.tab2_st.delete("1.0", tk.END)
        for idx, sentence in enumerate(self.sentences):
            self.tab2_st.insert(tk.INSERT, f"{idx + 1} - {sentence}\n")

    def update_tab3(self):
        """Update the unigram tab with word frequencies and probabilities."""
        self.tab3_tree.tree.delete(*self.tab3_tree.tree.get_children())
        for idx, word in enumerate(self.word_occurrence, 1):
            prob = self.get_unigram_prob(word)
            self.tab3_tree.tree.insert(
                "", "end", values=(idx, word, self.word_occurrence[word], f"{prob:.6f}")
            )

    def get_unigram_prob(self, word):
        """Calculate the unigram probability of a word."""
        return self.word_occurrence[word] / len(self.all_words)

    def update_tab4(self):
        """Update the bigram tab with bigram frequencies and probabilities."""
        self.tab4_tree.tree.delete(*self.tab4_tree.tree.get_children())
        for idx, bigram in enumerate(self.bigram_occurrence, 1):
            prob = self.get_bigram_prob(bigram)
            self.tab4_tree.tree.insert(
                "",
                "end",
                values=(
                    idx,
                    f"{bigram[0]}|{bigram[1]}",
                    self.bigram_occurrence[bigram],
                    f"{prob:.6f}",
                ),
            )

    def get_bigram_prob(self, bigram):
        """Calculate the conditional probability P(w2|w1) for a bigram (w1,w2)."""
        try:
            if (
                bigram[0] not in self.word_occurrence
                or self.word_occurrence[bigram[0]] == 0
            ):
                return 0
            return self.bigram_occurrence[bigram] / self.word_occurrence[bigram[0]]
        except KeyError:
            print("KeyError in get_bigram_prob", bigram)
            return 0

    def update_tab5(self):
        """Update the smoothed unigram tab with word frequencies and smoothed probabilities."""
        self.tab5_tree.tree.delete(*self.tab5_tree.tree.get_children())
        for idx, word in enumerate(self.word_occurrence, 1):
            prob = self.get_unigram_prob_smooth(word)
            self.tab5_tree.tree.insert(
                "", "end", values=(idx, word, self.word_occurrence[word], f"{prob:.6f}")
            )

    @lru_cache(maxsize=1024)
    def get_unigram_prob_smooth(self, word):
        """Calculate the smoothed unigram probability of a word using add-k smoothing."""
        if word in self.unique_words:
            return (
                self.word_occurrence[word] + self.smoothing_k
            ) / self._denominator_cache["unigram"]
        return self.smoothing_k / self._denominator_cache["unigram"]

    def update_tab6(self):
        """Update the smoothed bigram tab with bigram frequencies and smoothed probabilities."""
        self.tab6_tree.tree.delete(*self.tab6_tree.tree.get_children())
        for idx, bigram in enumerate(self.bigram_occurrence, 1):
            prob = self.get_bigram_prob_smooth(bigram)
            self.tab6_tree.tree.insert(
                "",
                "end",
                values=(
                    idx,
                    f"{bigram[0]}|{bigram[1]}",
                    self.bigram_occurrence[bigram],
                    f"{prob:.6f}",
                ),
            )

    @lru_cache(maxsize=1024)
    def get_bigram_prob_smooth(self, bigram):
        """Calculate the smoothed conditional probability P(w2|w1) for a bigram using add-k smoothing."""
        try:
            if self._denominator_cache["bigram_base"] == 0:
                return 0

            denominator = (
                self.word_occurrence[bigram[0]] + self._denominator_cache["bigram_base"]
            )
            if bigram in self.bigram_occurrence:
                return (self.bigram_occurrence[bigram] + self.smoothing_k) / denominator
            elif bigram[0] in self.word_occurrence:
                return self.smoothing_k / denominator
            return self.smoothing_k / self._denominator_cache["bigram_base"]
        except KeyError:
            print("KeyError in get_bigram_prob_smooth", bigram)
            return 1

    def update_tab7(self):
        """Update the bigram probability matrix tab with a cross-tabulation of all word pairs."""
        # First, remove old columns and data
        self.tab7_tree.tree.delete(*self.tab7_tree.tree.get_children())
        for col in self.tab7_tree.tree["columns"]:
            self.tab7_tree.tree.heading(col, text="")

        # Create new columns based on current unique words
        words_list = sorted(list(self.unique_words))
        columns = ["Word"] + words_list

        # Configure the treeview with new columns
        self.tab7_tree.tree["columns"] = columns
        self.tab7_tree.tree.column("Word", width=100, anchor="w")
        self.tab7_tree.tree.heading("Word", text="Word")

        # Setup column headings and widths
        for word in words_list:
            self.tab7_tree.tree.column(word, width=70, anchor="center")
            self.tab7_tree.tree.heading(word, text=word)

        # Add rows with probabilities
        for word1 in words_list:
            row_values = [word1]
            for word2 in words_list:
                prob = self.get_bigram_prob_smooth((word1, word2))
                row_values.append(f"{prob:.3f}")
            self.tab7_tree.tree.insert("", "end", values=row_values)

    def clear_data(self):
        """Clear all analysis data and reset UI components."""
        self.sentences = []
        self.all_words = []
        self.unique_words = set()
        self.word_occurrence = Counter()
        self.all_bigrams = []
        self.unique_bigrams = set()
        self.bigram_occurrence = Counter()

        # Clear treeviews
        if hasattr(self, "tab3_tree"):
            self.tab3_tree.tree.delete(*self.tab3_tree.tree.get_children())
        if hasattr(self, "tab4_tree"):
            self.tab4_tree.tree.delete(*self.tab4_tree.tree.get_children())
        if hasattr(self, "tab5_tree"):
            self.tab5_tree.tree.delete(*self.tab5_tree.tree.get_children())
        if hasattr(self, "tab6_tree"):
            self.tab6_tree.tree.delete(*self.tab6_tree.tree.get_children())
        if hasattr(self, "tab7_tree"):
            self.tab7_tree.tree.delete(*self.tab7_tree.tree.get_children())

    def create_tabs(self):
        """Create the tab structure for the application."""
        self.tab_parent = ttk.Notebook(
            self.root, width=890, height=420
        )  # Added fixed height
        self.tab1 = ttk.Frame(self.tab_parent)
        self.tab2 = ttk.Frame(self.tab_parent)
        self.tab3 = ttk.Frame(self.tab_parent)
        self.tab4 = ttk.Frame(self.tab_parent)
        self.tab5 = ttk.Frame(self.tab_parent)
        self.tab6 = ttk.Frame(self.tab_parent)
        self.tab7 = ttk.Frame(self.tab_parent)
        self.tab8 = ttk.Frame(self.tab_parent)

        # Configure grid weights to allow status bar to show
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, minsize=30)  # Reserve space for status bar
        self.root.grid_columnconfigure(0, weight=1)

        self.tab_parent.grid(
            column=0, row=2, sticky=tk.NSEW, padx=5, pady=5, columnspan=2
        )

        self.tab_parent.add(self.tab1, text="Statistics")
        self.tab_parent.add(self.tab2, text="Sentences")
        self.tab_parent.add(self.tab3, text="Unigram")
        self.tab_parent.add(self.tab4, text="Bigram")
        self.tab_parent.add(self.tab5, text="Unigram (k=0.5 smooth)")
        self.tab_parent.add(self.tab6, text="Bigram (k=0.5 smooth)")
        self.tab_parent.add(self.tab7, text="Smooth Bigram Table")
        self.tab_parent.add(self.tab8, text="Test")

        self.create_tab1()
        self.create_tab2()
        self.create_tab3()
        self.create_tab4()
        self.create_tab5()
        self.create_tab6()
        self.create_tab7()
        self.create_tab8()

    def create_tab1(self):
        """Create the statistics tab with basic corpus metrics."""
        sentences_label = ttk.Label(self.tab1, text="Number of sentences:")
        sentences_label_value = ttk.Label(self.tab1, textvariable=self.num_sentences)
        allword_label = ttk.Label(self.tab1, text="Number of all words:")
        allword_label_value = ttk.Label(self.tab1, textvariable=self.num_all_words)
        uniqueword_label = ttk.Label(self.tab1, text="Number of unique words:")
        uniqueword_label_value = ttk.Label(
            self.tab1, textvariable=self.num_unique_words
        )

        sentences_label.grid(column=0, row=0, sticky=tk.NW, padx=5, pady=5)
        sentences_label_value.grid(column=1, row=0, sticky=tk.NW, padx=5, pady=5)
        allword_label.grid(column=0, row=1, sticky=tk.NW, padx=5, pady=5)
        allword_label_value.grid(column=1, row=1, sticky=tk.NW, padx=5, pady=5)
        uniqueword_label.grid(column=0, row=2, sticky=tk.NW, padx=5, pady=5)
        uniqueword_label_value.grid(column=1, row=2, sticky=tk.NW, padx=5, pady=5)

    def create_tab2(self):
        """Create the sentences tab with a scrollable text area."""
        self.tab2_st = ScrolledText(self.tab2)
        self.tab2_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    def create_tab3(self):
        """Create the unigram tab with a treeview for unigram probabilities."""
        columns = ("Index", "Word", "Occurrence", "Probability")
        self.tab3_tree = TreeviewWithScroll(self.tab3, columns)
        self.tab3_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col in columns:
            self.tab3_tree.tree.column(col, width=100)

    def create_tab4(self):
        """Create the bigram tab with a treeview for bigram probabilities."""
        columns = ("Index", "Bigram", "Occurrence", "Probability")
        self.tab4_tree = TreeviewWithScroll(self.tab4, columns)
        self.tab4_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col in columns:
            self.tab4_tree.tree.column(col, width=150)

    def create_tab5(self):
        """Create the smoothed unigram tab with a treeview for smoothed unigram probabilities."""
        columns = ("Index", "Word", "Occurrence", "Smoothed Probability")
        self.tab5_tree = TreeviewWithScroll(self.tab5, columns)
        self.tab5_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col in columns:
            self.tab5_tree.tree.column(col, width=100)

    def create_tab6(self):
        """Create the smoothed bigram tab with a treeview for smoothed bigram probabilities."""
        columns = ("Index", "Bigram", "Occurrence", "Smoothed Probability")
        self.tab6_tree = TreeviewWithScroll(self.tab6, columns)
        self.tab6_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col in columns:
            self.tab6_tree.tree.column(col, width=150)

    def create_tab7(self):
        """Create the bigram matrix tab with a matrix view of all word pairs."""
        # For tab7, we'll use a matrix view for all word pairs
        frame = ttk.Frame(self.tab7)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize with just a "Word" column - we'll update columns when data is loaded
        self.tab7_tree = TreeviewWithScroll(frame, ["Word"])
        self.tab7_tree.pack(fill=tk.BOTH, expand=True)

        # Set reasonable default width
        self.tab7_tree.tree.column("Word", width=100, anchor="w")

    def create_tab8(self):
        """Create the test tab for calculating probabilities of user-entered sentences."""
        labela = tk.Label(self.tab8, text="Enter your sentence:", font=("Courier 15"))
        labela.pack(pady=20)
        self.entry = tk.Entry(self.tab8, width=40)
        self.entry.focus_set()
        self.entry.pack()
        ttk.Button(
            self.tab8, text="Calculate", width=20, command=self.find_prob_of_sentence
        ).pack(pady=20)
        self.tab8_result_label = tk.Label(self.tab8, text="", font="Courier 22 bold")
        self.tab8_result_label.pack()
        self.tab8_details_label = tk.Label(self.tab8, text="", font="Courier 11")
        self.tab8_details_label.pack()

    def find_prob_of_sentence(self):
        """Calculate the probability of a user-entered sentence."""
        prob = 1
        detail_str = ""
        string = self.entry.get()
        words = self.word_tokenizer.tokenize(string.lower())
        sentence_bigrams = list(bigrams(pad_both_ends(words, n=2)))

        # Calculate probabilities once
        probs = [(big, self.get_bigram_prob_smooth(big)) for big in sentence_bigrams]

        for big, p in probs:
            prob *= p
            detail_str += f"P({big[1]}|{big[0]}) = {p:.3f}\n"

        self.tab8_result_label.configure(text=str(prob))
        self.tab8_details_label.configure(text=detail_str)

    def run(self):
        """Run the application main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = CorpusAnalyzer()
    app.run()
