import pytest
from pathlib import Path
from main import CorpusAnalyzer, TreeviewWithScroll
from collections import Counter
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


@pytest.fixture(scope="class")
def analyzer():
    # Create a test instance without running the GUI
    analyzer = CorpusAnalyzer()
    # Prevent mainloop from running
    analyzer.root.withdraw()
    yield analyzer
    # Clean up Tkinter resources after all tests
    analyzer.root.destroy()


@pytest.fixture
def treeview():
    root = tk.Tk()
    tree = TreeviewWithScroll(root, columns=("Col1", "Col2"))
    yield tree
    root.destroy()


@pytest.fixture(autouse=True)
def clear_data(analyzer):
    # Reset data before each test
    analyzer.clear_data()
    yield


def test_process_text_basic(analyzer):
    """Test basic text processing functionality"""
    test_text = "Hello world. This is a test."
    analyzer.process_text(test_text)

    # Test sentence splitting
    assert len(analyzer.sentences) == 2
    assert analyzer.sentences[0] == "hello world."
    assert analyzer.sentences[1] == "this is a test."

    # Test word tokenization (including special tokens)
    expected_words = ["hello", "world", "this", "is", "a", "test"]
    expected_words.extend(["<s>", "</s>"] * 2)  # Add special tokens for 2 sentences
    assert len(analyzer.all_words) == len(expected_words)

    # Test unique words
    expected_unique = {"hello", "world", "this", "is", "a", "test", "<s>", "</s>"}
    assert analyzer.unique_words == expected_unique


def test_unigram_probabilities(analyzer):
    """Test unigram probability calculations"""
    test_text = "the cat and the dog"
    analyzer.process_text(test_text)

    # Test word occurrences
    assert analyzer.word_occurrence["the"] == 2
    assert analyzer.word_occurrence["cat"] == 1

    # Test unigram probabilities
    total_words = len(analyzer.all_words)
    assert pytest.approx(analyzer.get_unigram_prob("the")) == 2 / total_words
    assert pytest.approx(analyzer.get_unigram_prob("cat")) == 1 / total_words


def test_bigram_probabilities(analyzer):
    """Test bigram probability calculations"""
    test_text = "the cat and the cat"
    analyzer.process_text(test_text)

    # Test bigram occurrences
    assert analyzer.bigram_occurrence[("<s>", "the")] == 1
    assert analyzer.bigram_occurrence[("the", "cat")] == 2

    # Test bigram probabilities
    assert (
        pytest.approx(analyzer.get_bigram_prob(("the", "cat")))
        == 2 / analyzer.word_occurrence["the"]
    )


def test_smoothing(analyzer):
    """Test smoothed probability calculations"""
    test_text = "the cat"
    analyzer.process_text(test_text)

    # Test smoothed unigram probability for unseen word
    unseen_prob = analyzer.get_unigram_prob_smooth("dog")
    assert unseen_prob > 0

    # Test smoothed bigram probability for unseen bigram
    unseen_bigram_prob = analyzer.get_bigram_prob_smooth(("the", "dog"))
    assert unseen_bigram_prob > 0


def test_clear_data(analyzer):
    """Test data clearing functionality"""
    test_text = "the cat and the dog"
    analyzer.process_text(test_text)
    analyzer.clear_data()

    assert len(analyzer.sentences) == 0
    assert len(analyzer.all_words) == 0
    assert len(analyzer.unique_words) == 0
    assert len(analyzer.word_occurrence) == 0
    assert len(analyzer.bigram_occurrence) == 0


def test_treeview_sorting(treeview):
    """Test TreeviewWithScroll sorting functionality"""
    # Add test data
    treeview.tree.insert("", "end", values=("1", "10"))
    treeview.tree.insert("", "end", values=("2", "5"))
    treeview.tree.insert("", "end", values=("3", "15"))

    # Test numeric sorting
    treeview.sort_by("Col2")
    items = [
        (treeview.tree.set(item, "Col2"), item)
        for item in treeview.tree.get_children("")
    ]
    assert [float(val) for val, _ in items] == [5.0, 10.0, 15.0]

    # Test string sorting
    treeview.sort_by("Col1")
    items = [
        (treeview.tree.set(item, "Col1"), item)
        for item in treeview.tree.get_children("")
    ]
    assert [val for val, _ in items] == ["1", "2", "3"]

    # Test sorting with non-numeric values when attempting numeric conversion
    treeview.tree.delete(*treeview.tree.get_children())
    treeview.tree.insert("", "end", values=("a", "abc"))
    treeview.tree.insert("", "end", values=("b", "def"))
    treeview.sort_by("Col2")
    items = [
        (treeview.tree.set(item, "Col2"), item)
        for item in treeview.tree.get_children("")
    ]
    assert [val for val, _ in items] == ["abc", "def"]


def test_calculate_statistics(analyzer):
    """Test statistics calculation"""
    test_text = "the cat and the dog"
    analyzer.process_text(test_text)
    analyzer.calculate_statistics()

    assert len(analyzer.word_occurrence) > 0
    assert len(analyzer.bigram_occurrence) > 0
    assert analyzer._denominator_cache["unigram"] > 0
    assert analyzer._denominator_cache["bigram_base"] > 0


def test_update_tabs(analyzer):
    """Test tab updates"""
    test_text = "the cat and the dog"
    analyzer.process_text(test_text)

    # Test individual tab updates
    analyzer.update_tab1()
    assert analyzer.num_sentences.get() == "1"
    assert analyzer.num_all_words.get() == str(len(analyzer.all_words))
    assert analyzer.num_unique_words.get() == str(len(analyzer.unique_words))


def test_error_handling(analyzer):
    """Test error handling in probability calculations"""
    # Test bigram probability with non-existent words
    prob = analyzer.get_bigram_prob(("nonexistent", "word"))
    assert prob == 0  # Should return 0 for non-existent bigrams

    # Test smoothed bigram probability with non-existent words
    prob = analyzer.get_bigram_prob_smooth(("nonexistent", "word"))
    assert prob > 0  # Should return small non-zero probability


def test_analyze_button_click(analyzer):
    """Test analyze button click functionality"""
    # Create a temporary test file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test file.")
        test_file = f.name

    # Add a test button to simulate the ANALYZE button
    test_button = ttk.Button(analyzer.root, text="🔍 ANALYZE")
    test_button.grid()

    # Set test filename
    analyzer.filename.set(test_file)

    # Call analyze_button_click
    analyzer.analyze_button_click()

    # Verify status updates
    assert analyzer.status_text.get().startswith("Analysis completed")

    # Clean up
    import os

    os.unlink(test_file)


def test_tab_creation(analyzer):
    """Test tab creation and configuration"""
    # Verify all tabs were created
    assert hasattr(analyzer, "tab1")
    assert hasattr(analyzer, "tab2")
    assert hasattr(analyzer, "tab3")
    assert hasattr(analyzer, "tab4")
    assert hasattr(analyzer, "tab5")
    assert hasattr(analyzer, "tab6")
    assert hasattr(analyzer, "tab7")
    assert hasattr(analyzer, "tab8")

    # Test tab configurations
    assert isinstance(analyzer.tab2_st, ScrolledText)
    assert isinstance(analyzer.tab3_tree, TreeviewWithScroll)
    assert isinstance(analyzer.tab4_tree, TreeviewWithScroll)


def test_update_tabs_edge_cases(analyzer):
    """Test tab updates with edge cases"""
    # Test with empty text
    analyzer.process_text("")
    analyzer.update_all_tabs()
    assert analyzer.num_sentences.get() == "0"
    assert analyzer.num_all_words.get() == "0"

    # Test with special characters
    analyzer.process_text("Hello! @#$%^&*()")
    analyzer.update_all_tabs()
    assert analyzer.num_sentences.get() == "2"  # NLTK treats "!" as sentence boundary
    assert len(analyzer.unique_words) == 3  # hello, <s>, </s>


def test_find_prob_of_sentence(analyzer):
    """Test sentence probability calculation"""
    # Process some training text first
    analyzer.process_text("the cat sat on the mat")

    # Test probability calculation
    analyzer.entry = tk.Entry(analyzer.root)
    analyzer.entry.insert(0, "the cat")
    analyzer.tab8_result_label = tk.Label(analyzer.root)
    analyzer.tab8_details_label = tk.Label(analyzer.root)

    analyzer.find_prob_of_sentence()

    # Verify labels were updated
    assert analyzer.tab8_result_label.cget("text") != ""
    assert analyzer.tab8_details_label.cget("text") != ""


def test_file_dialog(analyzer, monkeypatch):
    """Test file dialog functionality"""

    # Mock filedialog.askopenfilename to return a known value
    def mock_filedialog(*args, **kwargs):
        return "/path/to/test.txt"

    monkeypatch.setattr("tkinter.filedialog.askopenfilename", mock_filedialog)

    # Test file selection
    analyzer.select_file()
    assert analyzer.filename.get() == "/path/to/test.txt"


def test_tab7_full_update(analyzer):
    """Test tab7 tree configuration with actual data"""
    # Setup test data with words that will force column updates
    test_text = "word1 word2"
    analyzer.process_text(test_text)

    # Clear existing columns
    analyzer.tab7_tree.tree["columns"] = ["Word"]

    # Update tab7
    analyzer.update_tab7()

    # Verify column configuration
    assert "word1" in analyzer.tab7_tree.tree["columns"]
    assert "word2" in analyzer.tab7_tree.tree["columns"]

    # Get width of a column
    assert analyzer.tab7_tree.tree.column("word1")["width"] == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
