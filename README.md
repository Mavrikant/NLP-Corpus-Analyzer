[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-270/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Mavrikant/NLP-Corpus-Analyzer?color=green)](https://github.com/Mavrikant/NLP-Corpus-Analyzer) [![Open issues on GitHub](https://img.shields.io/github/issues-raw/Mavrikant/NLP-Corpus-Analyzer)](https://github.com/Mavrikant/NLP-Corpus-Analyzer/issues) [![Open pull requests on GitHub](https://img.shields.io/github/issues-pr-raw/Mavrikant/NLP-Corpus-Analyzer)](https://github.com/Mavrikant/NLP-Corpus-Analyzer/pulls) [![GitHub contributors](https://img.shields.io/github/contributors/Mavrikant/NLP-Corpus-Analyzer)](https://github.com/Mavrikant/NLP-Corpus-Analyzer/graphs/contributors) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/e4bdd4f986eb44da819ae4dabf8aa27b)](https://www.codacy.com/gh/Mavrikant/NLP-Corpus-Analyzer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Mavrikant/NLP-Corpus-Analyzer&amp;utm_campaign=Badge_Grade) [![Tests](https://github.com/Mavrikant/NLP-Corpus-Analyzer/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/Mavrikant/NLP-Corpus-Analyzer/actions/workflows/test-coverage.yml) [![codecov](https://codecov.io/gh/Mavrikant/NLP-Corpus-Analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/Mavrikant/NLP-Corpus-Analyzer)

<div align="center">
  <a href="https://github.com/Mavrikant/NLP-Corpus-Analyzer">
    <img src="images/icon.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">NLP Corpus Analyzer</h3>
  <p align="center">
    A comprehensive text corpus analyzer with GUI for Natural Language Processing tasks
    <br />
    <a href="https://github.com/Mavrikant/NLP-Corpus-Analyzer"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Mavrikant/NLP-Corpus-Analyzer">View Demo</a>
    ·
    <a href="https://github.com/Mavrikant/NLP-Corpus-Analyzer/issues">Report Bug</a>
    ·
    <a href="https://github.com/Mavrikant/NLP-Corpus-Analyzer/issues">Request Feature</a>
  </p>
</div>

## About The Project
![Screenshot](images/Screenshot1.png)
![Screenshot](images/Screenshot2.png)

NLP Corpus Analyzer is a powerful Python application that provides comprehensive text analysis capabilities through an intuitive graphical interface. It's designed for analyzing text corpora using various NLP techniques, with a focus on n-gram probability calculations.

### Key Features

- **Text Analysis Dashboard**: View basic corpus statistics including sentence count, total words, and unique words
- **Sentence Tokenization**: Advanced sentence detection using NLTK
- **N-gram Analysis**:
  - Unigram frequency and probability calculations
  - Bigram frequency and conditional probability calculations
  - Support for add-k smoothing (k=0.5)
- **Interactive Visualization**:
  - Sortable tables for unigram and bigram statistics
  - Complete bigram probability matrix view
- **Sentence Probability Calculator**: Test the probability of custom sentences using the trained model
- **File Handling**:
  - Support for various text encodings
  - Drag-and-drop file support (when tkinterdnd2 is available)
  - Clear visualization of analysis progress

### Built With

* [Python](https://www.python.org/) - Core programming language
* [NLTK](https://www.nltk.org/) - Natural Language Processing toolkit
* [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework
* Optional: [tkinterdnd2](https://pypi.org/project/tkinterdnd2/) - Drag-and-drop support

## Getting Started

### Prerequisites

- Python 3.12 or later
- pip (Python package installer)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Mavrikant/NLP-Corpus-Analyzer.git
```

2. Install required packages
```bash
pip install -r requirements.txt
```

### Usage

1. Run the application:
```bash
python main.py
```

2. Use the interface to:
   - Open text files for analysis
   - View corpus statistics
   - Analyze n-gram probabilities
   - Test sentence probabilities
   - Export or save analysis results

### Keyboard Shortcuts

- `Ctrl+O`: Open file
- `Ctrl+R`: Run analysis
- `Ctrl+W`: Close application
- `Esc`: Clear current data

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!

## Contact

M. Serdar Karaman - m.serdar.karaman@gmail.com

Project Link: [https://github.com/Mavrikant/NLP-Corpus-Analyzer](https://github.com/Mavrikant/NLP-Corpus-Analyzer)

## License

Distributed under the MIT License. See [LICENSE](https://choosealicense.com/licenses/mit/) for more information.
