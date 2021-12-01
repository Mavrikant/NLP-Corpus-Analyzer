#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from builtins import str

import nltk
import tkinter as tk
# nltk.download('punkt')
from pathlib import Path
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.scrolledtext import ScrolledText
from nltk import bigrams
from nltk.lm.preprocessing import pad_both_ends

k = 0.5
root = tk.Tk()

filename = tk.StringVar()
filename.set('')

textvariable_numOfSentences = tk.StringVar()
textvariable_numOfSentences.set('')

textvariable_numOfAllWords = tk.StringVar()
textvariable_numOfAllWords.set('')

textvariable_numOfUniqueWords = tk.StringVar()
textvariable_numOfUniqueWords.set('')

numOfUniqueWords = 0

sentences = []

allWords = []
uniqueWords = set()
word_occurence_dict = {}

allBigrams = []
uniqueBigrams = set()
bigram_occurence_dict = {}


def select_file():
    filetypes = (('text files', '*.txt'), ('All files', '*.*'))
    global filename
    filename.set(fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes))


def analyze_file():
    global filename
    global numOfUniqueWords
    global textvariable_numOfSentences, textvariable_numOfAllWords, textvariable_numOfUniqueWords
    global sentences, allWords, uniqueWords
    clearData()

    text = Path(filename.get()).read_text(encoding='utf-8').strip().lower()
    text = re.sub(r"\n\s+", " ", text)  # remove empty space at the start of lines

    sentences = nltk.sent_tokenize(text, language='english')
    allWords = nltk.RegexpTokenizer(r"\w+").tokenize(text)
    allWords.extend(['<s>'] * len(sentences))
    allWords.extend(['</s>'] * len(sentences))
    uniqueWords = set([word.lower() for word in allWords if word.isalpha()])
    uniqueWords.add('<s>')
    uniqueWords.add('</s>')

    findbigrams()
    calculateOccurenceOfWords()
    calculateOccurenceOfBigrams()

    fillTab1()
    fillTab2()
    fillTab3()
    fillTab4()
    fillTab5()
    fillTab6()
    fillTab7()


def clearData():
    global numOfUniqueWords, sentences, allWords, uniqueWords, word_occurence_dict
    global allBigrams, uniqueBigrams, bigram_occurence_dict
    numOfUniqueWords = 0
    sentences = []
    allWords = []
    uniqueWords = set()
    word_occurence_dict = {}
    allBigrams = []
    uniqueBigrams = set()
    bigram_occurence_dict = {}
    tab2_st.delete("1.0", tk.END)
    tab3_st.delete("1.0", tk.END)
    tab4_st.delete("1.0", tk.END)
    tab5_st.delete("1.0", tk.END)
    tab6_st.delete("1.0", tk.END)
    tab7_st.delete("1.0", tk.END)


def fillTab1():
    global textvariable_numOfSentences, textvariable_numOfAllWords, textvariable_numOfUniqueWords
    textvariable_numOfSentences.set(str(len(sentences)))
    textvariable_numOfAllWords.set(str(len(allWords)))
    textvariable_numOfUniqueWords.set(str(len(uniqueWords)))


def fillTab2():
    for idx, sen in enumerate(sentences):
        tab2_st.insert(tk.INSERT, str(idx + 1) + " - " + sen + "\n")


def fillTab3():
    global word_occurence_dict
    tab3_st.insert(tk.INSERT, "#" + " - " + "Word" + " \t\t " + "Occurance" + "\t\t" + "P()\n")
    for idx, word in enumerate(word_occurence_dict):
        tab3_st.insert(tk.INSERT,
                       str(idx + 1) + " - " + word + " \t\t " + str(word_occurence_dict[word]) + " \t\t " + str(
                           getUnigramProb(word)) + "\n")


def getUnigramProb(word):
    return word_occurence_dict[word] / len(allWords)


def fillTab4():
    global word_occurence_dict, bigram_occurence_dict
    tab4_st.insert(tk.INSERT, "#" + " - " + "Bigram" + " \t\t\t " + "Occurance" + "\t\t" + "P()\n")
    for idx, bigram in enumerate(bigram_occurence_dict):
        tab4_st.insert(tk.INSERT, str(idx + 1) + " - P(" + bigram[1] + "|" + bigram[0] + ") \t\t\t " + str(
            bigram_occurence_dict[bigram]) + "\t\t" + str(
            getBigramProb(bigram)) + "\n")


def getBigramProb(bigram):
    try:
        return bigram_occurence_dict[bigram] / word_occurence_dict[bigram[0]]
    except:
        print("except getBigramProb", bigram)  # There is key error on dict in one case
        return 1


def fillTab5():
    global word_occurence_dict
    tab5_st.insert(tk.INSERT, "#" + " - " + "Word" + " \t\t " + "Occurance" + "\t\t" + "P()\n")
    for idx, word in enumerate(word_occurence_dict):
        tab5_st.insert(tk.INSERT,
                       str(idx + 1) + " - " + word + " \t\t " + str(word_occurence_dict[word]) + " \t\t " + str(
                           getUnigramProbSmooth(word)) + "\n")


def getUnigramProbSmooth(word):
    if word in uniqueWords:
        return (word_occurence_dict[word] + k) / (len(allWords) + k * len(uniqueWords))
    else:
        return (0 + k) / (len(allWords) + k * len(uniqueWords))


def fillTab6():
    global word_occurence_dict, bigram_occurence_dict
    tab6_st.insert(tk.INSERT, "#" + " - " + "Bigram" + " \t\t\t " + "Occurance" + "\t\t" + "P()\n")
    for idx, bigram in enumerate(bigram_occurence_dict):
        tab6_st.insert(tk.INSERT, str(idx + 1) + " - P(" + bigram[1] + "|" + bigram[0] + ") \t\t\t " + str(
            bigram_occurence_dict[bigram]) + "\t\t" + str(
            getBigramProbSmooth(bigram)) + "\n")


def getBigramProbSmooth(bigram):
    try:
        if bigram in bigram_occurence_dict:
            return (bigram_occurence_dict[bigram] + k) / (word_occurence_dict[bigram[0]] + k * (len(uniqueWords) - 2))
        elif bigram[0] in word_occurence_dict:
            return (0 + k) / (word_occurence_dict[bigram[0]] + k * (len(uniqueWords) - 2))
        else:
            return (0 + k) / (0 + k * (len(uniqueWords) - 2))
    except:
        print("except getBigramProbSmooth", bigram)  # There is key error on dict in one case
        return 1


def fillTab7():
    str = ""
    str += "\t"
    for word2 in uniqueWords:
        str += word2
        str += "\t"

    str += "\n"

    for word1 in uniqueWords:
        str += word1
        str += "\t"
        for word2 in uniqueWords:
            str += ("%.3f" % getBigramProbSmooth((word1, word2)))
            str += "\t"
        str += "\n"
    tab7_st.insert(tk.INSERT, str)


def findbigrams():
    global allBigrams, uniqueBigrams
    for sen in sentences:
        allBigrams += list(bigrams(pad_both_ends(nltk.RegexpTokenizer(r"\w+").tokenize(sen.lower()), n=2)))

    uniqueBigrams = set(allBigrams)


def calculateOccurenceOfWords():
    global word_occurence_dict

    for wordd in uniqueWords:
        word_occurence_dict[wordd] = allWords.count(wordd)

    word_occurence_dict = dict(
        sorted(word_occurence_dict.items(), key=lambda item: item[1], reverse=True))  # sort dictionary based on values


def calculateOccurenceOfBigrams():
    global bigram_occurence_dict

    for bigram in uniqueBigrams:
        bigram_occurence_dict[bigram] = allBigrams.count(bigram)

    bigram_occurence_dict = dict(
        sorted(bigram_occurence_dict.items(), key=lambda item: item[1],
               reverse=True))  # sort dictionary based on values


def find_prob_of_sentence():
    global entry
    prob = 1
    detailstr = ""
    string = entry.get()
    sentenceBigrams = list(bigrams(pad_both_ends(nltk.RegexpTokenizer(r"\w+").tokenize(string.lower()), n=2)))
    for big in sentenceBigrams:
        prob = prob * getBigramProbSmooth(big)
        detailstr += "P(" + big[1] + "|" + big[0] + ") = " + str(("%.3f" % getBigramProbSmooth(big))) + "\n"

    tab8ResultLabel.configure(text=str(prob))
    tab8DetailsLabel.configure(text=detailstr)


############## MAIN ################


# create the root window
root.title('NLP Corpus Analyzer')
root.iconbitmap('images/icon.ico')
root.resizable(False, False)
root.geometry('900x600')
root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=1)

# open button
open_button = ttk.Button(root, text='Open File', command=select_file)
run_button = ttk.Button(root, text='ANALYZE', command=analyze_file)

filename_label = ttk.Label(root, text="Selected Filename")
filename_label.grid(column=0, row=1, sticky=tk.NW, padx=5, pady=5)

filenamevar_label = ttk.Label(root, textvariable=filename)
filenamevar_label.grid(column=1, row=1, sticky=tk.NW, padx=5, pady=5)

open_button.grid(column=0, row=0, sticky=tk.NW, padx=5, pady=5)
run_button.grid(column=1, row=0, sticky=tk.NW, padx=5, pady=5)

tab_parent = ttk.Notebook(root, width=900 - 10)
tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab3 = ttk.Frame(tab_parent)
tab4 = ttk.Frame(tab_parent)
tab5 = ttk.Frame(tab_parent)
tab6 = ttk.Frame(tab_parent)
tab7 = ttk.Frame(tab_parent)
tab8 = ttk.Frame(tab_parent)

tab_parent.grid(column=0, row=2, sticky=tk.NSEW, padx=5, pady=5, columnspan=2)

tab_parent.add(tab1, text="Statistics")
tab_parent.add(tab2, text="Sentences")
tab_parent.add(tab3, text="Unigram")
tab_parent.add(tab4, text="Bigram")
tab_parent.add(tab5, text="Unigram (k=0.5 smooth)")
tab_parent.add(tab6, text="Bigram (k=0.5 smooth)")
tab_parent.add(tab7, text="Smooth Bigram Table")
tab_parent.add(tab8, text="Test")

# Tab1
sentences_label = ttk.Label(tab1, text="Number of sentences:")
sentences_label_value = ttk.Label(tab1, textvariable=textvariable_numOfSentences)
allword_label = ttk.Label(tab1, text="Number of all words:")
allword_label_value = ttk.Label(tab1, textvariable=textvariable_numOfAllWords)
uniqueword_label = ttk.Label(tab1, text="Number of unique words:")
uniqueword_label_value = ttk.Label(tab1, textvariable=textvariable_numOfUniqueWords)

sentences_label.grid(column=0, row=0, sticky=tk.NW, padx=5, pady=5)
sentences_label_value.grid(column=1, row=0, sticky=tk.NW, padx=5, pady=5)
allword_label.grid(column=0, row=1, sticky=tk.NW, padx=5, pady=5)
allword_label_value.grid(column=1, row=1, sticky=tk.NW, padx=5, pady=5)
uniqueword_label.grid(column=0, row=2, sticky=tk.NW, padx=5, pady=5)
uniqueword_label_value.grid(column=1, row=2, sticky=tk.NW, padx=5, pady=5)

# Tab2
tab2_st = ScrolledText(tab2)
tab2_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab3
tab3_st = ScrolledText(tab3)
tab3_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab4
tab4_st = ScrolledText(tab4)
tab4_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab5
tab5_st = ScrolledText(tab5)
tab5_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab6
tab6_st = ScrolledText(tab6)
tab6_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab7
tab7_st = ScrolledText(tab7)
tab7_st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Tab8
labela = tk.Label(tab8, text="Enter your sentence:", font=("Courier 15"))
labela.pack(pady=20)
entry = tk.Entry(tab8, width=40)
entry.focus_set()
entry.pack()
ttk.Button(tab8, text="Calculate", width=20, command=find_prob_of_sentence).pack(pady=20)
tab8ResultLabel = tk.Label(tab8, text="", font=("Courier 22 bold"))
tab8ResultLabel.pack()
tab8DetailsLabel = tk.Label(tab8, text="", font=("Courier 11 "))
tab8DetailsLabel.pack()
# run the application
root.mainloop()
