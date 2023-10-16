# PorterStemmer
A simple Python program using Porter Stemmer Algorithm.

### Code Purpose:

The code implements the Porter Stemming algorithm, which is a widely used algorithm for reducing words to their root form. It reads an input sentence or paragraph, tokenizes it into words, and then applies the Porter Stemmer to each word. Additionally, it provides the capability to exclude certain words from stemming if they are considered root words.

Here's an explanation of each function's purpose in the code:

```
    PorterStemmer Class:
        __init__(self): Initializes the PorterStemmer class with an empty word.

    is_consonant(self, char):
        Determines if a character is a consonant or not.

    measure(self):
        Calculates the "measure" of a word, which represents the number of consonant sequences in it.

    contains_vowel(self):
        Checks if a word contains at least one vowel.

    ends_with(self, suffix):
        Checks if a word ends with a specific suffix and removes it if it does.

    step1a(self):
        Implements the first step of the Porter Stemming algorithm, handling plural forms and possessives.

    step1b(self):
        Implements the second step of the algorithm, dealing with past tense and continuous tense forms.

    step1b1(self):
        A helper function for step 1b that performs additional transformations.

    step1c(self):
        Implements the third step of the algorithm, handling various cases involving "y" at the end of a word.

    step2(self):
        Implements the fourth step of the algorithm, reducing words to their root form based on specific suffixes.

    step3(self):
        Implements the fifth step of the algorithm, reducing words to their root form based on additional suffixes.

    step4(self):
        Implements the sixth step of the algorithm, reducing words to their root form based on certain suffixes.

    step5a(self):
        Implements the seventh step of the algorithm, removing an "e" at the end of words in certain conditions.

    step5b(self):
        Implements the eighth step of the algorithm, removing an "l" at the end of words in certain conditions.

    is_root_word(self, word, root_words):
        Checks if a word is in the list of root words, allowing it to be excluded from stemming.

    stem(self, word, root_words):
        The main stemming function that applies all the Porter Stemming steps to a word while considering root words.

    read_root_words_from_file(filename):
        Reads root words from a text file and returns them as a list.

    Main Program:
        The main program reads root words from a file, takes user input, tokenizes it, applies stemming to each word, and prints the stemmed text.
```

The code's primary purpose is to reduce words to their root form using the Porter Stemming algorithm, and it provides the flexibility to exclude specific words from stemming by considering them as root words.
