import csv
import os
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Define the Porter Stemmer class
class PorterStemmer:
    def __init__(self):
        self.word = ""

    # Function to check if a character is a consonant
    def is_consonant(self, char):
        if char in "aeiou":
            return False
        if char == 'y':
            return True
        if char not in "aeiouy":
            return True

    # Function to calculate the measure of a word
    def measure(self):
        word = self.word
        c = 0
        for i in range(len(word)):
            if self.is_consonant(word[i]):
                if i > 0 and not self.is_consonant(word[i - 1]):
                    c += 1
        return c

    # Function to check if a word contains a vowel
    def contains_vowel(self):
        word = self.word
        for char in word:
            if char in "aeiou":
                return True
        return False

    # Function to check if a word ends with a specific suffix
    def ends_with(self, suffix):
        word = self.word
        if word.endswith(suffix):
            remaining = word[: -len(suffix)]
            if self.measure() > 0:
                self.word = remaining
            return True
        return False

    # Step 1a of the Porter Stemming algorithm
    def step1a(self):
        word = self.word
        if word.endswith("sses"):
            word = word[: -2]
        elif word.endswith("ies"):
            word = word[: -2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[: -1]
        self.word = word

    # Step 1b of the Porter Stemming algorithm
    def step1b(self):
        word = self.word
        if word.endswith("eed"):
            if self.measure() > 0:
                word = word[: -1]
        elif word.endswith("ed"):
            if self.contains_vowel():
                word = word[: -2]
                self.step1b1()
        elif word.endswith("ing"):
            if self.contains_vowel():
                word = word[: -3]
                self.step1b1()
        self.word = word

    # Step 1b1 of the Porter Stemming algorithm
    def step1b1(self):
        word = self.word
        if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
            word += "e"
        elif (
            word[-1] == word[-2]
            and word[-1] not in "lsz"
            and not self.is_consonant(word[-1])
        ):
            word = word[: -1]
        elif self.measure() == 1 and self.ends_with("vcv"):
            word += "e"
        self.word = word

    # Step 1c of the Porter Stemming algorithm
    def step1c(self):
        word = self.word
        if word.endswith("y"):
            if self.contains_vowel():
                word = word[: -1] + "i"
        self.word = word

    # Step 2 of the Porter Stemming algorithm
    def step2(self):
        word = self.word
        if word.endswith("ational"):
            if self.measure() > 0:
                word = word[: -7] + "ate"
        elif word.endswith("tional"):
            if self.measure() > 0:
                word = word[: -6] + "tion"
        elif word.endswith("enci"):
            if self.measure() > 0:
                word = word[: -4] + "ence"
        elif word.endswith("anci"):
            if self.measure() > 0:
                word = word[: -4] + "ance"
        elif word.endswith("izer"):
            if self.measure() > 0:
                word = word[: -4] + "ize"
        elif word.endswith("abli"):
            if self.measure() > 0:
                word = word[: -4] + "able"
        elif word.endswith("alli"):
            if self.measure() > 0:
                word = word[: -4] + "al"
        elif word.endswith("entli"):
            if self.measure() > 0:
                word = word[: -5] + "ent"
        elif word.endswith("eli"):
            if self.measure() > 0:
                word = word[: -3] + "e"
        elif word.endswith("ousli"):
            if self.measure() > 0:
                word = word[: -5] + "ous"
        elif word.endswith("ization"):
            if self.measure() > 0:
                word = word[: -7] + "ize"
        elif word.endswith("ation"):
            if self.measure() > 0:
                word = word[: -5] + "ate"
        elif word.endswith("ator"):
            if self.measure() > 0:
                word = word[: -4] + "ate"
        elif word.endswith("alism"):
            if self.measure() > 0:
                word = word[: -5] + "al"
        elif word.endswith("iveness"):
            if self.measure() > 0:
                word = word[: -7] + "ive"
        elif word.endswith("fulness"):
            if self.measure() > 0:
                word = word[: -7] + "ful"
        elif word.endswith("ousness"):
            if self.measure() > 0:
                word = word[: -7] + "ous"
        elif word.endswith("aliti"):
            if self.measure() > 0:
                word = word[: -5] + "al"
        elif word.endswith("iviti"):
            if self.measure() > 0:
                word = word[: -5] + "ive"
        elif word.endswith("biliti"):
            if self.measure() > 0:
                word = word[: -6] + "ble"
        self.word = word

    # Step 3 of the Porter Stemming algorithm
    def step3(self):
        word = self.word
        if word.endswith("icate"):
            if self.measure() > 0:
                word = word[: -5] + "ic"
        elif word.endswith("ative"):
            if self.measure() > 0:
                word = word[: -5] + ""
        elif word.endswith("alize"):
            if self.measure() > 0:
                word = word[: -5] + "al"
        elif word.endswith("iciti"):
            if self.measure() > 0:
                word = word[: -5] + "ic"
        elif word.endswith("ical"):
            if self.measure() > 0:
                word = word[: -4] + "ic"
        elif word.endswith("ful"):
            if self.measure() > 0:
                word = word[: -3] + ""
        elif word.endswith("ness"):
            if self.measure() > 0:
                word = word[: -4] + ""
        self.word = word

    # Step 4 of the Porter Stemming algorithm
    def step4(self):
        word = self.word
        if word.endswith("al"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("ance"):
            if self.measure() > 1:
                word = word[: -4]
        elif word.endswith("ence"):
            if self.measure() > 1:
                word = word[: -4]
        elif word.endswith("er"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("ic"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("able"):
            if self.measure() > 1:
                word = word[: -4]
        elif word.endswith("ible"):
            if self.measure() > 1:
                word = word[: -4]
        elif word.endswith("ant"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ement"):
            if self.measure() > 1:
                word = word[: -5]
        elif word.endswith("ment"):
            if self.measure() > 1:
                word = word[: -4]
        elif word.endswith("ent"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ion"):
            if (
                self.measure() > 1
                and (word[-4] == "s" or word[-4] == "t")
            ):
                word = word[: -3]
        elif word.endswith("ou"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("ism"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ate"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("iti"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ous"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ive"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ize"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("ion"):
            if self.measure() > 1:
                word = word[: -3]
        elif word.endswith("al"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("er"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("ic"):
            if self.measure() > 1:
                word = word[: -2]
        elif word.endswith("s"):
            if self.measure() > 1:
                word = word[: -1]
        elif word.endswith("t"):
            if self.measure() > 1:
                word = word[: -1]
        self.word = word

    # Step 5a of the Porter Stemming algorithm
    def step5a(self):
        word = self.word
        if word.endswith("e"):
            if self.measure() > 1:
                word = word[: -1]
            elif self.measure() == 1 and not self.ends_with("cvc"):
                word = word[: -1]
        self.word = word

    # Step 5b of the Porter Stemming algorithm
    def step5b(self):
        word = self.word
        if self.measure() > 1 and word.endswith("l"):
            word = word[: -1]
        self.word = word

    # Function to check if a word is a root word
    def is_root_word(self, word, root_words):
        return word in root_words

    # Main stemming function
    def stem(self, word, root_words):
        if self.is_root_word(word, root_words):
            return word

        self.word = word.lower()

        if len(self.word) <= 2:
            return self.word

        # Apply the Porter Stemming steps
        self.step1a()
        self.step1b()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5a()
        self.step5b()

        return self.word


# Read data from CSV file and return a list of strings
def read_csv(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)  # Store rows as lists
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return data


def calculate_similarity(master_row, slave_row, counter):
    try:
        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Transform the data into TF-IDF vectors
        tfidf_matrix1 = tfidf_vectorizer.fit_transform([master_row])
        tfidf_matrix2 = tfidf_vectorizer.transform([slave_row])

        # Calculate cosine similarity for rows
        cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

        # Calculate the row difference
        row_difference = abs(len(master_row.split()) - len(slave_row.split()))

        # Adjust the similarity score based on row difference
        adjusted_similarity = cosine_sim[0][0] * 100 - row_difference  # Deduct the row difference

        # Debugging print statement
        # print(f"Similarity {counter}: {adjusted_similarity:.2f}%")
        return adjusted_similarity  # Return the adjusted similarity score
    except Exception as e:
        # Print any exceptions that occur during similarity calculation
        # print(f"Error calculating similarity: {str(e)}")
        return None


def pad_or_truncate(data, target_rows, target_cols):
    padded_data = []
    for i in range(target_rows):
        if i < len(data):
            row = data[i]
            if len(row) < target_cols:
                row += [''] * (target_cols - len(row))
            elif len(row) > target_cols:
                row = row[:target_cols]
            padded_data.append(row)
        else:
            padded_data.append([''] * target_cols)
    return padded_data


if __name__ == "__main__":
    # File paths for the two CSV files
    master_file_path = 'example-stemmed-dataset_15k-rows_tilaon-antonin.csv'
    slave_file_path = 'stemmed-dataset_15k-rows_sabroso-dean-clifford.csv'

    # Read data from CSV files
    master_data = read_csv(master_file_path)  # Master file
    slave_data = read_csv(slave_file_path)

    if master_data and slave_data:
        # Determine the number of CPU cores
        num_cores = os.cpu_count()

        with Pool(processes=num_cores) as pool:
            similarities = []

            # Calculate dimensions for padding or truncating slave_data to match master_data
            max_master_rows = len(master_data)
            max_master_cols = max(len(row) for row in master_data)
            max_slave_rows = len(slave_data)
            max_slave_cols = max(len(row) for row in slave_data)

            # Ensure both datasets have the same dimensions
            target_rows = max(max_master_rows, max_slave_rows)
            target_cols = max(max_master_cols, max_slave_cols)

            # Pad or truncate both datasets to match dimensions
            master_data = pad_or_truncate(master_data, target_rows, target_cols)
            slave_data = pad_or_truncate(slave_data, target_rows, target_cols)

            counter = 0

            # Calculate cosine similarity for each cell
            for i in range(target_rows):
                for j in range(target_cols):
                    cell1 = master_data[i][j]
                    cell2 = slave_data[i][j]

                    counter += 1
                    similarity = pool.apply_async(calculate_similarity, args=(cell1, cell2, counter))
                    similarities.append(similarity)

            # Calculate the total similarity
            total_similarity = 0.0
            num_valid_similarities = 0

            for similarity in similarities:
                sim = similarity.get()
                if sim is not None:
                    total_similarity += sim
                    num_valid_similarities += 1

            # Debugging print statements
            print("\nMaster file: ", master_file_path)
            print("Slave file: ", slave_file_path, "\n")
            print("Number of Valid Similarities:", num_valid_similarities)

            # Calculate the average similarity
            average_similarity = total_similarity / num_valid_similarities if num_valid_similarities > 0 else 0.0

            # Format the average similarity as a float with two decimal places
            formatted_average_similarity = f"{average_similarity:.2f}"
            print(f"Average Cell Similarity: {formatted_average_similarity}%\n")
    else:
        print("Data could not be loaded from one or both files.")






























""""
# Function to read root words from a text file
def read_root_words_from_csv(csv_file_path):


  root_words = []
  with open(csv_file_path, "r", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
      root_words.append(row[0])
  return root_words

def stem_word(word, root_words):



  if word in root_words:
    return word

  stemmer = PorterStemmer()
  stemmed_word = stemmer.stem(word)

  return stemmed_word

# Read the root words from the CSV file.
root_words = read_root_words_from_csv("4-cols_15k-rows.csv")


if __name__ == "__main__":
    input_csv_file = "4-cols_15k-rows.csv"  # Replace with your input CSV file
    output_csv_file = "stemmed-dataset_15k-rows_sabroso-dean-clifford.csv"  # Replace with your output CSV file

    with open(input_csv_file, "r", encoding="utf-8") as input_file, open(output_csv_file, "w", encoding="utf-8", newline="") as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)

        for row in csv_reader:
            original_text = row[0]+ " ", " " + row[1]+ " ", " " +row[2]+ " ", " " +row[3]
            stemmed_text = (original_text)
            csv_writer.writerow([stemmed_text])

    print("Stemming complete. Results written to", output_csv_file)


# Main program
if __name__ == "__main__":
    stemmer = PorterStemmer()
    
    # Specify the path to your text file containing root words
    root_words_file = "root_words.txt"
    
    # Read root words from the file
    root_words = read_root_words_from_csv(root_words_file)
    
    # User input
    user_input = input("Enter a sentence or paragraph: ")
    words = user_input.split()
    
    # Stemming
    stemmed_words = [stemmer.stem(word, root_words) for word in words]
    stemmed_sentence = " ".join(stemmed_words)
    
    # Display the stemmed text
    print("Stemmed text:")
    print(stemmed_sentence)

"""