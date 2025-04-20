from collections import Counter
import math


def cosine_similarity(s1: str, s2: str) -> float:
    """
    Calculates the cosine similarity between two strings using a bag-of-words model.

    Parameters:
      s1: The first string.
      s2: The second string.

    Returns:
      Cosine similarity, ranging from [0, 1].
    """
    # Split strings into words and construct frequency vectors
    words1 = s1.split()
    words2 = s2.split()
    vec1 = Counter(words1)
    vec2 = Counter(words2)

    # Get the union of all words
    all_words = set(vec1.keys()).union(vec2.keys())

    # Calculate the dot product of the vectors
    dot_product = sum(vec1[word] * vec2[word] for word in all_words)

    # Calculate the norms of the vectors
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculates the Jaccard similarity between two strings based on word sets.

    Parameters:
      s1: The first string.
      s2: The second string.

    Returns:
      Jaccard similarity, ranging from [0, 1].
    """
    # Split strings into words and construct sets
    set1 = set(s1.split())
    set2 = set(s2.split())

    # Calculate the size of the intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0
    return len(intersection) / len(union)


def euclidean_distance(s1: str, s2: str) -> float:
    """
    Calculates the Euclidean similarity between two strings using a bag-of-words model.
    Note: A higher similarity value indicates greater similarity between the strings.

    Parameters:
      s1: The first string.
      s2: The second string.

    Returns:
      Euclidean similarity, ranging from [0, 1].
    """
    # Split strings into words and construct frequency vectors
    words1 = s1.split()
    words2 = s2.split()
    vec1 = Counter(words1)
    vec2 = Counter(words2)

    # Get the union of all words to construct the complete feature set
    all_words = set(vec1.keys()).union(vec2.keys())

    # Calculate the squared sum of Euclidean distances (then take the square root)
    distance_sq = sum((vec1[word] - vec2[word]) ** 2 for word in all_words)
    distance = math.sqrt(distance_sq)

    # Convert to similarity using the formula 1 / (1 + distance)
    return 1 / (1 + distance)