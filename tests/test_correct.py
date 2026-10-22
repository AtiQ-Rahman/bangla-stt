"""The guard around the local AI fix: it may re-spell and punctuate, never reword."""
from stt.correct import _chunks, _merge


def test_keeps_spelling_and_punctuation_fixes():
    assert _merge("আমি ভাত খাই ঢাকা যাব", "আমি ভাত খাই। ঢাকায় যাবো।") == ("আমি ভাত খাই। ঢাকায় যাবো।", 0)
    assert _merge("জন প্রশাসন মন্ত্রণালয়", "জনপ্রশাসন মন্ত্রণালয়ে") == ("জনপ্রশাসন মন্ত্রণালয়ে", 0)


def test_undoes_rewording_and_added_words():
    assert _merge("তিনি প্রথম আলোকে বলেন", "তিনি জানিয়েছেন") == ("তিনি প্রথম আলোকে বলেন", 1)
    assert _merge("এটা হয়", "এটা সব সময়ই হয়") == ("এটা হয়", 1)


def test_undoes_foreign_script():
    assert _merge("রাষ্টাচার ভেঙে", "ראשתארץ ভেঙে")[0] == "রাষ্টাচার ভেঙে"
    assert _merge("we went to dhaka", "We went to ঢাকা.", "English")[0] == "We went to dhaka"


def test_english_capitals_are_fine():
    assert _merge("so i went home", "So I went home", "English") == ("So I went home", 0)


def test_chunks_stretch_to_the_next_sentence_end():
    words = ("a " * 84 + "end. b c d").split()  # the sentence ends just after 80 words
    first = next(_chunks(words, 80))
    assert first.endswith("end.") and len(first.split()) == 85
