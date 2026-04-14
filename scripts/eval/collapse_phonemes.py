"""PHONEME_MAP: dict[str, str] = {
    # English normalization
    # stripping suprasegmental information as well as nasalized and devoiced marking
    'l̩': 'l',       # syllabic l̩ -> l
    'm̩': 'm',       # syllabic m̩ -> m
    'n̩': 'n',       # syllabic n̩ -> n
    'ŋ̍': 'ŋ',       # syllabic ŋ̍ -> ŋ
    'ə̥': 'ə',       # devoiced ə̥ -> ə
    'ɾ̃': 'ɾ',       # nasal ɾ̃ -> ɾ
    
    # Irish normalization
    # stripping velarization, laminal, dental, and devoiced marking
    'bˠ': 'b',      # velarized bˠ -> b
    'd̪ˠ': 'd',      # velarized, dental d̪ˠ -> d
    'fˠ': 'f',      # velarized fˠ -> f
    'l̥ʲ': 'lʲ',     # devoiced l̥ʲ -> lʲ
    'l̻ʲ': 'lʲ',     # laminal l̻ʲ -> lʲ
    'l̻̊ˠ': 'l',      # laminal, devoiced l̻̊ˠ -> l
    'l̻ˠ': 'l',      # laminal l̻ˠ -> l
    'm̥ʲ': 'mʲ',     # devoiced m̥ʲ -> mʲ
    'mˠ': 'm',      # velarized mˠ -> m
    'm̥ˠ': 'm',      # velarized, devoiced m̥ˠ -> m
    'n̥ʲ': 'nʲ',     # devoiced n̥ʲ -> nʲ
    'n̻ʲ': 'nʲ',     # laminal n̻ʲ -> nʲ
    'nˠ': 'n',      # velarized nˠ -> n 
    'n̻̊ˠ': 'n',      # velarized, laminal, devoiced n̻̊ˠ -> n
    'n̻ˠ': 'n',      # laminal n̻ˠ -> n
    'pˠ': 'p',      # velarized pˠ -> p
    'sˠ': 's',      # velarized sˠ -> s
    't̪': 't',       # dental t̪ -> t
    't̪ˠ': 't',      # velarized, dental t̪ˠ -> t
    'vˠ': 'v',      # velarized vˠ -> v
    'zˠ': 'z',      # velarized zˠ -> z
    'ɲ̊': 'ɲ',       # devoiced ɲ̊ -> ɲ
    'ɾˠ': 'ɾ',      # velarized ɾˠ -> ɾ
    'ɾ̥ʲ': 'ɾʲ',     # devoiced ɾ̥ʲ -> ɾʲ
    'ɾ̥ˠ': 'ɾ',      # velarized, devoiced ɾ̥ˠ -> ɾ
}
def collapse_phones(phones: list[str], mapping: dict[str, str] = PHONEME_MAP) -> list[str]:
    # Apply PHONEME_MAP to a phone sequence before alignment.
    # Phones not in the mapping are left unchanged.
    
    return [mapping.get(p, p) for p in phones]
"""

STRIP_DIACRITICS = {
    'ˠ',  # velarization
    '̪',   # dental
    '̻',   # laminal
    '̥',   # devoiced
    '̃',   # nasalized
    '̩',   # syllabic
    '̍',   # syllabic (for characters with hangy bits)
}

def collapse_phones(phones: list[str]) -> list[str]:
    return [''.join(c for c in phone if c not in STRIP_DIACRITICS) for phone in phones]