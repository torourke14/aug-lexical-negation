from typing import Dict, Any, List, Set, Tuple
import collections

import spacy
from spacy.tokens import Token, Doc, Span
from spacy.matcher import DependencyMatcher, Matcher

def _ensure_extensions() -> None:
    """Register token/span custom extensions if not already present."""
    if not Token.has_extension("is_neg_cue"):
        Token.set_extension("is_neg_cue", default=False, force=True)
    if not Token.has_extension("neg_scope_id"):
        Token.set_extension("neg_scope_id", default=None, force=True)

    if not Span.has_extension("is_neg_scope"):
        Span.set_extension("is_neg_scope", default=False, force=True)
    if not Span.has_extension("neg_scope_id"):
        Span.set_extension("neg_scope_id", default=None, force=True)


_ensure_extensions()


NLP = spacy.load("en_core_web_sm")

NEG_SYNTAX = {"not", "n't", "never", "no"}
NEG_PRONOUNS = {"nobody", "nothing", "none", "nowhere", "neither"}
NEG_VERBS = {"deny", "refuse", "lack", "fail"}
NEG_ADPOSITIONS = {"without"}

REGION_ROOT = "root"                # main verb/predicate
REGION_SUBJECT = "subject"          # no X vs X
REGION_OBJECT = "object"            
REGION_LOCATIVE = "locative"
REGION_ATTRIBUTE = "attribute"
REGION_MODAL = "modal"
REGION_QUANTIFIER = "quantifier"

# https://spacy.io/api/matcher
# define lowercase lexical multi-token negators
MATCHER = Matcher(NLP.vocab)
MATCHER.add(
    "NEVER_EVER",
    [  [{"LOWER": "never"}, {"LOWER": "ever"}]  ],
)
MATCHER.add(
    "IN_NO_WAY",
    [  [{"LOWER": "in"}, {"LOWER": "no"}, {"LOWER": "way"}]  ],
)

# DependencyMatcher: negator modifying a verb with an object
DEP_MATCHER = DependencyMatcher(NLP.vocab)
NEG_VERB_OBJ_PATTERN = [
    { 
        "RIGHT_ID": "verb", 
        "RIGHT_ATTRS": { "POS": "VERB" } 
    },
    {
        "RIGHT_ID": "neg",
        "RIGHT_ATTRS": {"DEP": "neg"},
        "LEFT_ID": "verb",
        "REL_OP": ">>",
    },
    {
        "RIGHT_ID": "obj",
        "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj"]}},
        "LEFT_ID": "verb",
        "REL_OP": ">>",
    },
]
DEP_MATCHER.add("NEG_VERB_OBJ", [NEG_VERB_OBJ_PATTERN])


### Helper Functions ###
#####################################################################

def _is_rule_out_verb(token: Token) -> bool:
    """ 'Rule out' as a verbal negation cue."""
    if token.lemma_.lower() != "rule":
        return False
    if not token.pos_.startswith("V"):
        return False
    for child in token.children: # for words preceding or following "rule"
        if child.lemma_.lower() == "out" and child.dep_ in {"prt", "advmod"}:
            return True
    return False


def _find_neither_nor_anchor(sent: Span) -> Token | None:
    """
    Detect 'neither ... nor', return the 'head' of 'neither',
    None in all other cases
    """
    tokens = list(sent)
    neither_tokens = [t for t in tokens if t.lemma_.lower() == "neither"]
    if not neither_tokens:
        return None
    if not any(t.lemma_.lower() == "nor" for t in tokens):
        return None

    t = neither_tokens[0]
    head = t.head
    if head in sent and head.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}:
        return head
    return t

def get_without_pobj(token: Token) -> Token | None:
    """ For 'without', return its object if present. """
    if token.lemma_.lower() != "without" or token.pos_ != "ADP":
        return None
    for child in token.children:
        if child.dep_ == "pobj":
            return child
    return None


### Region/scope classification ###
#######################################################################

def _classify_region(anchor: Token, sent_root: Token) -> Set[str]:
    """ Classify an *anchor* token into one or more NPAS++ regions.

    anchor: the token that is being negated (e.g., "eating" in "not really eating breakfast")
    sent_root: the ROOT token of the sentence span this anchor belongs to.
    """
    regions: Set[str] = set()
    lemma = anchor.lemma_.lower()
    dep = anchor.dep_
    pos = anchor.pos_

    # ROOT region: main predicate or copular predicate.
    if anchor == sent_root:
        regions.add(REGION_ROOT)
    else:
        # For copular sentences like "The boy is not happy":
        #   ROOT is "is", predicate is "happy" with dep "acomp".
        if dep in {"acomp", "ROOT"} and sent_root.lemma_ in {"be", "seem", "become"}:
            regions.add(REGION_ROOT)
    # Subject region.
    if dep in {"nsubj", "nsubjpass"}:
        regions.add(REGION_SUBJECT)
    # Object/complement region.
    if dep in {"dobj", "obj", "iobj", "attr", "oprd", "xcomp", "ccomp"}:
        regions.add(REGION_OBJECT)
    # Verb that governs an object-like child.
    if pos.startswith("V"):
        for child in anchor.children:
            if child.dep_ in {"xcomp", "ccomp", "dobj", "obj"}:
                regions.add(REGION_OBJECT)
                break
    # Locative / prepositional region (prepositions and their objects).
    if dep in {"prep", "pobj"} or pos == "ADP":
        regions.add(REGION_LOCATIVE)
    else:
        # Adjectives/adverbs that are inherently locative.
        if lemma in {"inside", "outside", "indoors", "outdoors"}:
            regions.add(REGION_LOCATIVE)
    # Attribute / adjectival modifiers.
    if pos == "ADJ" or dep in {"amod", "acomp", "advmod"}:
        regions.add(REGION_ATTRIBUTE)
    # Modals / auxiliaries (can, must, aux verbs).
    if dep == "aux" or anchor.tag_ == "MD":
        regions.add(REGION_MODAL)
    # Quantifiers and numbers (all, some, no, numerals).
    if dep in {"det", "nummod", "quantmod"} or anchor.like_num:
        regions.add(REGION_QUANTIFIER)
    else:
        if lemma in {"all", "no", "none", "some", "many", "few"}:
            regions.add(REGION_QUANTIFIER)
    return regions


def _scope_for_anchor(anchor: Token, sent: Span) -> Span:
    """
    Get the "negation scope" as the anchor.subtree for an anchor using anchor.subtree
    - clip the subtree to the sentence boundaries
    - return a Span representing the negation scope for the anchor.
    """
    subtree_tokens = list(anchor.subtree)
    if not subtree_tokens:
        print("[NPAS] empty subtree for anchor:", anchor.text)
        return sent

    start = max(min(t.i for t in subtree_tokens), sent.start)
    end = min(max(t.i for t in subtree_tokens) + 1, sent.end)
    return anchor.doc[start:end]


### Sentence-level negation analysis
########################################################################

def _analyze_sentence(sent: Span, 
                      neg_verb_obj_verbs: Set[int],
                      scope_id_start: int,
) -> Tuple[List[Dict[str, Any]], collections.Counter, int]:
    """ Where we determine negation cues and NPAS++ regions in one span of text.

        sent: a Span representing one sentence from doc.sents
        neg_verb_obj_verbs: set of token indices that match NEG_VERB_OBJ pattern
        scope_id_start: integer counter to assign unique IDs to scope spans

        Returns:
        - cues: list of detailed cue dicts
        - region_counter: Counter(region >> count)
        - next_scope_id: updated scope id counter
    """
    sent_root: Token = sent.root
    doc: Doc = sent.doc

    cues: List[Dict[str, Any]] = []
    region_counter: collections.Counter = collections.Counter()
    scope_id = scope_id_start

    # Multi-token lexical cues
    # Matcher runs on the entire doc, so we filter matches to those fully inside this sentence.
    for match_id, start, end in MATCHER(doc):
        if not (sent.start <= start < end <= sent.end):
            continue
        span = doc[start:end]          # the matched phrase
        anchor = span.root             # syntactic head of the phrase
        regions = _classify_region(anchor, sent_root)
        scope_span = _scope_for_anchor(anchor, sent)

        # Mark token/span extensions so we can later trace scopes in the doc.
        # We know these spans are negation based on the matcher set
        anchor._.is_neg_cue = True

        # set scope span extensions
        scope_span._.is_neg_scope = True
        scope_span._.neg_scope_id = scope_id
        for tok in scope_span:
            tok._.neg_scope_id = scope_id

        for r in regions:
            region_counter[r] += 1

        cues.append({
            "cue": span.text,
            "cue_lemma": span.root.lemma_.lower(),
            "cue_span_start": start,
            "cue_span_end": end,
            "anchor": anchor.text,
            "anchor_lemma": anchor.lemma_.lower(),
            "anchor_dep": anchor.dep_,
            "anchor_head": anchor.head.text,
            "scope_text": scope_span.text,
            "scope_start": scope_span.start,
            "scope_end": scope_span.end,
            "regions": sorted(regions),
            "scope_id": scope_id,
            "matcher_label": NLP.vocab.strings[match_id],
        })
        scope_id += 1

    # Single-token cues
    # run same analysis, this time checking for negation based on syntax and POS usage
    for token in sent:
        lemma = token.lemma_.lower()
        pos = token.pos_
        dep = token.dep_

        anchor: Token | None = None

        # Basic syntactic negation ("not", "never", "n't")
        if dep == "neg":
            # token.head is the verb or predicate being negated.
            anchor = token.head
        # no dogs
        elif lemma == "no" and pos == "DET":
            anchor = token.head
        # Negative pronouns (nobody, nothing)
        elif lemma in NEG_PRONOUNS and pos in {"PRON", "NOUN"}:
            anchor = token
        # Negative verbs (deny, refuse, lack, fail).
        elif lemma in NEG_VERBS and pos.startswith("V"):
            anchor = token
        # "rule out" multiword cue
        elif _is_rule_out_verb(token):
            anchor = token
        # Preposition "without" — anchor on its object if present.
        without_pobj = get_without_pobj(token)
        if without_pobj is not None:
            anchor = without_pobj

        if anchor is None:
            continue

        regions = _classify_region(anchor, sent_root)
        scope_span = _scope_for_anchor(anchor, sent)

        token._.is_neg_cue = True
        scope_span._.is_neg_scope = True
        scope_span._.neg_scope_id = scope_id
        for tok in scope_span:
            tok._.neg_scope_id = scope_id

        for r in regions:
            region_counter[r] += 1

        cues.append({
            "cue": token.text,
            "cue_lemma": lemma,
            "cue_span_start": token.i,
            "cue_span_end": token.i + 1,
            "anchor": anchor.text,
            "anchor_lemma": anchor.lemma_.lower(),
            "anchor_dep": anchor.dep_,
            "anchor_head": anchor.head.text,
            "scope_text": scope_span.text,
            "scope_start": scope_span.start,
            "scope_end": scope_span.end,
            "regions": sorted(regions),
            "scope_id": scope_id,
            "matcher_label": None
        })
        scope_id += 1

    # Lastly check for "neither ... nor" sentence level cue
    neither_nor_anchor = _find_neither_nor_anchor(sent)
    if neither_nor_anchor is not None:
        anchor = neither_nor_anchor
        regions = _classify_region(anchor, sent_root)
        scope_span = _scope_for_anchor(anchor, sent)

        anchor._.is_neg_cue = True
        scope_span._.is_neg_scope = True
        scope_span._.neg_scope_id = scope_id
        for tok in scope_span:
            tok._.neg_scope_id = scope_id

        for r in regions:
            region_counter[r] += 1

        cues.append({
            "cue": "neither/nor",
            "cue_lemma": "neither/nor",
            "cue_span_start": anchor.i,
            "cue_span_end": anchor.i + 1,
            "anchor": anchor.text,
            "anchor_lemma": anchor.lemma_.lower(),
            "anchor_dep": anchor.dep_,
            "anchor_head": anchor.head.text,
            "scope_text": scope_span.text,
            "scope_start": scope_span.start,
            "scope_end": scope_span.end,
            "regions": sorted(regions),
            "scope_id": scope_id,
            "matcher_label": "NEITHER_NOR",
        })
        scope_id += 1

    return cues, region_counter, scope_id


### Top-level analysis (doc-level) ###
########################################################################

def _analyze_negation_regions(text: str) -> Dict[str, Any]:
    """ Analyze negation in a given text, classifying specific regions.

        High-level flow:
            - run spaCy pipeline -> Doc
            - ensure token/span extensions are registered
            - run DependencyMatcher to detect certain patterns
            - iterate over doc.sents and call _analyze_sentence on each
            - aggregate region counts and cue info

        Return: {
            "has_negation": bool,
            "negation_cues": [ ... detailed cue dicts ... ],
            "region_counts": {region: int},
            "region_flags": {region: bool},
        }
    """
    _ensure_extensions()
    
    doc = NLP(text)
    if not doc:
        return {
            "has_negation": False,
            "negation_cues": [],
            "region_counts": {},
            "region_flags": {},
        }

    # Run dependency-based pattern once on the whole doc
    # spaCy populates doc sents using the parser
    neg_verb_obj_verbs: Set[int] = set()
    for _, token_ids in DEP_MATCHER(doc):
        # token_ids are indices into doc; first one is the verb.
        verb_idx = token_ids[0]
        neg_verb_obj_verbs.add(verb_idx)

    all_cues: List[Dict[str, Any]] = []
    global_region_counter: collections.Counter = collections.Counter()
    scope_id = 0

    # Analyze each sentence separately
    for sent in doc.sents:
        cues, region_counter, scope_id = _analyze_sentence(
            sent,
            neg_verb_obj_verbs,
            scope_id,
        )
        all_cues.extend(cues)
        for r, c in region_counter.items():
            global_region_counter[r] += c

    has_negation = len(all_cues) > 0
    region_flags = { reg: (cnt > 0) for reg, cnt in global_region_counter.items() }

    return {
        "has_negation": has_negation,
        "negation_cues": all_cues,
        "region_counts": dict(global_region_counter),
        "region_flags": region_flags,
    }


def analyze_pair_npas(premise: str, hypothesis: str) -> Dict[str, Any]:
    """ Compute NPAS negation region for a (premise, hypothesis) pair """
    prem_info = _analyze_negation_regions(premise)
    hyp_info = _analyze_negation_regions(hypothesis)

    all_regions = {
        REGION_ROOT,
        REGION_SUBJECT,
        REGION_OBJECT,
        REGION_LOCATIVE,
        REGION_ATTRIBUTE,
        REGION_MODAL,
        REGION_QUANTIFIER,
    }

    premise_flags = { r: False for r in all_regions }
    premise_flags.update(prem_info.get("region_flags", {}))

    hypothesis_flags = { r: False for r in all_regions }
    hypothesis_flags.update(hyp_info.get("region_flags", {}))

    features = {
        "prem_has_negation": prem_info["has_negation"],
        "hyp_has_negation": hyp_info["has_negation"],
    }

    for r in sorted(all_regions):
        features[f"prem_neg_{r}"] = premise_flags[r]
        features[f"hyp_neg_{r}"] = hypothesis_flags[r]

    return features