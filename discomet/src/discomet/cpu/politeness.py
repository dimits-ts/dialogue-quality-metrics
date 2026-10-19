"""Politeness strategies (Danescu-Niculescu-Mizil et al., 2013), via ConvoKit."""

from __future__ import annotations

from typing import Optional, Sequence

from convokit import Corpus, PolitenessStrategies, TextParser

from .._utils import build_utterances, logger, save_json


def politeness_strategies(
    message_list: Sequence[str],
    speakers_list: Sequence[str],
    msgsid_list: Sequence[str],
    disc_id: str,
    discussion_level: bool,
    *,
    output_dir: Optional[str] = None,
) -> dict:
    """Detect politeness strategies (markers) in a discussion.

    Uses ConvoKit's ``PolitenessStrategies`` with its default ``politeness_api``
    collection, which implements the strategies of Danescu-Niculescu-Mizil et al.
    (2013), "A computational approach to politeness with application to social
    factors". Requires the spaCy model ``en_core_web_sm``
    (``python -m spacy download en_core_web_sm``).

    Args:
        message_list: Text of each utterance.
        speakers_list: Speaker of each utterance.
        msgsid_list: Unique id of each utterance.
        disc_id: Identifier of the discussion.
        discussion_level: If True, return the share of utterances using each
            strategy over the whole discussion; otherwise return the strategy
            indicators (0 or 1) of each utterance.
        output_dir: If given, the result is also saved there as JSON. Nothing is
            written by default.

    Returns:
        ``{disc_id: {strategy: share}}`` when ``discussion_level`` is True, else
        ``{disc_id: [{message_id: {strategy: 0.0 or 1.0}}, ...]}``.

    Raises:
        ValueError: If the input lists have different lengths or ids repeat.
        OSError: If the spaCy model is not installed.
    """
    utterances = build_utterances(message_list, speakers_list, msgsid_list, disc_id)
    corpus = Corpus(utterances=utterances)
    logger.info("Politeness: corpus of %d utterances created", len(utterances))

    try:
        corpus = TextParser().transform(corpus)
    except OSError as err:
        raise OSError(
            "The spaCy model 'en_core_web_sm' is required. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from err

    transformer = PolitenessStrategies()
    corpus = transformer.transform(corpus, markers=True)

    if discussion_level:
        summary = transformer.summarize(corpus, plot=False)
        result = {disc_id: {k: float(v) for k, v in dict(summary).items()}}
        save_json(result, "politeness_per_disc", disc_id, output_dir)
        return result

    # The corpus is parsed once; each utterance carries its own strategy indicators.
    per_utterance = []
    for source_utt in utterances:  # keep the caller's order
        utt = corpus.get_utterance(source_utt.id)
        strategies = utt.meta[transformer.strategy_attribute_name]
        per_utterance.append({utt.id: {k: float(v) for k, v in strategies.items()}})
    result = {disc_id: per_utterance}
    save_json(result, "politeness_per_utt", disc_id, output_dir)
    return result
