"""Reciprocity motifs (Zhang et al., 2018, CSCW), via ConvoKit's HyperConvo."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from convokit import Corpus, HyperConvo

from .._utils import build_utterances, logger, save_json


def _motif_counts(
    utterances: Sequence, disc_id: str, prefix_len: int, min_convo_len: int
) -> dict[str, float]:
    """Return the HyperConvo motif-count features of one discussion."""
    corpus = Corpus(utterances=list(utterances))
    HyperConvo(prefix_len=prefix_len, min_convo_len=min_convo_len).transform(
        corpus
    )
    features = corpus.get_vector_matrix("hyperconvo").to_dataframe().fillna(-1)
    # Keep motif counts; drop the "mid" (mid-thread) variants, as the original package did.
    motif_columns = [
        c for c in features.columns if "count" in c and "mid" not in c
    ]
    by_conversation = features[motif_columns].transpose().to_dict()
    # Conversations shorter than min_convo_len are dropped by HyperConvo.
    return by_conversation.get(str(disc_id), {})


def reciprocity_motifs(
    message_list: Sequence[str],
    speakers_list: Sequence[str],
    msgsid_list: Sequence[str],
    replyto_list: Sequence[Optional[str]],
    disc_id: str,
    discussion_level: bool,
    *,
    prefix_len: int = 40,
    min_convo_len: int = 2,
    timestamps_list: Optional[Sequence[Any]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Count reciprocity and related interaction motifs in a discussion.

    Uses ConvoKit's ``HyperConvo``, which builds the hypergraph of participant
    interactions described by Zhang et al. (2018), "Characterizing online public
    discussions through patterns of participant interactions". Motifs include
    reciprocity, external reciprocity, dyadic interaction, and incoming and
    outgoing triads. Only the motif *count* features are returned.

    Args:
        message_list: Text of each utterance, in chronological order.
        speakers_list: Speaker of each utterance.
        msgsid_list: Unique id of each utterance.
        replyto_list: Id of the utterance each one replies to (``None`` for the
            root). Every id must appear in ``msgsid_list``.
        disc_id: Identifier of the discussion.
        discussion_level: If True, compute once over the full discussion;
            otherwise recompute over each growing prefix of the discussion.
        prefix_len: Only the first ``prefix_len`` utterances build the hypergraph.
        min_convo_len: Conversations shorter than this are skipped by ConvoKit.
        timestamps_list: Optional sortable timestamps. If omitted, the position in
            the input list is used, i.e. the input order is taken as chronological.
        output_dir: If given, the result is also saved there as JSON. Nothing is
            written by default.

    Returns:
        ``{disc_id: {motif: count}}`` when ``discussion_level`` is True. Otherwise
        ``{disc_id: {"<first_id>-<last_id>": {motif: count}, ...}}`` with one
        entry per prefix of two or more utterances. Features ConvoKit cannot
        compute are reported as -1.

    Raises:
        ValueError: If the input lists are inconsistent (lengths, duplicate ids,
            unknown reply-to ids).
    """
    utterances = build_utterances(
        message_list,
        speakers_list,
        msgsid_list,
        disc_id,
        replyto_list,
        timestamps_list,
    )
    logger.info("Reciprocity: %d utterances", len(utterances))

    if discussion_level:
        result = {
            disc_id: _motif_counts(
                utterances, disc_id, prefix_len, min_convo_len
            )
        }
        save_json(result, "reciprocity_per_discussion", disc_id, output_dir)
        return result

    windows = {}
    for end in range(1, len(utterances)):
        key = f"{utterances[0].id}-{utterances[end].id}"
        windows[key] = _motif_counts(
            utterances[: end + 1], disc_id, prefix_len, min_convo_len
        )
    result = {disc_id: windows}
    save_json(result, "reciprocity_per_utt", disc_id, output_dir)
    return result
