"""Linguistic style coordination (Danescu-Niculescu-Mizil et al., 2012), via ConvoKit."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from convokit import Coordination, Corpus

from .._utils import build_utterances, logger, save_json


def _coordination_summaries(utterances: Sequence, speaker_ids: Sequence[str]) -> dict:
    """Fit ConvoKit's Coordination on ``utterances`` and summarise it per speaker."""
    corpus = Corpus(utterances=list(utterances))
    coord = Coordination()
    coord.fit(corpus)
    coord.transform(corpus)

    def everyone(_speaker) -> bool:
        return True

    to_user: dict[str, Any] = {}
    from_user: dict[str, Any] = {}
    for user in speaker_ids:

        def is_user(speaker, user=user) -> bool:
            return speaker.id == user

        logger.info("Coordination of all speakers towards user %s", user)
        to_user[user] = coord.summarize(
            corpus,
            everyone,
            is_user,
            focus="targets",
            target_thresh=1,
            summary_report=True,
        )
        logger.info("Coordination of user %s towards all speakers", user)
        from_user[user] = coord.summarize(
            corpus,
            is_user,
            everyone,
            focus="speakers",
            speaker_thresh=1,
            target_thresh=1,
            summary_report=True,
        )
    return {
        "coord_allspeakers_2_user": to_user,
        "coord_user_2_allspeaker": from_user,
    }


def linguistic_coordination(
    message_list: Sequence[str],
    speakers_list: Sequence[str],
    msgsid_list: Sequence[str],
    replyto_list: Sequence[Optional[str]],
    disc_id: str,
    discussion_level: bool,
    *,
    timestamps_list: Optional[Sequence[Any]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Measure how much speakers mirror each other's linguistic style.

    Uses ConvoKit's ``Coordination``, which implements Danescu-Niculescu-Mizil et
    al. (2012), "Echoes of power: Language effects and power differences in
    social interaction". For each speaker it reports coordination from all other
    speakers towards that speaker, and from that speaker towards all others.

    Args:
        message_list: Text of each utterance, in chronological order.
        speakers_list: Speaker of each utterance.
        msgsid_list: Unique id of each utterance.
        replyto_list: Id of the utterance each one replies to (``None`` for the
            root). Every id must appear in ``msgsid_list``.
        disc_id: Identifier of the discussion.
        discussion_level: If True, compute once over the full discussion;
            otherwise recompute over each growing prefix of the discussion.
        timestamps_list: Optional sortable timestamps. If omitted, the position in
            the input list is used, i.e. the input order is taken as chronological.
        output_dir: If given, the result is also saved there as JSON. Nothing is
            written by default.

    Returns:
        ``{disc_id: {"coord_allspeakers_2_user": {...}, "coord_user_2_allspeaker":
        {...}}}`` when ``discussion_level`` is True. Otherwise
        ``{disc_id: {"<first_id>-<last_id>": <same structure>, ...}}`` with one
        entry per prefix of two or more utterances.

    Raises:
        ValueError: If the input lists are inconsistent (lengths, duplicate ids,
            unknown reply-to ids).
    """
    utterances = build_utterances(
        message_list, speakers_list, msgsid_list, disc_id, replyto_list, timestamps_list
    )
    speaker_ids = sorted({utt.speaker.id for utt in utterances})

    if discussion_level:
        result = {disc_id: _coordination_summaries(utterances, speaker_ids)}
        save_json(result, "coord_per_discussion", disc_id, output_dir)
        return result

    windows = {}
    for end in range(1, len(utterances)):
        prefix = utterances[: end + 1]
        prefix_speakers = sorted({utt.speaker.id for utt in prefix})
        key = f"{utterances[0].id}-{utterances[end].id}"
        windows[key] = _coordination_summaries(prefix, prefix_speakers)
    result = {disc_id: windows}
    save_json(result, "coord_per_utt", disc_id, output_dir)
    return result
