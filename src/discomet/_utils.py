"""Shared helpers for the discomet package (not part of the public API)."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping, Optional, Sequence

from convokit import Speaker, Utterance

logger = logging.getLogger("discomet")


def check_same_length(**named_lists: Sequence) -> None:
    """Raise ``ValueError`` if the given sequences do not all have the same length."""
    lengths = {name: len(values) for name, values in named_lists.items()}
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Input lists must have the same length, got: {details}.")


def build_utterances(
    message_list: Sequence[str],
    speakers_list: Sequence[str],
    msgsid_list: Sequence[str],
    disc_id: str,
    replyto_list: Optional[Sequence[Optional[str]]] = None,
    timestamps_list: Optional[Sequence[Any]] = None,
) -> list[Utterance]:
    """Build ConvoKit utterances for one discussion.

    ConvoKit orders utterances by timestamp, so every utterance needs one. When
    ``timestamps_list`` is not given, synthetic timestamps are used: the position
    of the utterance in the input (0, 1, 2, ...). The input order is therefore
    treated as chronological order.

    Args:
        message_list: Text of each utterance.
        speakers_list: Speaker of each utterance.
        msgsid_list: Unique id of each utterance.
        disc_id: Identifier of the discussion.
        replyto_list: Id of the utterance each utterance replies to. ``None`` or an
            empty value marks an utterance with no parent. Every non-empty value
            must appear in ``msgsid_list``.
        timestamps_list: Optional sortable timestamps, one per utterance.

    Raises:
        ValueError: If list lengths differ, ids are not unique, or a reply-to id
            is unknown.
    """
    check_same_length(
        message_list=message_list,
        speakers_list=speakers_list,
        msgsid_list=msgsid_list,
    )
    if replyto_list is not None:
        check_same_length(message_list=message_list, replyto_list=replyto_list)
    if timestamps_list is not None:
        check_same_length(message_list=message_list, timestamps_list=timestamps_list)

    ids = [str(msg_id) for msg_id in msgsid_list]
    if len(set(ids)) != len(ids):
        raise ValueError("msgsid_list must contain unique message ids.")
    known_ids = set(ids)

    speakers = {str(s): Speaker(id=str(s)) for s in set(speakers_list)}
    utterances = []
    for i, (text, speaker, msg_id) in enumerate(zip(message_list, speakers_list, ids)):
        reply_to = None
        if replyto_list is not None:
            parent = replyto_list[i]
            if parent is not None and str(parent) != "":
                if str(parent) not in known_ids:
                    raise ValueError(
                        f"Utterance '{msg_id}' replies to unknown message id '{parent}'."
                    )
                reply_to = str(parent)
        timestamp = timestamps_list[i] if timestamps_list is not None else i
        utterances.append(
            Utterance(
                id=msg_id,
                speaker=speakers[str(speaker)],
                conversation_id=str(disc_id),
                reply_to=reply_to,
                timestamp=timestamp,
                text=str(text),
                meta={"timestamp": timestamp},
            )
        )
    return utterances


def _json_default(value: Any) -> Any:
    """Make numpy / pandas scalars and other objects JSON-serialisable."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_json(
    data: Mapping | Sequence,
    name: str,
    disc_id: str,
    output_dir: Optional[str],
) -> Optional[str]:
    """Write ``data`` to ``<output_dir>/<name>_<disc_id>_<timestamp>.json``.

    Does nothing and returns ``None`` when ``output_dir`` is ``None`` (the default
    everywhere in the package), so no function writes files unless asked to.
    """
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(output_dir, f"{name}_{disc_id}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, default=_json_default)
    logger.info("Saved %s", path)
    return path
