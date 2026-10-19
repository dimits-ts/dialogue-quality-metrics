"""Text rendering of a discussion's reply structure, via ConvoKit."""

from __future__ import annotations

import contextlib
import io
from typing import Any, Optional, Sequence

from convokit import Corpus
from matplotlib.figure import Figure

from .._utils import build_utterances


def plot_conversation_structure(
    message_list: Sequence[str],
    speakers_list: Sequence[str],
    msgsid_list: Sequence[str],
    replyto_list: Sequence[Optional[str]],
    disc_id: str,
    *,
    timestamps_list: Optional[Sequence[Any]] = None,
    output_path: Optional[str] = None,
) -> Figure:
    """Draw the reply tree of a discussion, one speaker id per line.

    The tree is ConvoKit's ``Conversation.print_conversation_structure`` output,
    rendered as monospaced text so indentation lines up. This is a plain
    structure view, not a dominance or turn-taking dashboard.

    Args:
        message_list: Text of each utterance, in chronological order.
        speakers_list: Speaker of each utterance.
        msgsid_list: Unique id of each utterance.
        replyto_list: Id of the utterance each one replies to (``None`` for the
            root). Every id must appear in ``msgsid_list``.
        disc_id: Identifier of the discussion.
        timestamps_list: Optional sortable timestamps. If omitted, the position in
            the input list is used.
        output_path: If given, the figure is also saved to this path (for example
            ``"tree.png"``). Nothing is written by default.

    Returns:
        A matplotlib ``Figure``. It is not attached to pyplot, so no global
        matplotlib state is touched; display or save it as usual.

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
    corpus = Corpus(utterances=utterances)
    conversation = corpus.get_conversation(str(disc_id))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        conversation.print_conversation_structure(lambda utt: utt.speaker.id)
    text = buffer.getvalue()

    n_lines = max(1, text.count("\n"))
    width = max((len(line) for line in text.splitlines()), default=10)
    fig = Figure(figsize=(max(4.0, 0.12 * width + 1), max(2.0, 0.25 * n_lines + 0.5)))
    ax = fig.add_subplot(111)
    ax.text(
        0,
        1,
        text,
        fontsize=10,
        family="monospace",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.axis("off")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
