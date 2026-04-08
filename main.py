import time
from collections import deque, Counter

import character_check
import config
import screen_capture
import stage_check
import write_out

VOTE_WINDOW = 5   # frames to accumulate
VOTE_THRESHOLD = 3  # votes needed to confirm a result


def majority(dq):
    """Return the most common non-None value if it meets VOTE_THRESHOLD, else None."""
    counts = Counter(x for x in dq if x is not None)
    if not counts:
        return None
    winner, count = counts.most_common(1)[0]
    return winner if count >= VOTE_THRESHOLD else None


def main():
    character_check.load_templates()
    stage_check.load_index()
    last_p1, last_p2, last_stage = None, None, None

    p1_votes = deque(maxlen=VOTE_WINDOW)
    p2_votes = deque(maxlen=VOTE_WINDOW)
    stage_votes = deque(maxlen=VOTE_WINDOW)

    print("Running — press Ctrl+C to stop.")
    try:
        while True:
            frame = screen_capture.grab_frame()

            p1_crop = screen_capture.crop_region(frame, config.P1_REGION)
            p2_crop = screen_capture.crop_region(frame, config.P2_REGION)

            p1_raw, p1_conf = character_check.identify_character(p1_crop)
            p2_raw, p2_conf = character_check.identify_character(p2_crop)
            stage_raw, stage_conf = stage_check.identify_stage(frame)

            p1_votes.append(p1_raw)
            p2_votes.append(p2_raw)
            stage_votes.append(stage_raw)

            p1 = majority(p1_votes)
            p2 = majority(p2_votes)
            stage = majority(stage_votes)

            print(
                f"  raw: p1={p1_raw}({p1_conf:.2f}) p2={p2_raw}({p2_conf:.2f}) stage={stage_raw}({stage_conf:.2f})"
                f"  |  voted: p1={p1} p2={p2} stage={stage}"
            )

            changed = (
                p1 is not None
                and p2 is not None
                and (p1 != last_p1 or p2 != last_p2 or stage != last_stage)
            )
            if changed:
                print(
                    f"Detected: {p1} vs {p2} | {stage}"
                )
                # write_out.type_characters(p1, p2, stage)
                last_p1, last_p2, last_stage = p1, p2, stage

            time.sleep(config.POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        screen_capture.release()


if __name__ == "__main__":
    main()
