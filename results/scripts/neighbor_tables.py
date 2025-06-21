import json
import sys
from typing import TypeVar
import logging
import re
import pathlib
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass


def prefix_filename(filepath: str | Path, prefix: str, extension: str) -> Path:
    stripped_ext = extension.removeprefix(".")
    path = Path(filepath)
    return path.parent / Path(f"{prefix}_{path.stem}.{stripped_ext}")


def get_launch_file():
    return sys.argv[0]


def compare_neighbor_tables(
    base_path: str,
    neighbor_tables: list[tuple[str, str]],
    compare_func: Callable[[Path, Path], object],
    rerun: bool = False,
):
    """
    Iterate through pairs of files, executing a compare function on each pair and saving
    the results. Will skip running the comparison if it has already been run. Requires
    write access to the directory storing the data. Can be overridden to force a rerun.
    Writes results to base_path folder.
    """
    # Save results to a single file stored with the data
    results_path = Path(base_path) / f"neighbor_table_comparison_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for left_file, right_file in neighbor_tables:
        print(f"Comparing file: {left_file} with file: {right_file}")
        comparison_name = f"{left_file}, {right_file}"

        if not rerun and comparison_name in results:
            print("Files have already been compared.")
            continue

        left_path = pathlib.Path(base_path, left_file)
        right_path = pathlib.Path(base_path, right_file)

        pair_comparison = compare_func(left_path, right_path)

        results[comparison_name] = pair_comparison

        print("Comparison done")
        print(f"Results: {results}")

        # Save at intermediate steps in case crashes
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    print("Final comparison results")
    print(results)


# Generic type for type checking
ParsedValue = TypeVar("ParsedValue")


def parse_gds_line(
    line: str, value_type: str, converter: Callable[[str], ParsedValue]
) -> tuple[int, list[ParsedValue]]:
    """
    Parses a given line in GDS_Join output. Given a value_type like neighbors, or distances,
    and a Callable to convert each individual string to a parsed value, returns
    an ordered list of the values.
    """
    m = re.search(rf"^point id: (\d+), {value_type}: (.+),$", line)
    if m:
        point_index_string, values_string = m.group(1, 2)
        point_index = int(point_index_string)

        split_values_string = values_string.split(",")
        values = [converter(x) for x in split_values_string]
        logging.debug(f"{value_type}: {values}")

        return point_index, values
    else:
        raise Exception(f"Failed to read line from gds_join: {line}")


def parse_gds_neighbor_line(line: str) -> tuple[int, list[int]]:
    return parse_gds_line(line, "neighbors", lambda x: int(x))


def parse_gds_distance_line(line: str) -> tuple[int, list[float]]:
    return parse_gds_line(line, "distances", lambda x: float(x))


def parse_fasted_line(line: str) -> tuple[int, int, float | None]:
    m = re.search(r"^(\d+), (\d+)(, (.+))?$", line)
    if m:
        query_s, cand_s, dist_s = m.group(1, 2, 4)
        query, cand = int(query_s), int(cand_s)

        if dist_s is not None:
            dist = float(dist_s)
        else:
            dist = None

        return query, cand, dist
    else:
        raise Exception(f"Failed to parse line from fasted neighbor table: {line}")


@dataclass
class Neighbor:
    point_index: int
    distance: float | None


def get_fasted_neighbors_for_point(mptc_file, point_index: int) -> list[Neighbor]:
    """
    Have to read through current position of mptc_file line by line to build up
    the list of guesses and associated distances. Backtrack if we read too far.
    """
    # Build set for candidate guesses
    guessed_neighbors = []
    while True:
        # Save old position
        mptc_file_pos = mptc_file.tell()  # Save line in case we need to backtrack

        line = mptc_file.readline()
        # Handle last line in file
        if not line:
            logging.debug(
                f"Guessed neighbors for point {point_index} was\n{guessed_neighbors}"
            )
            return guessed_neighbors

        query, cand, dist = parse_fasted_line(line)
        if query < point_index:
            # Somehow we are still on an old point?? should never happen
            raise Exception("Somehow we seem to have skipped a point")
        elif query == point_index:
            guessed_neighbors.append(Neighbor(cand, dist))
        else:
            # Seek back since we overshot
            mptc_file.seek(mptc_file_pos)
            logging.debug(
                f"Guessed neighbors for point {point_index} was\n{guessed_neighbors}"
            )
            return guessed_neighbors
