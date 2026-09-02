"""Parsing the German CSV conventions.

A silent mis-parse here moves every euro figure in the German report and nothing
downstream notices: a comma decimal read as a thousands separator turns 103,97
into 10397, which is a plausible-looking price in a stressed quarter-hour. These
run without credentials, which is the point: the parse can be wrong long before
anyone has an API key.
"""

import pandas as pd
import pytest

from pull_data_de import parse_rebap


SAMPLE = (
    "Datum;von;bis;reBAP;Qualitaet;Einheit\n"
    "01.07.2025;00:00;00:15;103,97;Qualitaetsgesichert;EUR/MWh\n"
    "01.07.2025;00:15;00:30;104,06;Qualitaetsgesichert;EUR/MWh\n"
    "01.07.2025;00:30;00:45;-45,30;Qualitaetsgesichert;EUR/MWh\n"
    "01.07.2025;00:45;01:00;1.234,56;Qualitaetsgesichert;EUR/MWh\n"
)


def test_comma_is_a_decimal_point_not_a_thousands_separator():
    d = parse_rebap(SAMPLE)
    assert d["reBAP"].iloc[0] == pytest.approx(103.97)
    assert d["reBAP"].iloc[1] == pytest.approx(104.06)


def test_negative_prices_survive():
    # Negative imbalance prices are ordinary in a system with this much wind and
    # solar, and a parser that drops the sign would flip the adverse direction.
    d = parse_rebap(SAMPLE)
    assert d["reBAP"].iloc[2] == pytest.approx(-45.30)


def test_dotted_thousands_are_stripped():
    d = parse_rebap(SAMPLE)
    assert d["reBAP"].iloc[3] == pytest.approx(1234.56)


def test_dates_are_day_first_and_localised_to_berlin():
    d = parse_rebap(SAMPLE)
    first = d.index[0]
    assert (first.year, first.month, first.day) == (2025, 7, 1)
    assert str(first.tz) == "Europe/Berlin"


def test_index_is_quarter_hourly_and_ordered():
    d = parse_rebap(SAMPLE)
    steps = pd.Series(d.index).diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(minutes=15)]
    assert d.index.is_monotonic_increasing
