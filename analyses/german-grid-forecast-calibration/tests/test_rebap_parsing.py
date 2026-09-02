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


# Copied from the live endpoint's response, not written from the documentation.
# The first version of this fixture was invented from the spec and had both the
# wrong columns and the wrong timezone, which is the argument for fixtures taken
# from a real response.
SAMPLE = "\n".join([
    "Datum;Zeitzone;von;bis;Datenkategorie;Datentyp;Einheit;reBAP unterdeckt;reBAP ueberdeckt",
    "01.07.2025;UTC;00:00;00:15;reBAP;Qualitaetsgesichert;EUR/MWh;148,03;148,03",
    "01.07.2025;UTC;00:15;00:30;reBAP;Qualitaetsgesichert;EUR/MWh;75,09;75,09",
    "01.07.2025;UTC;00:30;00:45;reBAP;Qualitaetsgesichert;EUR/MWh;-45,30;-45,30",
    "01.07.2025;UTC;00:45;01:00;reBAP;Qualitaetsgesichert;EUR/MWh;1.234,56;1.234,56",
    "",
])


def test_comma_is_a_decimal_point_not_a_thousands_separator():
    d = parse_rebap(SAMPLE)
    assert d["short"].iloc[0] == pytest.approx(148.03)
    assert d["short"].iloc[1] == pytest.approx(75.09)


def test_negative_prices_survive():
    # Negative imbalance prices are ordinary in a system with this much wind and
    # solar, and a parser that drops the sign would flip the adverse direction.
    d = parse_rebap(SAMPLE)
    assert d["short"].iloc[2] == pytest.approx(-45.30)


def test_dotted_thousands_are_stripped():
    d = parse_rebap(SAMPLE)
    assert d["short"].iloc[3] == pytest.approx(1234.56)


def test_stamps_are_utc_not_berlin():
    # The Zeitzone column says UTC. Localising to Berlin would shift every price
    # by an hour or two and mis-price the summer half of the year against the
    # winter half.
    d = parse_rebap(SAMPLE)
    first = d.index[0]
    assert (first.year, first.month, first.day, first.hour) == (2025, 7, 1, 0)
    assert str(first.tz) == "UTC"


def test_both_price_columns_are_read():
    # unterdeckt is what a short balancing group pays, ueberdeckt what a long one
    # receives. They are equal under the uniform reBAP and read separately anyway.
    d = parse_rebap(SAMPLE)
    assert list(d.columns) == ["short", "long"]
    assert d["long"].iloc[0] == pytest.approx(148.03)


def test_a_non_utc_response_is_refused_rather_than_guessed():
    bad = SAMPLE.replace(";UTC;", ";MEZ;")
    with pytest.raises(SystemExit):
        parse_rebap(bad)


def test_index_is_quarter_hourly_and_ordered():
    d = parse_rebap(SAMPLE)
    steps = pd.Series(d.index).diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(minutes=15)]
    assert d.index.is_monotonic_increasing
