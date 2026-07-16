"""Unit tests for xcom_dashboard.aggregator.compute_metrics.

All tests use hand-built fake state dicts — no live engine required.
Run with: py -m pytest python/tests/test_aggregator.py -v
"""

from __future__ import annotations

import sys
import os

# Make xcom_dashboard importable from the python/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xcom_dashboard.aggregator import (
    compute_metrics,
    _soldier_display,
    _armor_display,
    _type_to_display,
    SELL_PRICES,
    RANK_ORDER,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _base(
    name="Alpha",
    scientists=0,
    engineers=0,
    soldiers=None,
    crafts=None,
    stores=None,
    facilities=None,
    research=None,
    manufacture=None,
):
    return {
        "name": name,
        "lon": 0.2,
        "lat": 0.8,
        "scientists": scientists,
        "engineers": engineers,
        "soldiers": soldiers or [],
        "crafts": crafts or [],
        "stores": stores or {},
        "facilities": facilities or [],
        "research": research or [],
        "manufacture": manufacture or [],
        "transfers": [],
    }


def _soldier(name="John Smith", rank="STR_ROOKIE", armor="STR_NONE_UC"):
    return {
        "name": name,
        "rank": rank,
        "armor": armor,
        "tu": 50, "stamina": 50, "health": 30, "bravery": 60,
        "reactions": 40, "firing": 50, "throwing": 50, "strength": 30,
        "psi_strength": 0, "psi_skill": 0, "melee": 0,
        "wound_recovery": 0, "craft": None,
    }


def _facility(type_str="STR_LAB", labs=50, workshops=0, build_days=0):
    return {
        "type": type_str,
        "x": 0, "y": 0,
        "build_days": build_days,
        "capacity": {
            "personnel": 0, "storage": 0, "crafts": 0,
            "labs": labs, "workshops": workshops, "psi_labs": 0,
        },
    }


def _craft(type_str="STR_SKYRANGER", status="STR_READY"):
    return {
        "type": type_str,
        "id": 1,
        "name": type_str,
        "status": status,
        "fuel": 100, "fuel_max": 100,
        "damage": 0, "damage_max": 200,
        "soldiers": 0,
        "weapons": [],
    }


def _state(bases=None, funds=1_000_000, event="tick"):
    return {
        "type": "state",
        "event": event,
        "game": {
            "funds": funds,
            "day": 1, "month": 3, "year": 1999,
            "hour": 0, "minute": 0,
            "score": 0,
            "countries": [],
        },
        "bases": bases or [],
        "ufos": [],
        "mission_sites": [],
    }


# ---------------------------------------------------------------------------
# Metric 1: Cash
# ---------------------------------------------------------------------------

def test_cash_basic():
    st = _state(funds=5_000_000)
    result = compute_metrics(st)
    assert result["metrics"]["cash"] == 5_000_000


def test_cash_zero():
    st = _state(funds=0)
    assert compute_metrics(st)["metrics"]["cash"] == 0


# ---------------------------------------------------------------------------
# Metric 2: Soldiers
# ---------------------------------------------------------------------------

def test_soldiers_format_two_word_name():
    assert _soldier_display({"name": "John Smith"}) == "J. Smith"


def test_soldiers_format_three_word_name():
    # Only first initial + last word
    assert _soldier_display({"name": "Mary Jane Watson"}) == "M. Watson"


def test_soldiers_format_single_word():
    assert _soldier_display({"name": "Rambo"}) == "Rambo"


def test_armor_display_none():
    assert _armor_display({"armor": "STR_NONE_UC"}) == "None"


def test_armor_display_personal():
    assert _armor_display({"armor": "STR_PERSONAL_ARMOR_UC"}) == "Personal Armor"


def test_armor_display_power_suit():
    assert _armor_display({"armor": "STR_POWER_SUIT_UC"}) == "Power Suit"


def test_armor_display_flying():
    assert _armor_display({"armor": "STR_FLYING_SUIT_UC"}) == "Flying Suit"


def test_soldiers_sorted_by_rank_descending():
    base = _base(soldiers=[
        _soldier("Alice Roe",   rank="STR_ROOKIE"),
        _soldier("Bob Brown",   rank="STR_COLONEL"),
        _soldier("Carol Green", rank="STR_SERGEANT"),
        _soldier("Dave White",  rank="STR_CAPTAIN"),
    ])
    result = compute_metrics(_state(bases=[base]))
    names = [s["display"] for s in result["metrics"]["soldiers"]]
    assert names == ["B. Brown", "D. White", "C. Green", "A. Roe"]


def test_soldiers_top_10_limit():
    # 15 soldiers all at same rank — should return only 10
    soldiers = [_soldier(f"X{i} Doe", rank="STR_SQUADDIE") for i in range(15)]
    base = _base(soldiers=soldiers)
    result = compute_metrics(_state(bases=[base]))
    assert len(result["metrics"]["soldiers"]) == 10


def test_soldiers_across_multiple_bases():
    b1 = _base("Alpha", soldiers=[_soldier("A One", rank="STR_COMMANDER")])
    b2 = _base("Bravo", soldiers=[_soldier("B Two", rank="STR_ROOKIE")])
    result = compute_metrics(_state(bases=[b1, b2]))
    names = [s["display"] for s in result["metrics"]["soldiers"]]
    assert names == ["A. One", "B. Two"]


def test_soldiers_empty():
    result = compute_metrics(_state(bases=[_base()]))
    assert result["metrics"]["soldiers"] == []


# ---------------------------------------------------------------------------
# Metric 3: Aircraft
# ---------------------------------------------------------------------------

def test_aircraft_single_of_type_no_index():
    base = _base(crafts=[_craft("STR_SKYRANGER")])
    result = compute_metrics(_state(bases=[base]))
    ac = result["metrics"]["aircraft"]
    assert len(ac) == 1
    assert ac[0]["index"] is None
    assert ac[0]["display_name"] == "Skyranger"


def test_aircraft_two_of_same_type_indexed():
    base = _base(crafts=[_craft("STR_INTERCEPTOR"), _craft("STR_INTERCEPTOR")])
    result = compute_metrics(_state(bases=[base]))
    ac = result["metrics"]["aircraft"]
    assert len(ac) == 2
    assert ac[0]["display_name"] == "Interceptor-1"
    assert ac[1]["display_name"] == "Interceptor-2"
    assert ac[0]["index"] == 1
    assert ac[1]["index"] == 2


def test_aircraft_mixed_types():
    base = _base(crafts=[
        _craft("STR_SKYRANGER"),
        _craft("STR_INTERCEPTOR"),
        _craft("STR_INTERCEPTOR"),
    ])
    result = compute_metrics(_state(bases=[base]))
    ac = result["metrics"]["aircraft"]
    types = {e["type"] for e in ac}
    assert types == {"STR_SKYRANGER", "STR_INTERCEPTOR"}
    skyranger = next(e for e in ac if e["type"] == "STR_SKYRANGER")
    assert skyranger["index"] is None


def test_aircraft_across_bases():
    b1 = _base("Alpha", crafts=[_craft("STR_SKYRANGER")])
    b2 = _base("Bravo", crafts=[_craft("STR_SKYRANGER")])
    result = compute_metrics(_state(bases=[b1, b2]))
    ac = result["metrics"]["aircraft"]
    assert len(ac) == 2
    assert all(e["index"] is not None for e in ac)


def test_aircraft_type_display_conversion():
    assert _type_to_display("STR_SKYRANGER") == "Skyranger"
    assert _type_to_display("STR_FIRESTORM") == "Firestorm"
    assert _type_to_display("STR_AVENGER") == "Avenger"


def test_aircraft_status_preserved():
    base = _base(crafts=[_craft("STR_SKYRANGER", status="STR_OUT")])
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["aircraft"][0]["status"] == "STR_OUT"


# ---------------------------------------------------------------------------
# Metric 4: Stores sell value
# ---------------------------------------------------------------------------

def test_stores_sell_value_single_item():
    base = _base(stores={"STR_RIFLE": 3})
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["stores_sell_value"] == 3 * SELL_PRICES["STR_RIFLE"]


def test_stores_sell_value_unknown_item_ignored():
    base = _base(stores={"STR_UNKNOWN_THING": 99})
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["stores_sell_value"] == 0


def test_stores_sell_value_multiple_items():
    stores = {"STR_ELERIUM_115": 10, "STR_ALIEN_ALLOYS": 20}
    base = _base(stores=stores)
    expected = 10 * 5000 + 20 * 6500
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["stores_sell_value"] == expected


def test_stores_sell_value_across_bases():
    b1 = _base("Alpha", stores={"STR_PISTOL": 2})
    b2 = _base("Bravo", stores={"STR_RIFLE": 1})
    expected = 2 * SELL_PRICES["STR_PISTOL"] + 1 * SELL_PRICES["STR_RIFLE"]
    result = compute_metrics(_state(bases=[b1, b2]))
    assert result["metrics"]["stores_sell_value"] == expected


def test_stores_sell_value_empty():
    result = compute_metrics(_state(bases=[_base()]))
    assert result["metrics"]["stores_sell_value"] == 0


# ---------------------------------------------------------------------------
# Metric 5: Research (scientists + lab capacity)
# ---------------------------------------------------------------------------

def test_research_free_scientists_summed():
    b1 = _base("Alpha", scientists=5)
    b2 = _base("Bravo", scientists=3)
    result = compute_metrics(_state(bases=[b1, b2]))
    assert result["metrics"]["research"]["scientists"] == 8


def test_research_lab_capacity_built_only():
    built_lab   = _facility("STR_LAB",   labs=50, build_days=0)
    unbuilt_lab = _facility("STR_LAB",   labs=50, build_days=3)
    base = _base(facilities=[built_lab, unbuilt_lab])
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["research"]["lab_capacity"] == 50


def test_research_lab_capacity_zero_when_no_labs():
    fac = _facility("STR_GENERAL_STORES", labs=0, workshops=0, build_days=0)
    base = _base(facilities=[fac])
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["research"]["lab_capacity"] == 0


def test_research_lab_capacity_multiple_bases():
    b1 = _base(facilities=[_facility("STR_LAB", labs=50)])
    b2 = _base(facilities=[_facility("STR_LAB", labs=50)])
    result = compute_metrics(_state(bases=[b1, b2]))
    assert result["metrics"]["research"]["lab_capacity"] == 100


# ---------------------------------------------------------------------------
# Metric 6: Engineering (engineers + workshop capacity)
# ---------------------------------------------------------------------------

def test_engineering_free_engineers_summed():
    b1 = _base("Alpha", engineers=8)
    b2 = _base("Bravo", engineers=2)
    result = compute_metrics(_state(bases=[b1, b2]))
    assert result["metrics"]["engineering"]["engineers"] == 10


def test_engineering_workshop_capacity_built_only():
    built   = _facility("STR_WORKSHOP", workshops=50, build_days=0)
    unbuilt = _facility("STR_WORKSHOP", workshops=50, build_days=5)
    base = _base(facilities=[built, unbuilt])
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["engineering"]["workshop_capacity"] == 50


def test_engineering_zero_when_no_workshops():
    base = _base(facilities=[_facility("STR_LAB", workshops=0)])
    result = compute_metrics(_state(bases=[base]))
    assert result["metrics"]["engineering"]["workshop_capacity"] == 0


# ---------------------------------------------------------------------------
# Metric 7: Active research projects
# ---------------------------------------------------------------------------

def test_research_projects_listed():
    rp1 = {"name": "STR_LASER_PISTOL", "scientists": 5}
    rp2 = {"name": "STR_ALIEN_ALLOYS", "scientists": 3}
    base = _base(research=[rp1, rp2])
    result = compute_metrics(_state(bases=[base]))
    projects = result["metrics"]["research_projects"]
    assert len(projects) == 2
    assert {"name": "STR_LASER_PISTOL", "scientists": 5} in projects
    assert {"name": "STR_ALIEN_ALLOYS", "scientists": 3} in projects


def test_research_projects_across_bases():
    b1 = _base("Alpha", research=[{"name": "STR_LASER_PISTOL", "scientists": 5}])
    b2 = _base("Bravo", research=[{"name": "STR_PLASMA_RIFLE", "scientists": 10}])
    result = compute_metrics(_state(bases=[b1, b2]))
    assert len(result["metrics"]["research_projects"]) == 2


def test_research_projects_empty():
    result = compute_metrics(_state(bases=[_base()]))
    assert result["metrics"]["research_projects"] == []


# ---------------------------------------------------------------------------
# Metric 8: Active manufacture projects
# ---------------------------------------------------------------------------

def test_manufacture_projects_listed():
    mp = {"name": "STR_LASER_PISTOL", "engineers": 10}
    base = _base(manufacture=[mp])
    result = compute_metrics(_state(bases=[base]))
    projects = result["metrics"]["manufacture_projects"]
    assert len(projects) == 1
    assert projects[0] == {"name": "STR_LASER_PISTOL", "engineers": 10}


def test_manufacture_projects_across_bases():
    b1 = _base("Alpha", manufacture=[{"name": "STR_RIFLE_CLIP", "engineers": 4}])
    b2 = _base("Bravo", manufacture=[{"name": "STR_LASER_RIFLE", "engineers": 8}])
    result = compute_metrics(_state(bases=[b1, b2]))
    assert len(result["metrics"]["manufacture_projects"]) == 2


def test_manufacture_projects_empty():
    result = compute_metrics(_state(bases=[_base()]))
    assert result["metrics"]["manufacture_projects"] == []


# ---------------------------------------------------------------------------
# Output envelope
# ---------------------------------------------------------------------------

def test_output_type_is_aggregated():
    result = compute_metrics(_state())
    assert result["type"] == "aggregated"


def test_output_event_passthrough():
    result = compute_metrics(_state(event="purchase"))
    assert result["event"] == "purchase"


def test_output_has_all_8_metric_keys():
    result = compute_metrics(_state())
    keys = set(result["metrics"].keys())
    expected = {
        "cash", "soldiers", "aircraft", "stores_sell_value",
        "research", "engineering", "research_projects", "manufacture_projects",
    }
    assert keys == expected
