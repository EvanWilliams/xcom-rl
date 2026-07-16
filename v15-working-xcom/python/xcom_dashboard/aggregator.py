"""State aggregation layer: derives 8 dashboard metrics from a raw bridge state dict.

Called once per incoming geoscape state message; returns an ``aggregated`` message
that is broadcast alongside the raw message on the same /ws endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Static data tables
# ---------------------------------------------------------------------------

RANK_ORDER: Dict[str, int] = {
    "STR_ROOKIE":     1,
    "STR_SQUADDIE":   2,
    "STR_SERGEANT":   3,
    "STR_CAPTAIN":    4,
    "STR_COLONEL":    5,
    "STR_COMMANDER":  6,
}

# Sell prices from engine/openxcom/bin/standard/xcom1/items.rul
SELL_PRICES: Dict[str, int] = {
    "STR_AC_AP_AMMO": 400,
    "STR_AC_HE_AMMO": 560,
    "STR_AC_I_AMMO": 520,
    "STR_ALIEN_ALLOYS": 6500,
    "STR_ALIEN_ENTERTAINMENT": 20000,
    "STR_ALIEN_FOOD": 5000,
    "STR_ALIEN_GRENADE": 14850,
    "STR_ALIEN_HABITAT": 100,
    "STR_ALIEN_REPRODUCTION": 40000,
    "STR_ALIEN_SURGERY": 38000,
    "STR_AUTO_CANNON": 10125,
    "STR_AVALANCHE_LAUNCHER": 12750,
    "STR_AVALANCHE_MISSILES": 7200,
    "STR_BLASTER_BOMB": 17028,
    "STR_BLASTER_LAUNCHER": 144000,
    "STR_CANNON": 22500,
    "STR_CANNON_ROUNDS_X50": 1012,
    "STR_CELATID_CORPSE": 20000,
    "STR_CELATID_TERRORIST": 40000,
    "STR_CHRYSSALID_CORPSE": 20000,
    "STR_CHRYSSALID_TERRORIST": 60000,
    "STR_CYBERDISC_CORPSE": 20000,
    "STR_CYBERDISC_TERRORIST": 80000,
    "STR_ELECTRO_FLARE": 40,
    "STR_ELERIUM_115": 5000,
    "STR_ETHEREAL_COMMANDER": 80000,
    "STR_ETHEREAL_CORPSE": 20000,
    "STR_ETHEREAL_LEADER": 60000,
    "STR_ETHEREAL_SOLDIER": 60000,
    "STR_EXAMINATION_ROOM": 9000,
    "STR_FLOATER_COMMANDER": 60000,
    "STR_FLOATER_CORPSE": 20000,
    "STR_FLOATER_ENGINEER": 60000,
    "STR_FLOATER_LEADER": 60000,
    "STR_FLOATER_MEDIC": 60000,
    "STR_FLOATER_NAVIGATOR": 60000,
    "STR_FLOATER_SOLDIER": 40000,
    "STR_FLYING_SUIT": 115000,
    "STR_FUSION_BALL": 53300,
    "STR_FUSION_BALL_LAUNCHER": 281000,
    "STR_GRENADE": 240,
    "STR_HC_AP_AMMO": 225,
    "STR_HC_HE_AMMO": 275,
    "STR_HC_I_AMMO": 300,
    "STR_HEAVY_CANNON": 4800,
    "STR_HEAVY_LASER": 61000,
    "STR_HEAVY_PLASMA": 171600,
    "STR_HEAVY_PLASMA_CLIP": 9590,
    "STR_HIGH_EXPLOSIVE": 1200,
    "STR_HOVERTANK_LAUNCHER": 1043000,
    "STR_HOVERTANK_PLASMA": 980000,
    "STR_HWP_CANNON_SHELLS": 100,
    "STR_HWP_FUSION_BOMB": 31500,
    "STR_HWP_ROCKETS": 2250,
    "STR_INCENDIARY_ROCKET": 960,
    "STR_LARGE_ROCKET": 720,
    "STR_LASER_CANNON": 211000,
    "STR_LASER_PISTOL": 20000,
    "STR_LASER_RIFLE": 36900,
    "STR_MEDI_KIT": 46500,
    "STR_MIND_PROBE": 304000,
    "STR_MOTION_SCANNER": 45600,
    "STR_MUTON_CORPSE": 20000,
    "STR_MUTON_ENGINEER": 60000,
    "STR_MUTON_NAVIGATOR": 60000,
    "STR_MUTON_SOLDIER": 40000,
    "STR_PERSONAL_ARMOR": 54000,
    "STR_PISTOL": 600,
    "STR_PISTOL_CLIP": 52,
    "STR_PLASMA_BEAM": 267300,
    "STR_PLASMA_PISTOL": 84000,
    "STR_PLASMA_PISTOL_CLIP": 4440,
    "STR_PLASMA_RIFLE": 126500,
    "STR_PLASMA_RIFLE_CLIP": 6290,
    "STR_POWER_SUIT": 85000,
    "STR_PROXIMITY_GRENADE": 400,
    "STR_PSI_AMP": 194700,
    "STR_REAPER_CORPSE": 20000,
    "STR_REAPER_TERRORIST": 40000,
    "STR_RIFLE": 2250,
    "STR_RIFLE_CLIP": 150,
    "STR_ROCKET_LAUNCHER": 3000,
    "STR_SECTOID_COMMANDER": 60000,
    "STR_SECTOID_CORPSE": 20000,
    "STR_SECTOID_ENGINEER": 60000,
    "STR_SECTOID_LEADER": 60000,
    "STR_SECTOID_MEDIC": 60000,
    "STR_SECTOID_NAVIGATOR": 60000,
    "STR_SECTOID_SOLDIER": 40000,
    "STR_SECTOPOD_CORPSE": 20000,
    "STR_SECTOPOD_TERRORIST": 80000,
    "STR_SILACOID_CORPSE": 20000,
    "STR_SILACOID_TERRORIST": 40000,
    "STR_SMALL_LAUNCHER": 120000,
    "STR_SMALL_ROCKET": 480,
    "STR_SMOKE_GRENADE": 120,
    "STR_SNAKEMAN_COMMANDER": 60000,
    "STR_SNAKEMAN_CORPSE": 20000,
    "STR_SNAKEMAN_ENGINEER": 60000,
    "STR_SNAKEMAN_LEADER": 60000,
    "STR_SNAKEMAN_NAVIGATOR": 60000,
    "STR_SNAKEMAN_SOLDIER": 40000,
    "STR_STINGRAY_LAUNCHER": 12000,
    "STR_STINGRAY_MISSILES": 2400,
    "STR_STUN_BOMB": 15200,
    "STR_STUN_ROD": 945,
    "STR_TANK_CANNON": 340000,
    "STR_TANK_LASER_CANNON": 594000,
    "STR_TANK_ROCKET_LAUNCHER": 360000,
    "STR_UFO_CONSTRUCTION": 20000,
    "STR_UFO_NAVIGATION": 80000,
    "STR_UFO_POWER_SOURCE": 250000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soldier_display(soldier: Dict[str, Any]) -> str:
    """Return 'F. Lastname' formatted name from a full soldier name string."""
    name = soldier.get("name", "")
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name  # single-word name: return as-is


def _armor_display(soldier: Dict[str, Any]) -> str:
    """Human-readable armor label from the STR_*_UC type string."""
    raw = soldier.get("armor", "STR_NONE_UC")
    if raw in ("STR_NONE_UC", "STR_ARMOR_NONE_UC", ""):
        return "None"
    name = raw.removeprefix("STR_").removesuffix("_UC").replace("_", " ").title()
    return name


def _type_to_display(type_str: str) -> str:
    """'STR_SKYRANGER' → 'Skyranger'."""
    return type_str.removeprefix("STR_").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Metric computers
# ---------------------------------------------------------------------------

def _metric_cash(state: Dict[str, Any]) -> int:
    return state.get("game", {}).get("funds", 0)


def _metric_soldiers(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_soldiers = [
        s
        for base in state.get("bases", [])
        for s in base.get("soldiers", [])
    ]
    all_soldiers.sort(
        key=lambda s: RANK_ORDER.get(s.get("rank", ""), 0),
        reverse=True,
    )
    return [
        {"display": _soldier_display(s), "armor": _armor_display(s)}
        for s in all_soldiers[:10]
    ]


def _metric_aircraft(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect all crafts across bases; auto-index per type when >1 exist."""
    from collections import defaultdict

    # First pass: count per type
    type_counts: Dict[str, int] = defaultdict(int)
    flat: List[Dict[str, Any]] = []
    for base in state.get("bases", []):
        for craft in base.get("crafts", []):
            t = craft.get("type", "")
            type_counts[t] += 1
            flat.append({
                "type": t,
                "status": craft.get("status", ""),
                "base": base.get("name", ""),
            })

    # Second pass: assign sequential index per type when count > 1
    type_seen: Dict[str, int] = defaultdict(int)
    result: List[Dict[str, Any]] = []
    for entry in flat:
        t = entry["type"]
        total = type_counts[t]
        if total > 1:
            type_seen[t] += 1
            idx: Optional[int] = type_seen[t]
            display = f"{_type_to_display(t)}-{idx}"
        else:
            idx = None
            display = _type_to_display(t)
        result.append({
            "type": t,
            "display_name": display,
            "index": idx,
            "status": entry["status"],
            "base": entry["base"],
        })
    return result


def _metric_stores_sell_value(state: Dict[str, Any]) -> int:
    total = 0
    for base in state.get("bases", []):
        for item_id, qty in base.get("stores", {}).items():
            total += qty * SELL_PRICES.get(item_id, 0)
    return total


def _metric_research(state: Dict[str, Any]) -> Dict[str, int]:
    scientists = sum(b.get("scientists", 0) for b in state.get("bases", []))
    lab_capacity = sum(
        fac["capacity"]["labs"]
        for b in state.get("bases", [])
        for fac in b.get("facilities", [])
        if fac.get("build_days", 1) == 0 and fac.get("capacity", {}).get("labs", 0) > 0
    )
    return {"scientists": scientists, "lab_capacity": lab_capacity}


def _metric_engineering(state: Dict[str, Any]) -> Dict[str, int]:
    engineers = sum(b.get("engineers", 0) for b in state.get("bases", []))
    workshop_capacity = sum(
        fac["capacity"]["workshops"]
        for b in state.get("bases", [])
        for fac in b.get("facilities", [])
        if fac.get("build_days", 1) == 0 and fac.get("capacity", {}).get("workshops", 0) > 0
    )
    return {"engineers": engineers, "workshop_capacity": workshop_capacity}


def _metric_research_projects(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": rp["name"], "scientists": rp["scientists"]}
        for b in state.get("bases", [])
        for rp in b.get("research", [])
    ]


def _metric_manufacture_projects(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": p["name"], "engineers": p["engineers"]}
        for b in state.get("bases", [])
        for p in b.get("manufacture", [])
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Derive all 8 metrics from a raw geoscape state dict.

    Returns a dict ready to serialize as the ``aggregated`` WebSocket message.
    Only called for ``type == "state"`` messages; callers should guard.
    """
    return {
        "type": "aggregated",
        "event": state.get("event", ""),
        "metrics": {
            "cash": _metric_cash(state),
            "soldiers": _metric_soldiers(state),
            "aircraft": _metric_aircraft(state),
            "stores_sell_value": _metric_stores_sell_value(state),
            "research": _metric_research(state),
            "engineering": _metric_engineering(state),
            "research_projects": _metric_research_projects(state),
            "manufacture_projects": _metric_manufacture_projects(state),
        },
    }
