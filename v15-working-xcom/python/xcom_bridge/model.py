"""Typed dataclasses for the OpenXcom bridge state message.

Every message from the engine is a JSON object with top-level keys:
  type   – always "state"
  event  – what triggered the broadcast: "tick" | "purchase" | "build" |
            "sack" | "transfer"
  game   – global game state (funds, date, score, countries)
  bases  – list of Base objects
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Game-level
# ---------------------------------------------------------------------------

@dataclass
class GameDate:
    year: int
    month: int
    day: int
    hour: int
    minute: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}"


@dataclass
class Country:
    id: str           # e.g. "STR_USA"
    funding: int      # current monthly funding (USD × 1000)
    satisfaction: int # 0=content, 1=happy, -1=unhappy (engine values vary)


@dataclass
class Game:
    funds: int
    date: GameDate
    score: int
    countries: list[Country]


# ---------------------------------------------------------------------------
# Base-level
# ---------------------------------------------------------------------------

@dataclass
class FacilityCapacity:
    personnel: int
    storage: int
    crafts: int
    labs: int
    workshops: int
    psi_labs: int


@dataclass
class Facility:
    type: str               # e.g. "STR_ACCESS_LIFT"
    x: int
    y: int
    build_days: int         # 0 = complete, >0 = under construction
    capacity: FacilityCapacity

    @property
    def is_built(self) -> bool:
        return self.build_days == 0


@dataclass
class UnitStats:
    tu: int
    stamina: int
    health: int
    bravery: int
    reactions: int
    firing: int
    throwing: int
    strength: int
    psi_strength: int
    psi_skill: int
    melee: int


@dataclass
class Soldier:
    name: str
    rank: str               # e.g. "STR_ROOKIE"
    stats: UnitStats
    armor: str              # e.g. "STR_NONE_UC"
    wound_recovery: int     # 0 = healthy
    craft: Optional[str]    # craft type if assigned, else None


@dataclass
class CraftWeapon:
    type: str
    ammo: int
    ammo_max: int


@dataclass
class Craft:
    type: str               # e.g. "STR_SKYRANGER"
    id: int = 0             # craft serial number (e.g. 1 for Skyranger-1)
    name: str = ""          # localized name e.g. "Skyranger-1"
    status: str = ""        # e.g. "STR_READY"
    fuel: int = 0
    fuel_max: int = 0
    damage: int = 0
    damage_max: int = 0
    soldiers: int = 0       # soldiers currently aboard
    weapons: list = field(default_factory=list)

    @property
    def fuel_pct(self) -> int:
        return int(100 * self.fuel / self.fuel_max) if self.fuel_max else 0

    @property
    def damage_pct(self) -> int:
        return int(100 * self.damage / self.damage_max) if self.damage_max else 0


@dataclass
class Transfer:
    kind: str               # "soldier"|"craft"|"item"|"scientists"|"engineers"
    hours: int
    # kind-specific fields — only one of these is populated per transfer
    soldier: Optional[Soldier] = None
    craft: Optional[Craft]    = None
    item: Optional[str]       = None   # item type string
    qty: int                  = 0


@dataclass
class Base:
    name: str
    lon: float
    lat: float
    scientists: int
    engineers: int
    facilities: list[Facility]
    stores: dict[str, int]      # item_type_id -> quantity
    soldiers: list[Soldier]
    crafts: list[Craft]
    transfers: list[Transfer]


# ---------------------------------------------------------------------------
# Battlescape types
# ---------------------------------------------------------------------------

# UnitStatus int values from the engine
UNIT_STATUS = {
    0: "standing", 1: "walking", 2: "flying", 3: "turning",
    4: "aiming",   5: "collapsing", 6: "dead", 7: "unconscious",
    8: "panicking", 9: "berserk", 10: "ignore",
}
UNIT_FACTION = {0: "player", 1: "hostile", 2: "neutral"}
# BattleActionType int values (TU-reserve setting)
BA_TYPE = {0: "none", 1: "turn", 2: "walk", 3: "prime", 4: "throw",
           5: "snapshot", 6: "autoshot", 7: "aimedshot", 8: "hit",
           9: "use", 10: "launch", 11: "mindcontrol", 12: "panic"}


@dataclass
class BattleWeapon:
    type: str
    ammo: int
    ammo_max: int


@dataclass
class BattleUnit:
    id: int
    type: str
    faction: int        # 0=player, 1=hostile, 2=neutral
    status: int         # 0=standing … 6=dead, 7=unconscious
    x: int
    y: int
    z: int
    dir: int            # 0–7 facing direction
    tu: int
    tu_max: int
    health: int
    health_max: int
    energy: int
    energy_max: int
    stun: int
    morale: int
    reactions: int
    kneeling: bool
    armor: list[int]    # [front, left, right, rear, under]
    right_hand: Optional[BattleWeapon]
    left_hand: Optional[BattleWeapon]

    @property
    def is_out(self) -> bool:
        return self.status in (6, 7)   # dead or unconscious

    @property
    def faction_name(self) -> str:
        return UNIT_FACTION.get(self.faction, str(self.faction))

    @property
    def status_name(self) -> str:
        return UNIT_STATUS.get(self.status, str(self.status))


@dataclass
class BattleMap:
    """Decoded tile grid.  All arrays are flat, indexed z*W*L + y*W + x."""
    width: int
    length: int
    height: int
    # Raw byte buffers decoded from base64 (uint8 per tile, or uint16le for unit_id)
    smoke: bytes
    fire: bytes
    unit_id: bytes      # uint16le per tile — use get_unit_id(x,y,z) to read
    visible: bytes      # bit0=player disc, bit1=alien disc, bit2=above, bit3=currently lit
    terrain: Optional[bytes]  # uint8[4] per tile (floor/west/north/object mapDataID, 255=empty)
                              # only present in battle_start messages

    def _idx(self, x: int, y: int, z: int) -> int:
        return z * self.width * self.length + y * self.width + x

    def get_smoke(self, x: int, y: int, z: int) -> int:
        return self.smoke[self._idx(x, y, z)]

    def get_fire(self, x: int, y: int, z: int) -> int:
        return self.fire[self._idx(x, y, z)]

    def get_visible(self, x: int, y: int, z: int) -> int:
        return self.visible[self._idx(x, y, z)]

    def get_unit_id(self, x: int, y: int, z: int) -> int:
        i = self._idx(x, y, z) * 2
        return int.from_bytes(self.unit_id[i:i+2], "little")

    def get_terrain(self, x: int, y: int, z: int) -> Optional[tuple]:
        """Returns (floor_id, west_id, north_id, object_id) or None if no terrain data."""
        if self.terrain is None:
            return None
        i = self._idx(x, y, z) * 4
        return tuple(self.terrain[i:i+4])


@dataclass
class BattleMessage:
    event: str          # "battle_start"|"turn_start"|"move"|"shoot"|"throw"|"melee"
    turn: int
    side: int           # 0=player, 1=hostile, 2=neutral
    tu_reserved: int    # BA_* enum
    active_unit: Optional[int]
    map: BattleMap
    units: list[BattleUnit]

    @property
    def side_name(self) -> str:
        return UNIT_FACTION.get(self.side, str(self.side))

    @classmethod
    def from_dict(cls, d: dict) -> "BattleMessage":
        return cls(
            event=d["event"],
            turn=d["turn"],
            side=d["side"],
            tu_reserved=d.get("tu_reserved", 0),
            active_unit=d.get("active_unit"),
            map=_parse_battle_map(d["map"]),
            units=[_parse_battle_unit(u) for u in d.get("units", [])],
        )


# ---------------------------------------------------------------------------
# Top-level message
# ---------------------------------------------------------------------------

@dataclass
class StateMessage:
    event: str              # "tick" | "purchase" | "build" | "sack" | "transfer"
    game: Game
    bases: list[Base]

    # ── factories ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "StateMessage":
        return cls(
            event=d.get("event", "tick"),
            game=_parse_game(d["game"]),
            bases=[_parse_base(b) for b in d["bases"]],
        )


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_game(g: dict) -> Game:
    return Game(
        funds=g["funds"],
        date=GameDate(g["year"], g["month"], g["day"], g["hour"], g["minute"]),
        score=g["score"],
        countries=[Country(**c) for c in g["countries"]],
    )


def _parse_facility(f: dict) -> Facility:
    cap = f.get("capacity", {})
    return Facility(
        type=f["type"], x=f["x"], y=f["y"], build_days=f["build_days"],
        capacity=FacilityCapacity(
            personnel=cap.get("personnel", 0),
            storage=cap.get("storage", 0),
            crafts=cap.get("crafts", 0),
            labs=cap.get("labs", 0),
            workshops=cap.get("workshops", 0),
            psi_labs=cap.get("psi_labs", 0),
        ),
    )


def _parse_stats(s: dict) -> UnitStats:
    return UnitStats(
        tu=s["tu"], stamina=s["stamina"], health=s["health"],
        bravery=s["bravery"], reactions=s["reactions"],
        firing=s["firing"], throwing=s["throwing"], strength=s["strength"],
        psi_strength=s["psi_strength"], psi_skill=s["psi_skill"],
        melee=s["melee"],
    )


def _parse_soldier(s: dict) -> Soldier:
    return Soldier(
        name=s["name"], rank=s["rank"],
        stats=_parse_stats(s),
        armor=s["armor"],
        wound_recovery=s["wound_recovery"],
        craft=s.get("craft"),
    )


def _parse_craft(c: dict) -> Craft:
    return Craft(
        type=c["type"],
        id=c.get("id", 0),
        name=c.get("name", c["type"]),
        status=c["status"],
        fuel=c["fuel"], fuel_max=c["fuel_max"],
        damage=c["damage"], damage_max=c["damage_max"],
        soldiers=c.get("soldiers", 0),
        weapons=[CraftWeapon(**w) for w in c.get("weapons", [])],
    )


def _parse_transfer(t: dict) -> Transfer:
    kind = t["kind"]
    return Transfer(
        kind=kind,
        hours=t["hours"],
        soldier=_parse_soldier(t["soldier"]) if kind == "soldier" and "soldier" in t else None,
        craft=_parse_craft(t["craft"])       if kind == "craft"   and "craft"   in t else None,
        item=t.get("item"),
        qty=t.get("qty", 0),
    )


def _parse_base(b: dict) -> Base:
    return Base(
        name=b["name"], lon=b["lon"], lat=b["lat"],
        scientists=b["scientists"], engineers=b["engineers"],
        facilities=[_parse_facility(f) for f in b.get("facilities", [])],
        stores=b.get("stores", {}),
        soldiers=[_parse_soldier(s) for s in b.get("soldiers", [])],
        crafts=[_parse_craft(c)     for c in b.get("crafts", [])],
        transfers=[_parse_transfer(t) for t in b.get("transfers", [])],
    )


# ---------------------------------------------------------------------------
# Battle parsers
# ---------------------------------------------------------------------------

def _parse_battle_weapon(w: Optional[dict]) -> Optional[BattleWeapon]:
    if w is None:
        return None
    return BattleWeapon(type=w["type"], ammo=w["ammo"], ammo_max=w["ammo_max"])


def _parse_battle_unit(u: dict) -> BattleUnit:
    return BattleUnit(
        id=u["id"], type=u["type"],
        faction=u["faction"], status=u["status"],
        x=u["x"], y=u["y"], z=u["z"],
        dir=u["dir"],
        tu=u["tu"],     tu_max=u["tu_max"],
        health=u["health"], health_max=u["health_max"],
        energy=u["energy"], energy_max=u["energy_max"],
        stun=u["stun"], morale=u["morale"], reactions=u["reactions"],
        kneeling=u["kneeling"],
        armor=u["armor"],
        right_hand=_parse_battle_weapon(u.get("right_hand")),
        left_hand=_parse_battle_weapon(u.get("left_hand")),
    )


def _parse_battle_map(m: dict) -> BattleMap:
    import base64 as _b64
    def _dec(key: str) -> bytes:
        raw = m.get(key, "")
        return _b64.b64decode(raw) if raw else b""

    return BattleMap(
        width=m["width"], length=m["length"], height=m["height"],
        smoke=_dec("smoke"),
        fire=_dec("fire"),
        unit_id=_dec("unit_id"),
        visible=_dec("visible"),
        terrain=_dec("terrain") if "terrain" in m else None,
    )
