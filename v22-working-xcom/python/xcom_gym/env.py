"""
Gymnasium Env — click-only interface to OpenXcom.

All actions are simulated mouse clicks (or Escape) injected via SDL, exactly
as a human player would interact with the game.  No internal API shortcuts.

Action space: Discrete(212)  — see ACTIONS table below.
Observation space: Dict — game state + screen_hint so the agent can tell
                          which dialog/screen is currently showing.

Reward:
  +0.02         commit click (intercept, confirm, research_info_ok, mfg_start, etc.)
  +0.01         navigation click (sidebar, list rows, cancel/back, other named buttons)
  -0.005        wait (each agent-step covers up to 60 game-minutes in headless mode)
  +0.003        random_click / random_right_click (exploration incentive)
  +10           UFO first detected on radar
  +50           craft dispatched (STR_OUT transition)
  +1000         game won (Cydonia)
  delta*1.0     routine strategic score ticks (|delta| < 10)
  delta*2.0     post-battle total_score delta (|delta| >= 10) — covers:
                  alien kills (Sectoid Soldier +10 … Ethereal Commander +40)
                  alien stuns (2× kill value)
                  civilian killed by X-COM  −50 | civilian killed by aliens −30
                  soldier killed/MIA −20 | HWP destroyed −20
                  civilian saved +30 | alien base destroyed +500
  delta*3.0     research score delta (score field, any positive delta) — covers:
                  Plasma Pistol Clip +5 … Cydonia or Bust +60 (§4.8 table)

The random_click action samples a uniform random (x, y) from the full
320×200 game screen at runtime so the agent can discover UI elements
not covered by named buttons.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xcom_bridge import _winsock as _ws

# ── paths ──────────────────────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_EXE = os.path.join(_ROOT, "engine", "openxcom", "build", "bin", "openxcom.exe")
ENGINE_CWD = os.path.join(_ROOT, "engine", "openxcom", "build", "bin")
_TEMP      = os.environ.get("TEMP", r"C:\Windows\Temp")
_LOG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ── action table ───────────────────────────────────────────────────────────────
# Each entry: (name, x_or_None, y_or_None)
# x=None  →  special action (wait / dismiss)
# x≥0     →  click(x, y) injected as SDL mouse event
#
# Geoscape button positions (320×200 screen, sidebar at x=257..320):
#   screenWidth=320, screenHeight=200, screenHeight/2=100
#   _btnIntercept  TextButton(63,11, 257,  0) → centre (288,  5)
#   _btnBases      TextButton(63,11, 257, 12) → centre (288, 17)
#   _btnOptions    TextButton(63,11, 257, 48) → centre (288, 53)
#   _btn5Secs      TextButton(31,13, 257,112) → centre (272,118)
#   _btn1Min       TextButton(31,13, 289,112) → centre (304,118)
#   _btn5Mins      TextButton(31,13, 257,126) → centre (272,132)
#   _btn30Mins     TextButton(31,13, 289,126) → centre (304,132)
#   _btn1Hour      TextButton(31,13, 257,140) → centre (272,146)
#   _btn1Day       TextButton(31,13, 289,140) → centre (304,146)
#
# Dialog button positions (from source + previous analysis):
#   UfoDetectedState    Intercept(128,124) Centre(128,140) Cancel(128,156)
#   MissionDetectedState Intercept(128,138) Centre(128,158) Cancel(128,178)
#   InterceptState      row-1 craft (152,82)  Cancel (244,152)
#   ConfirmDestState    OK (93,110)  Cancel (163,110)
#   Dismiss (Escape)    → closes LowFuel/CraftError/MonthlyReport/ItemsArriving etc.

ACTIONS: list[tuple[str, Optional[int], Optional[int], int]] = [
    # (name, x, y, button)  button: 1=left  3=right
    # ── Geoscape idle / wait ────────────────────────────────────────────
    ("wait",                 None, None, 1),   # 0 — no-op; in headless sends step

    # ── Geoscape sidebar ────────────────────────────────────────────────
    ("geo_intercept",         288,    5, 1),   # 1 — open Intercept screen manually
    ("geo_bases",             288,   17, 1),   # 2 — go to Bases screen
    ("geo_options",           288,   53, 1),   # 3 — options / pause

    # ── Time speed buttons ──────────────────────────────────────────────
    ("speed_5sec",            272,  118, 1),   # 4
    ("speed_1min",            304,  118, 1),   # 5
    ("speed_5min",            272,  132, 1),   # 6
    ("speed_30min",           304,  132, 1),   # 7
    ("speed_1hour",           272,  146, 1),   # 8
    ("speed_1day",            304,  146, 1),   # 9

    # ── UFO Detected dialog ─────────────────────────────────────────────
    ("ufo_intercept",         128,  124, 1),   # 10
    ("ufo_centre",            128,  140, 1),   # 11
    ("ufo_cancel",            128,  156, 1),   # 12

    # ── Mission Detected dialog ─────────────────────────────────────────
    ("mission_intercept",     128,  138, 1),   # 13
    ("mission_cancel",        128,  178, 1),   # 14

    # ── Intercept screen ────────────────────────────────────────────────
    ("intercept_craft1",      152,   82, 1),   # 15 — first craft row
    ("intercept_cancel",       87,  154, 1),   # 16 — cancel; (87,154) is inside button regardless of _base context

    # ── Confirm Destination dialog ──────────────────────────────────────
    ("confirm_ok",             93,  110, 1),   # 17
    ("confirm_cancel",        163,  110, 1),   # 18

    # ── Dismiss (Escape) ────────────────────────────────────────────────
    ("dismiss",               None, None, 1),  # 19 — closes info dialogs, monthly report, etc.

    # ── Exploratory random clicks ─────────────────────────────────────────
    # x=-1 is a sentinel; _execute_action samples uniform random (x, y) at
    # runtime from the full 320×200 game screen.  Earn no click reward so the
    # policy learns to prefer named buttons and eventually stops using these.
    ("random_click",            -1,   -1, 1),  # 20 — left-click at random position
    ("random_right_click",      -1,   -1, 3),  # 21 — right-click at random position

    # ── Keyboard escapes ─────────────────────────────────────────────────
    # Used to navigate screens that have no exposed click targets, e.g. the
    # "Create New Game" / name-entry screens that appear before the geoscape.
    # No click reward — keyboard actions earn 0 like wait/dismiss.
    ("key_a",                 None, None, 1),  # 22 — 'A' key
    ("key_enter",             None, None, 1),  # 23 — Enter / Return key

    # ── Main-menu escape ──────────────────────────────────────────────────
    # Sends the bridge "new_game" command which bypasses the GUI menu flow
    # and jumps directly to a fresh Beginner geoscape.  Only takes effect
    # when screen_type == main_menu; otherwise it is a no-op (like wait).
    # Earns MENU_ESCAPE_REWARD (+5) when the screen successfully transitions
    # from main_menu to geoscape.
    ("menu_new_game",         None, None, 1),  # 24 — bridge new_game (main-menu only)

    # ── BasescapeState navigation ─────────────────────────────────────────────
    # Buttons at x=192, width=128 → centre_x=256; heights 12 → centre_y from top:
    #   _btnFacilities  y=110 → 116   _btnResearch  y=123 → 129   _btnGeoscape  y=188 → 194
    ("base_facilities",      256,  116, 1),  # 25 — Build Facilities
    ("base_research",        256,  129, 1),  # 26 — Research
    ("base_geoscape",        256,  194, 1),  # 27 — return to Geoscape

    # ── ResearchState controls ────────────────────────────────────────────────
    # Full-screen state (320×200).
    #   _btnNew  TextButton(148,16, 8,176) → centre (82, 184)
    #   _btnOk   TextButton(148,16,164,176)→ centre (238,184)
    #   _lstResearch TextList(288,112, 8,62) → rows at y=62, 9-px rows, x-centre=152
    ("research_new_project",  82,  184, 1),  # 28 — "New Project" → NewResearchListState
    ("research_exit",        238,  184, 1),  # 29 — "OK" → exit ResearchState
    ("research_proj_row0",   152,   66, 1),  # 30 — click active project row 0
    ("research_proj_row1",   152,   75, 1),  # 31
    ("research_proj_row2",   152,   84, 1),  # 32
    ("research_proj_row3",   152,   93, 1),  # 33
    ("research_proj_row4",   152,  102, 1),  # 34

    # ── NewResearchListState (research_new screen) ────────────────────────────
    # Popup window 230×140 at (45,30).
    #   _lstResearch TextList(198,88, 53,54) → row height 9 px, x-centre=152
    #   _btnOK  TextButton(214,16, 53,146)   → centre (160,154)
    ("research_list_row0",   152,   58, 1),  # 35 — first available project
    ("research_list_row1",   152,   67, 1),  # 36
    ("research_list_row2",   152,   76, 1),  # 37
    ("research_list_row3",   152,   85, 1),  # 38
    ("research_list_row4",   152,   94, 1),  # 39
    ("research_list_row5",   152,  103, 1),  # 40
    ("research_list_row6",   152,  112, 1),  # 41
    ("research_list_row7",   152,  121, 1),  # 42
    ("research_list_row8",   152,  130, 1),  # 43
    ("research_list_cancel", 160,  154, 1),  # 44 — OK/close without selecting

    # ── ResearchInfoState (research_info screen) ──────────────────────────────
    # Popup window 230×140 at (45,30).
    #   _btnMore ArrowButton(ARROW_BIG_UP,  13,14,195,100) → centre (201,107)
    #   _btnLess ArrowButton(ARROW_BIG_DOWN,13,14,195,120) → centre (201,127)
    #   _btnOk   TextButton(90,16,169,145)                 → centre (214,153)
    #   _btnCancel TextButton(90,16, 61,145)               → centre (106,153)
    ("research_info_more",     201,  107, 1),  # 45 — left-click ▲  = +1 scientist
    ("research_info_more_all", 201,  107, 3),  # 46 — right-click ▲ = add all
    ("research_info_ok",       214,  153, 1),  # 47 — Start/OK → confirm & return
    ("research_info_cancel",   106,  153, 1),  # 48 — Cancel → discard & return

    # ── Battlescape item pickup ────────────────────────────────────────────────
    # Sends a pick_up bridge command for the currently selected unit's tile.
    # Silently ignored when in geoscape (bridge geo-handler doesn't know "pick_up").
    ("battle_pickup",          None, None, 1),  # 49

    # ── BasescapeState — complete navigation buttons ───────────────────────────────
    # All buttons: 128×12 at x=192 → centre_x=256; centre_y = btn_y + 6
    #   _btnNewBase y=58→64  _btnBaseInfo y=71→77  _btnSoldiers y=84→90
    #   _btnCrafts y=97→103  _btnManufacture y=136→142  _btnTransfer y=149→155
    #   _btnPurchase y=162→168  _btnSell y=175→181
    ("base_new_base",      256,   64, 1),  # 50 — Build New Base
    ("base_info",          256,   77, 1),  # 51 — Base Information
    ("base_soldiers",      256,   90, 1),  # 52 — Soldiers list
    ("base_crafts",        256,  103, 1),  # 53 — Craft list
    ("base_manufacture",   256,  142, 1),  # 54 — Manufacture
    ("base_transfer",      256,  155, 1),  # 55 — Transfer
    ("base_purchase",      256,  168, 1),  # 56 — Purchase / Hire Personnel
    ("base_sell",          256,  181, 1),  # 57 — Sell / Sack Personnel

    # ── PurchaseState ─────────────────────────────────────────────────────────────
    # List at (8,54,287×120); arrowPos=227; a1(▲) at x=8+227=235 w=11→centre 240;
    # a2(▼) at x=235+12=247 w=11→centre 252; row height ≈8 px.
    # down actions appended at 200–211 (end of table).
    #   _btnOk (8,176,148×16)→(82,184)  _btnCancel (164,176,148×16)→(238,184)
    ("purchase_ok",        82,   184, 1),  # 58
    ("purchase_cancel",   238,   184, 1),  # 59
    ("purchase_row_up0",  240,    58, 1),  # 60 — +1 qty row 0
    ("purchase_row_up1",  240,    66, 1),  # 61
    ("purchase_row_up2",  240,    74, 1),  # 62
    ("purchase_row_up3",  240,    82, 1),  # 63
    ("purchase_row_up4",  240,    90, 1),  # 64
    ("purchase_row_up5",  240,    98, 1),  # 65

    # ── SellState ─────────────────────────────────────────────────────────────────
    # List at (8,54,287×120); arrowPos=182; a1(▲) at x=8+182=190 w=11→centre 195;
    # a2(▼) at x=190+12=202 w=11→centre 207.
    # down actions appended at 200–211 (end of table).
    #   _btnOk (8,176,...)→(82,184)  _btnCancel (164,176,...)→(238,184)
    ("sell_ok",            82,   184, 1),  # 66
    ("sell_cancel",       238,   184, 1),  # 67
    ("sell_row_up0",      195,    58, 1),  # 68 — +1 sell qty row 0
    ("sell_row_up1",      195,    66, 1),  # 69
    ("sell_row_up2",      195,    74, 1),  # 70
    ("sell_row_up3",      195,    82, 1),  # 71
    ("sell_row_up4",      195,    90, 1),  # 72
    ("sell_row_up5",      195,    98, 1),  # 73

    # ── SoldiersState ─────────────────────────────────────────────────────────────
    # List at (8,40,288×128); row height ≈8 px; centre_x=152.
    #   _btnOk TextButton(148,16,164,176)→(238,184)
    ("soldiers_ok",       238,   184, 1),  # 74
    ("soldiers_row0",     152,    44, 1),  # 75
    ("soldiers_row1",     152,    52, 1),  # 76
    ("soldiers_row2",     152,    60, 1),  # 77
    ("soldiers_row3",     152,    68, 1),  # 78
    ("soldiers_row4",     152,    76, 1),  # 79
    ("soldiers_row5",     152,    84, 1),  # 80

    # ── SoldierInfoState ──────────────────────────────────────────────────────────
    #   _btnPrev  (0,33,28×14)   → centre (14,40)
    #   _btnOk    (30,33,48×14)  → centre (54,40)
    #   _btnNext  (80,33,28×14)  → centre (94,40)
    #   _btnArmor (130,33,110×14)→ centre (185,40)
    ("soldier_info_prev",  14,    40, 1),  # 81
    ("soldier_info_ok",    54,    40, 1),  # 82
    ("soldier_info_next",  94,    40, 1),  # 83
    ("soldier_info_armor", 185,   40, 1),  # 84

    # ── SoldierArmorState (popup 192×160 at 64,20) ─────────────────────────────────
    # List at (73,68,160×80); row height ≈8 px; centre_x=153.
    #   _btnCancel TextButton(140,16,90,156)→ centre (160,164)
    ("soldier_armor_row0",   153,  72, 1),  # 85
    ("soldier_armor_row1",   153,  80, 1),  # 86
    ("soldier_armor_row2",   153,  88, 1),  # 87
    ("soldier_armor_row3",   153,  96, 1),  # 88
    ("soldier_armor_row4",   153, 104, 1),  # 89
    ("soldier_armor_row5",   153, 112, 1),  # 90
    ("soldier_armor_cancel", 160, 164, 1),  # 91

    # ── CraftsState ───────────────────────────────────────────────────────────────
    # List at (8,58,288×118); row height ≈8 px; centre_x=152.
    #   _btnOk TextButton(288,16,16,176)→ centre (160,184)
    ("crafts_ok",          160,  184, 1),  # 92
    ("crafts_row0",        152,   62, 1),  # 93 → CraftInfoState
    ("crafts_row1",        152,   70, 1),  # 94
    ("crafts_row2",        152,   78, 1),  # 95
    ("crafts_row3",        152,   86, 1),  # 96

    # ── CraftInfoState (320×200 POPUP_BOTH) ───────────────────────────────────────
    #   _btnOk    (128,168,64×24)→ centre (160,180)
    #   _btnCrew  (14,96,64×16)  → centre (46,104)
    #   _btnEquip (14,120,64×16) → centre (46,128)
    #   _btnArmor (14,144,64×16) → centre (46,152)
    ("craft_info_ok",      160,  180, 1),  # 97
    ("craft_info_crew",     46,  104, 1),  # 98
    ("craft_info_equip",    46,  128, 1),  # 99 → CraftEquipmentState (HWPs + items)
    ("craft_info_armor",    46,  152, 1),  # 100

    # ── CraftEquipmentState ───────────────────────────────────────────────────────
    # List at (8,40,288×128); ARROW_HORIZONTAL at list-x+203=screen_x 211.
    # Right-arrow (add to craft) centre at screen_x=232 (211+14+7); row height ≈8 px.
    #   _btnOk        (164,176,148×16)→ centre (238,184)
    #   _btnInventory (8,176,148×16)  → centre (82,184)
    ("craft_equip_ok",     238,  184, 1),  # 101
    ("craft_equip_inv",     82,  184, 1),  # 102
    ("craft_equip_row0",   120,   44, 1),  # 103 — select item row 0
    ("craft_equip_row1",   120,   52, 1),  # 104
    ("craft_equip_row2",   120,   60, 1),  # 105
    ("craft_equip_row3",   120,   68, 1),  # 106
    ("craft_equip_row4",   120,   76, 1),  # 107
    ("craft_equip_add0",   232,   44, 1),  # 108 — right-arrow: +1 of row 0 to craft
    ("craft_equip_add1",   232,   52, 1),  # 109
    ("craft_equip_add2",   232,   60, 1),  # 110
    ("craft_equip_add3",   232,   68, 1),  # 111
    ("craft_equip_add4",   232,   76, 1),  # 112

    # ── DismantleFacilityState (popup 152×80 at 20,60) ────────────────────────────
    #   _btnOk     (36,115,44×16)→ centre (58,123)
    #   _btnCancel (112,115,44×16)→ centre (134,123)
    ("dismantle_ok",        58,  123, 1),  # 113
    ("dismantle_cancel",   134,  123, 1),  # 114

    # ── DogfightState (160×96 overlay at x=0, y=0; _x=_y=0 hardcoded) ────────────
    # Buttons (all absolute):
    #   _btnStandoff   (83,4,  36×15) → centre (101, 11)
    #   _btnCautious   (120,4, 36×15) → centre (138, 11)
    #   _btnStandard   (83,20, 36×15) → centre (101, 27)
    #   _btnAggressive (120,20,36×15) → centre (138, 27)
    #   _btnDisengage  (120,36,36×15) → centre (138, 43)
    ("dogfight_standoff",   101,   11, 1),  # 115
    ("dogfight_cautious",   138,   11, 1),  # 116
    ("dogfight_standard",   101,   27, 1),  # 117
    ("dogfight_aggressive", 138,   27, 1),  # 118
    ("dogfight_disengage",  138,   43, 1),  # 119

    # ── ManufactureState (hint="manufacture") ──────────────────────────────────
    # _btnNew  TextButton(148,16,  8,176) → centre (84, 184)
    # _btnOk   TextButton(148,16,164,176) → centre (240,184)
    # _lstManufacture TextList(288,88, 8,80) → row height 10px, x-centre 152
    ("manufacture_new_item",   84,  184, 1),  # 120 — New Item → NewManufactureListState
    ("manufacture_ok",        240,  184, 1),  # 121 — OK → pop back to basescape
    ("manufacture_row0",      152,   85, 1),  # 122 — edit existing job row 0
    ("manufacture_row1",      152,   95, 1),  # 123
    ("manufacture_row2",      152,  105, 1),  # 124
    ("manufacture_row3",      152,  115, 1),  # 125
    ("manufacture_row4",      152,  125, 1),  # 126

    # ── NewManufactureListState (hint="manufacture", no setScreenHint) ─────────
    # popup (0,22,320,156); _lstManufacture TextList(288,80, 8,70) → rows y=75+
    # _btnOK TextButton(304,16, 8,154) → centre (160,162)
    ("mfg_list_row0",         152,   75, 1),  # 127
    ("mfg_list_row1",         152,   85, 1),  # 128
    ("mfg_list_row2",         152,   95, 1),  # 129
    ("mfg_list_row3",         152,  105, 1),  # 130
    ("mfg_list_row4",         152,  115, 1),  # 131
    ("mfg_list_row5",         152,  125, 1),  # 132
    ("mfg_list_row6",         152,  135, 1),  # 133
    ("mfg_list_row7",         152,  145, 1),  # 134
    ("mfg_list_cancel",       160,  162, 1),  # 135 — OK/close without selecting

    # ── ManufactureStartState (hint="manufacture", no setScreenHint) ───────────
    # _btnCancel TextButton(136,16, 16,155) → centre (84, 163)
    # _btnStart  TextButton(136,16,168,155) → centre (236,163)
    ("mfg_start",             236,  163, 1),  # 136 — Start Production → ManufactureInfoState
    ("mfg_start_cancel",       84,  163, 1),  # 137 — Cancel

    # ── ManufactureInfoState (hint="manufacture", no setScreenHint) ────────────
    # _btnOk   TextButton(136,16,168,155) → centre (236,163)
    # _btnStop TextButton(136,16, 16,155) → centre (84, 163)
    # _btnEngUp   ArrowButton(13,14,132,114) → centre (138,121)
    # _btnEngDown ArrowButton(13,14,132,136) → centre (138,143)
    ("mfg_info_ok",           236,  163, 1),  # 138 — confirm / exit info screen
    ("mfg_info_stop",          84,  163, 1),  # 139 — stop production + exit
    ("mfg_eng_up",            138,  121, 1),  # 140 — +1 engineer (left-click)
    ("mfg_eng_down",          138,  143, 1),  # 141 — -1 engineer (left-click)

    # ── BuildFacilitiesState (hint="build_facilities") ─────────────────────────
    # popup Window(192,40,128,160); _lstFacilities TextList(104,104,200,64)
    # rows at x-centre 252; row height 10px starting y=69
    # _btnOk TextButton(112,16,200,176) → centre (256,184)  [acts as Cancel/close]
    ("build_row0",            252,   69, 1),  # 142 — select facility type row 0
    ("build_row1",            252,   79, 1),  # 143
    ("build_row2",            252,   89, 1),  # 144
    ("build_row3",            252,   99, 1),  # 145
    ("build_row4",            252,  109, 1),  # 146
    ("build_row5",            252,  119, 1),  # 147
    ("build_ok",              256,  184, 1),  # 148 — close list (no build)

    # ── PlaceFacilityState (hint="place_facility") ─────────────────────────────
    # BaseView at (0,8,192,192); cells 32×32px; cell (col,row) centre = (16+32c, 24+32r)
    # N = row×6 + col; 36 actions cover all cells (indices 150–185).
    # _btnCancel TextButton(112,16,200,176) → centre (256,184)
    ("place_cancel",          256,  184, 1),  # 149 — cancel placement
    ("place_pos0",             16,   24, 1),  # 150 — row 0 col 0
    ("place_pos1",             48,   24, 1),  # 151 — row 0 col 1
    ("place_pos2",             80,   24, 1),  # 152 — row 0 col 2
    ("place_pos3",            112,   24, 1),  # 153 — row 0 col 3
    ("place_pos4",            144,   24, 1),  # 154 — row 0 col 4
    ("place_pos5",            176,   24, 1),  # 155 — row 0 col 5
    ("place_pos6",             16,   56, 1),  # 156 — row 1 col 0
    ("place_pos7",             48,   56, 1),  # 157 — row 1 col 1
    ("place_pos8",             80,   56, 1),  # 158 — row 1 col 2
    ("place_pos9",            112,   56, 1),  # 159 — row 1 col 3
    ("place_pos10",           144,   56, 1),  # 160 — row 1 col 4
    ("place_pos11",           176,   56, 1),  # 161 — row 1 col 5
    ("place_pos12",            16,   88, 1),  # 162 — row 2 col 0
    ("place_pos13",            48,   88, 1),  # 163 — row 2 col 1
    ("place_pos14",            80,   88, 1),  # 164 — row 2 col 2
    ("place_pos15",           112,   88, 1),  # 165 — row 2 col 3
    ("place_pos16",           144,   88, 1),  # 166 — row 2 col 4
    ("place_pos17",           176,   88, 1),  # 167 — row 2 col 5
    ("place_pos18",            16,  120, 1),  # 168 — row 3 col 0
    ("place_pos19",            48,  120, 1),  # 169 — row 3 col 1
    ("place_pos20",            80,  120, 1),  # 170 — row 3 col 2
    ("place_pos21",           112,  120, 1),  # 171 — row 3 col 3
    ("place_pos22",           144,  120, 1),  # 172 — row 3 col 4
    ("place_pos23",           176,  120, 1),  # 173 — row 3 col 5
    ("place_pos24",            16,  152, 1),  # 174 — row 4 col 0
    ("place_pos25",            48,  152, 1),  # 175 — row 4 col 1
    ("place_pos26",            80,  152, 1),  # 176 — row 4 col 2
    ("place_pos27",           112,  152, 1),  # 177 — row 4 col 3
    ("place_pos28",           144,  152, 1),  # 178 — row 4 col 4
    ("place_pos29",           176,  152, 1),  # 179 — row 4 col 5
    ("place_pos30",            16,  184, 1),  # 180 — row 5 col 0
    ("place_pos31",            48,  184, 1),  # 181 — row 5 col 1
    ("place_pos32",            80,  184, 1),  # 182 — row 5 col 2
    ("place_pos33",           112,  184, 1),  # 183 — row 5 col 3
    ("place_pos34",           144,  184, 1),  # 184 — row 5 col 4
    ("place_pos35",           176,  184, 1),  # 185 — row 5 col 5

    # ── InterceptState extra craft rows ────────────────────────────────────────
    # _lstCrafts TextList(288,64, 8,78); row height 10px; centre_x=152
    # Row 0 already at action 15 (intercept_craft1, y=82)
    ("intercept_craft2",      152,   93, 1),  # 186 — craft row 1
    ("intercept_craft3",      152,  103, 1),  # 187 — craft row 2
    ("intercept_craft4",      152,  113, 1),  # 188 — craft row 3

    # ── CraftSoldiersState (hint="craft_soldiers") ─────────────────────────────
    # _lstSoldiers TextList(288,128, 8,40); row height 10px; centre_x=152
    # _btnOk TextButton(148,16,164,176) → centre (240,184)
    ("craft_soldiers_ok",     240,  184, 1),  # 189 — confirm assignments
    ("craft_soldiers_row0",   152,   45, 1),  # 190 — toggle soldier 0 assignment
    ("craft_soldiers_row1",   152,   55, 1),  # 191
    ("craft_soldiers_row2",   152,   65, 1),  # 192
    ("craft_soldiers_row3",   152,   75, 1),  # 193
    ("craft_soldiers_row4",   152,   85, 1),  # 194

    # ── InterceptState — go-to-base button ────────────────────────────────────
    # _btnGotoBase TextButton(142,16, 162,146) → centre (233,154)
    # Only visible when InterceptState called from a base (not from UfoDetectedState).
    ("intercept_goto_base",   233,  154, 1),  # 195

    # ── ManufactureInfoState — units-to-produce / sell controls ──────────────
    # _btnUnitUp   ArrowButton(13,14, 284,114) → centre (290,121)
    # _btnUnitDown ArrowButton(13,14, 284,136) → centre (290,143)
    # _btnSell     ToggleTextButton(60,16, 244,61) → centre (274,69)
    ("mfg_unit_up",           290,  121, 1),  # 196 — +1 unit to produce
    ("mfg_unit_down",         290,  143, 1),  # 197 — -1 unit to produce
    ("mfg_sell_toggle",       274,   69, 1),  # 198 — toggle sell-on-completion

    # ── Formerly menu_load_game — loading permanently disabled ───────────────
    # Repurposed as a second wait no-op.  Earns no reward, does nothing.
    ("wait_2",               None, None, 1),  # 199 — no-op (formerly menu_load_game)

    # ── PurchaseState decrease (▼) buttons ───────────────────────────────────
    # a2 at x=8+227+12=247 w=11→centre 252; same row y values as up buttons.
    # Appended here so indices 0-199 remain stable (checkpoint compatibility).
    ("purchase_row_down0", 252,    58, 1),  # 200 — −1 qty row 0
    ("purchase_row_down1", 252,    66, 1),  # 201
    ("purchase_row_down2", 252,    74, 1),  # 202
    ("purchase_row_down3", 252,    82, 1),  # 203
    ("purchase_row_down4", 252,    90, 1),  # 204
    ("purchase_row_down5", 252,    98, 1),  # 205

    # ── SellState decrease (▼) buttons ───────────────────────────────────────
    # a2 at x=8+182+12=202 w=11→centre 207.
    ("sell_row_down0",     207,    58, 1),  # 206 — −1 sell qty row 0
    ("sell_row_down1",     207,    66, 1),  # 207
    ("sell_row_down2",     207,    74, 1),  # 208
    ("sell_row_down3",     207,    82, 1),  # 209
    ("sell_row_down4",     207,    90, 1),  # 210
    ("sell_row_down5",     207,    98, 1),  # 211
]

N_ACTIONS    = len(ACTIONS)
ACTION_NAMES = [a[0] for a in ACTIONS]

# Sentinel value used in the ACTIONS table for random_click
_RANDOM_CLICK_SENTINEL = -1

# Full game-screen bounds (logical pixels, 320×200 native resolution)
SCREEN_W = 320
SCREEN_H = 200

# Screen hints → integer for obs encoding
SCREEN_TYPES: dict[str, int] = {
    "geoscape":         0,
    "ufo_detected":     1,
    "mission_detected": 2,
    "intercept":        3,
    "confirm_dest":     4,
    "low_fuel":         5,
    "craft_error":      6,
    "monthly_report":   7,
    "items_arriving":   8,
    "craft_patrol":     9,
    "other":            10,
    "main_menu":        11,   # engine at start/main-menu (no SavedGame yet)
    "basescape":        12,   # base management screen (after lift placement)
    "research_new":     13,   # NewResearchListState — "New Project" popup list
    "research_info":    14,   # ResearchInfoState    — scientist assignment
    "purchase":         15,   # PurchaseState — buy items / hire personnel
    "sell":             16,   # SellState     — sell items / sack personnel
    "soldiers":         17,   # SoldiersState — soldier roster list
    "soldier_info":     18,   # SoldierInfoState — individual soldier stats
    "crafts":           19,   # CraftsState   — craft list at base
    "craft_info":       20,   # CraftInfoState — single craft details
    "craft_soldiers":   21,   # CraftSoldiersState — assign soldiers to craft
    "craft_equipment":  22,   # CraftEquipmentState — load items / HWPs onto craft
    "soldier_armor":    23,   # SoldierArmorState — choose armor for a soldier
    "dismantle_facility": 24, # DismantleFacilityState — confirm facility removal
    "dogfight":         25,   # DogfightState — intercept battle controls
    "ufopaedia":        26,   # ArticleState  — UFOpedia article navigation
    "place_facility":   27,   # PlaceFacilityState — click grid to place a facility
    "manufacture":      28,   # ManufactureState + sub-states (all share this hint)
    "build_facilities": 29,   # BuildFacilitiesState — facility type selection list
    "research":         30,   # ResearchState — active research list
    "briefing":         31,   # BriefingState — pre-mission briefing screen
    "debriefing":       32,   # DebriefingState — post-mission debrief screen
    "statistics":       33,   # StatisticsState — defeat/victory end-game summary
}
N_SCREEN_TYPES = max(SCREEN_TYPES.values()) + 1

# Screens the agent intentionally navigates and may dwell on for many steps.
# The same-screen anti-stuck counter resets on any of these, just like geoscape.
_FREE_SCREENS = frozenset({
    "geoscape", "ufo_detected", "mission_detected", "intercept", "confirm_dest",
    "low_fuel", "craft_error", "craft_patrol", "items_arriving", "dogfight",
    "basescape", "soldiers", "soldier_info", "soldier_armor",
    "crafts", "craft_info", "craft_soldiers", "craft_equipment",
    "research", "research_new", "research_info",
    "manufacture", "build_facilities", "place_facility", "dismantle_facility",
    "purchase", "sell",
})

# Screens where 4×dismiss unwinding is the right recovery when the engine goes silent.
# Geoscape-adjacent screens are excluded — dismisses there would cancel the wrong thing.
_DISMISS_STACK = frozenset({
    "basescape", "soldiers", "soldier_info", "soldier_armor",
    "crafts", "craft_info", "craft_soldiers", "craft_equipment",
    "research", "research_new", "research_info",
    "manufacture", "build_facilities", "place_facility", "dismantle_facility",
    "purchase", "sell", "ufopaedia",
})

# Actions that commit to a strategic decision (resource spend, dispatch, confirm).
# These earn COMMIT_CLICK_REWARD (2×) instead of CLICK_REWARD.
# Deliberately excludes cancel/back buttons and informational navigation.
_COMMIT_ACTIONS: frozenset[int] = frozenset({
    10,                    # ufo_intercept — send a craft to a UFO
    13,                    # mission_intercept — send craft to mission site
    17,                    # confirm_ok — lock in craft destination
    47,                    # research_info_ok — start/update research project
    58,                    # purchase_ok — commit purchase
    97,                    # craft_info_ok — confirm craft assignment
    101,                   # craft_equip_ok — lock loadout
    113,                   # dismantle_ok — confirm facility dismantle
    121,                   # manufacture_ok — confirm manufacture settings
    136,                   # mfg_start — start a new production run
    138,                   # mfg_info_ok — update running production
    189,                   # craft_soldiers_ok — lock in crew assignment
    *range(142, 148),      # build_row0–5 — select a facility type to build
    *range(150, 186),      # place_pos0–35 — place a facility on the base grid (all 36 cells)
})

# ── Battlescape scoring reference (XCOM wiki §4.4) ────────────────────────────
# These are the game's internal point values that feed into total_score after a
# mission ends.  We don't parse them per-kill; instead we detect the post-battle
# score delta and apply BATTLE_SCORE_SCALE (see step() reward logic).
#
# Alien kill points (stun = 2× kill value):
#   Sectoid Soldier 10 | Sectoid Navigator 14 | Sectoid Medic 16
#   Sectoid Engineer 18 | Sectoid Leader 25 | Sectoid Commander 35
#   Floater Soldier 14 | Floater Navigator 15 | Floater Medic 16
#   Floater Engineer 18 | Floater Leader 25 | Floater Commander 35
#   Snakeman Soldier 15 | Snakeman Navigator 17 | Snakeman Engineer 20
#   Snakeman Leader 28 | Snakeman Commander 36
#   Ethereal Soldier 20 | Ethereal Leader 30 | Ethereal Commander 40
#   Muton Soldier 15 | Muton Navigator 17 | Muton Engineer 19
#   Reaper 25 | Chryssalid 25 | Silacoid 20 | Celatid 20
#   Sectopod 30 | Cyberdisc 20 | Zombie 18
#
# Mission outcome points:
#   Civilian killed by X-COM  -50 | Civilian killed by Aliens -30
#   X-COM Soldier MIA         -20 | X-COM Soldier Killed      -20*
#   HWP Destroyed             -20 | Civilian Saved            +30
#   Alien Base Destroyed      +500
#
# (* penalty scales up with soldier rank/experience)
#
# All of the above accumulate into total_score when the geoscape resumes.
# A delta >= BATTLE_SCORE_THRESHOLD on a single step is treated as a
# mission-end event and scaled by BATTLE_SCORE_SCALE instead of SCORE_SCALE.
BATTLE_SCORE_THRESHOLD = 10   # points; any |delta| >= this → battle scoring
BATTLE_SCORE_SCALE     = 2.0  # 1 game-point = 2 reward units after a battle

# ── Research completion scoring reference (XCOM wiki §4.8) ───────────────────
# Points feed into save->getResearchScores(), which we broadcast as the "score"
# field.  Any positive delta in that field means a project just completed.
# Applied at RESEARCH_SCORE_SCALE = 1.0  (same philosophy as battle scoring).
#
#  5 pts : Plasma Pistol Clip
#  6 pts : Plasma Rifle Clip
#  7 pts : Heavy Plasma Clip
# 10 pts : Blaster Bomb | Craft Laser Cannon | Heavy Laser | Laser Pistol
#          Laser Rifle | Laser Weapons | Stun Bomb
# 15 pts : Laser Defenses | Tank/Laser Cannon
# 20 pts : Alien Grenade | Flying Suit | Medi-Kit | Motion Scanner
#          Personal Armor | Plasma Pistol | Power Suit | Psi-Amp
# 25 pts : Craft Fusion Weapon | Craft Plasma Weapon | Fusion Defenses
#          Grav Shield | Hovertank/Launcher | Hovertank/Plasma
#          Hyper-Wave Decoder | Mind Probe | Mind Shield | Plasma Defenses
#          Plasma Rifle | Psi-Lab
# 30 pts : Alien Abduction | Alien Alloys | Alien Entertainment | Alien Food
#          Alien Reproduction | Alien Surgery | Heavy Plasma
#          New Fighter Craft (Firestorm) | New Fighter-Transporter (Lightning)
#          Small Launcher | UFO Construction | UFO Navigation | UFO Propulsion
#          Ultimate Craft (Avenger)
# 40 pts : Blaster Launcher | Elerium-115
# 50 pts : Alien Autopsies (each type) | Alien Interrogations (per new entry)
# 60 pts : Alien Origins | Cydonia or Bust | The Martian Solution
RESEARCH_SCORE_SCALE = 3.0  # 1 research-point = 3 reward units

MAX_STEPS = 75_000   # per-episode truncation (1-min steps, 31 days ≈ 44 640)

END_NONE = 0
END_WIN  = 1
END_LOSE = 2

_CRAFT_STATUS_INT: dict[str, int] = {
    "STR_READY":      0,
    "STR_OUT":        1,
    "STR_REPAIRS":    2,
    "STR_REFUELLING": 2,
    "STR_REARMING":   2,
}

# ── bridge client ──────────────────────────────────────────────────────────────

class _BridgeClient:
    def __init__(self, socket_path: str) -> None:
        self._sock = _ws.open_socket(socket_path)
        self._buf  = b""

    def close(self) -> None:
        try:
            _ws.close_socket(self._sock)
        except Exception:
            pass

    def send(self, **kw) -> None:
        try:
            _ws.send_bytes(self._sock, (json.dumps(kw) + "\n").encode())
        except OSError as exc:
            raise ConnectionError(f"engine send failed: {exc}") from exc

    def drain_buf(self) -> None:
        """Discard the in-memory receive buffer.

        The visible-mode geoscape timer broadcasts state every 10ms.  Over a
        long episode (thousands of steps) thousands of tick messages accumulate
        in _buf because each step only consumes one.  Call this before
        sending a command that expects a specific response (e.g. new_game_ready)
        so the search loop is not swamped by stale tick events.
        """
        self._buf = b""

    def recv_one(self, timeout: float = 10.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Drain any complete messages already in the local buffer
            # before waiting on the socket for more data.
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                text = line.strip()
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
            # Buffer exhausted — wait for the engine to send more.
            ms = max(1, int((deadline - time.time()) * 1000))
            if not _ws.recv_ready(self._sock, timeout_ms=min(ms, 500)):
                continue
            chunk = _ws.recv_bytes(self._sock)
            if chunk is None:
                raise ConnectionError("engine closed bridge socket")
            self._buf += chunk
        return None

    def next_state(self, timeout: float = 15.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv_one(timeout=max(0.05, deadline - time.time()))
            if msg is None:
                break
            if msg.get("type") == "state":
                return msg
        return None


# ── environment ────────────────────────────────────────────────────────────────

class XcomInterceptionEnv(gym.Env):
    """
    OpenXcom strategic-layer episode driven entirely by simulated GUI clicks.

    Observation space (Dict):
      funds            Box int32 [0, 100 000 000]
      score            Box int32 [-10 000, 10 000]
      total_score      Box int32 [-100 000, 100 000]
      month            Box int32 [1, 12]
      day              Box int32 [1, 31]
      hour             Box int32 [0, 23]
      craft_status     Discrete(3)    0=ready 1=out 2=repair/refuel
      craft_fuel_pct   Box int32 [0, 100]
      craft_damage_pct Box int32 [0, 100]
      ufo_detected     Discrete(2)    1 when ≥1 UFO on radar
      mission_site     Discrete(2)    1 when a terror/mission site exists
      popup_depth      Box int32 [0, 10]
      screen_type      Discrete(34)   which dialog is on top (see SCREEN_TYPES)
      ending           Discrete(3)    0=ongoing 1=won 2=lost
      num_soldiers     Box int32 [0, 50]

    Action space: Discrete(212)  — see ACTIONS table above
    """

    metadata = {"render_modes": []}

    # Reward constants
    SCORE_SCALE          = 1.0    # strategic score ticks — 1 game-point = 1 reward unit
    BATTLE_SCORE_SCALE   = 2.0    # post-battle / end-of-month total_score delta (§4.4)
    RESEARCH_SCORE_SCALE = 3.0    # research completion score delta (§4.8)
    CLICK_REWARD         = 0.01   # named-button mouse click (navigation/cancel/info)
    COMMIT_CLICK_REWARD  = 0.02   # clicks that commit to a strategic decision (2× navigate)
    RANDOM_CLICK_REWARD  = 0.003  # random/right-click exploration incentive (< nav click)
    WAIT_PENALTY         = -0.005 # per "wait" action — discourages idle polling in menus
    UFO_BONUS            = 10.0   # first tick with a detected UFO
    LAUNCH_BONUS         = 50.0   # craft transitions to STR_OUT
    CYDONIA_BONUS        = 1000.0
    MENU_ESCAPE_REWARD   = 5.0    # screen transitions from main_menu → geoscape
    RESEARCH_START_REWARD  = 25.0   # per new research project started
    RESEARCH_CANCEL_PENALTY = -25.0 # per research project cancelled (count drop with no score delta)
    HIRE_REWARD           = 0.5   # per scientist or engineer hired
    BUILD_REWARD          = 25.0  # per facility construction started
    PURCHASE_REWARD       = 0.05  # per net-new item in stores
    TIME_REWARD_PER_MINUTE = 0.0002  # per in-game minute elapsed
    SHOT_REWARD            = 2.0    # per weapon fired in battlescape (shoot event)
    ALIEN_HIT_REWARD       = 0.5    # per HP of damage dealt to alien units
    LOAD_PENALTY           = -50.0   # attempting to load a saved game
    ABORT_PENALTY          = -200.0  # abandoning the game mid-episode
    OPTIONS_MENU_PENALTY   = -500.0  # opening the pause/options menu
    FUNDS_SPEND_SCALE      = 0.00005 # per dollar spent; $500K spent → +25 reward
    EXPLORE_REWARD_PER_TILE = 0.5   # per newly uncovered tile (fog-of-war)
    ALIEN_SPOT_REWARD       = 3.0   # per alien spotted for the first time this battle
    PICKUP_REWARD           = 2.0   # per item picked up from the ground
    PROMOTION_REWARD        = 50.0  # per soldier promoted in rank
    DISMANTLE_PENALTY       = -20.0 # per non-lift facility dismantled
    NEW_BASE_REWARD         = 500.0 # build additional base when first base is filled
    BASE_FILLED_SQUARES     = 30    # first base is "filled" when ≥30 of 36 squares occupied
    FACILITY_ERROR_PENALTY  = -5.0  # clicking an invalid spot in PlaceFacilityState
    ALIEN_BASE_FOUND_REWARD = 50.0  # first time an alien base is discovered
    MISSION_DISPATCH_REWARD   = 75.0   # transport dispatched to terror site / grounded UFO / alien base
    UFO_SHOOT_DOWN_REWARD     = 30.0   # UFO transitions FLYING → CRASHED or DESTROYED
    MANUFACTURE_START_REWARD  = 20.0   # per new manufacturing project started
    UFOPAEDIA_PENALTY         = -10.0  # entering the UFOpedia (auto-dismissed; discourages navigation there)

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        *,
        headless: bool = True,
        socket_name: str = "openxcom_bridge.sock",
        speed_ms: int = 0,
        prefill_ufo: bool = False,        # no-op — kept for API compatibility
        autoload_save: Optional[str] = None,  # save file base name to load each episode
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Dict({
            "funds":            spaces.Box(0, 100_000_000, shape=(1,), dtype=np.int32),
            "score":            spaces.Box(-10_000, 10_000, shape=(1,), dtype=np.int32),
            "total_score":      spaces.Box(-100_000, 100_000, shape=(1,), dtype=np.int32),
            "month":            spaces.Box(1, 12,  shape=(1,), dtype=np.int32),
            "day":              spaces.Box(1, 31,  shape=(1,), dtype=np.int32),
            "hour":             spaces.Box(0, 23,  shape=(1,), dtype=np.int32),
            "craft_status":     spaces.Discrete(3),
            "craft_fuel_pct":   spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "craft_damage_pct": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "ufo_detected":     spaces.Discrete(2),
            "mission_site":     spaces.Discrete(2),
            "popup_depth":      spaces.Box(0, 10,  shape=(1,), dtype=np.int32),
            "screen_type":      spaces.Discrete(N_SCREEN_TYPES),
            "ending":           spaces.Discrete(3),
            "num_soldiers":     spaces.Box(0, 50,  shape=(1,), dtype=np.int32),
            "base_grid":        spaces.Box(0, 2,   shape=(36,), dtype=np.int8),
            # 36-element flat 6×6 grid; N=row×6+col; 0=empty, 1=under construction, 2=built
        })
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode  = render_mode
        self.max_steps    = max_steps

        self._headless       = headless
        self._socket_name    = socket_name
        self._speed_ms       = speed_ms
        self._autoload_save: Optional[str] = autoload_save

        self._proc:             Optional[subprocess.Popen] = None
        self._client:           Optional[_BridgeClient]    = None
        self._last_state:       Optional[Dict]             = None
        self._total_score_prev:    int   = 0
        self._research_score_prev: int   = 0
        self._step_count:          int   = 0
        self._ep_reward:           float = 0.0
        self._prev_craft_out:      bool  = False
        self._prev_ufo_seen:       bool  = False
        self._prev_at_menu:        bool  = False
        self._initial_launch_done: bool  = False
        self._prev_research_count:      int = 0
        self._prev_scientists:          int = 0
        self._prev_engineers:           int = 0
        self._prev_facilities_building: int = 0
        self._prev_store_total:         int = 0
        self._prev_game_minutes:        int = 0
        self._battle_reward_pending:  float = 0.0
        self._prev_alien_total_hp:      int = 0
        self._pending_load_penalty:    float = 0.0
        self._pending_options_penalty: float = 0.0
        self._pending_abort_penalty:  float = 0.0  # carries into the NEXT episode's first step
        self._dismiss_next_basescape:  bool = False
        self._prev_funds:               int = 0
        self._spotted_aliens:           set = set()
        self._prev_explored_tiles:      int = 0
        self._prev_soldier_ranks:      Dict[str, str] = {}
        self._prev_base_count:          int = 0
        self._prev_facilities_total:    int = 0
        self._facility_error_pending:  bool = False
        self._new_game_in_flight:      bool = False
        self._prev_alien_bases_found:   int = 0
        self._dispatched_crafts:        set = set()
        self._prev_manufacture_count:   int = 0
        self._prev_in_ufopaedia:       bool = False
        self._prev_ufo_statuses:       Dict[int, int] = {}
        self._stuck_screen_count:       int = 0
        self._stuck_screen_hint:        str = ""

        self._action_window: deque = deque(maxlen=200)
        self._reward_window: deque = deque(maxlen=60)

        # Persists after reset so eval runners can read it via get_attr after a done step
        self.last_ep_info: dict = {}
        self._ep_tally_start: dict = {}

        self._reward_tally: dict = {k: {"n": 0, "total": 0.0} for k in (
            "click_nav",      "click_commit",   "click_random",   "wait",
            "ufopaedia",      "score_routine",  "score_battle",
            "score_research", "ufo_detect",     "craft_launch",
            "menu_escape",    "abort",          "load",
            "options",        "research_start", "mfg_start",
            "hire",           "build",          "purchase",
            "time",           "funds_spent",    "promo",
            "research_cancel", "dismantle",     "new_base",       "fac_error",
            "alien_base",     "mission",        "shoot_down",
            "battle",         "cydonia",
        )}

        try:
            rank = int(socket_name.replace("openxcom_bridge_", "").replace(".sock", ""))
        except ValueError:
            rank = 0
        self._rank        = rank
        self._episode_num = 0
        os.makedirs(_LOG_DIR, exist_ok=True)
        self._log_path = os.path.join(_LOG_DIR, f"env_{rank}.jsonl")

        # Clean up any .tmp left by a previous crash before the first write.
        try:
            os.remove(os.path.join(_LOG_DIR, f"live_env_{rank}.json.tmp"))
        except FileNotFoundError:
            pass

        self.reward_scale_nav       = 1.0
        self.reward_scale_milestone = 1.0
        self.reward_scale_penalty   = 1.0

        self._action_hist: Dict[str, int] = {}
        self._screen_hist: Dict[str, int] = {}
        self._stuck_total: int = 0

        atexit.register(self._teardown)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._proc is not None and self._client is not None:
            if not self._new_game():
                self._teardown()
                self._launch_with_retry()
        else:
            self._teardown()
            self._launch_with_retry()

        state = self._last_state
        game  = state.get("game", {})
        self._total_score_prev    = game.get("total_score", 0)
        self._research_score_prev = game.get("score", 0)
        self._step_count          = 0
        self._ep_reward           = 0.0
        if self._client is not None:
            try:
                self._client.send(action="set_reward", value=0)
            except ConnectionError:
                pass
        self._prev_craft_out      = False
        self._prev_ufo_seen       = False
        self._prev_at_menu        = (game.get("screen_hint") == "main_menu")
        self._prev_research_count        = self._count_research(state)
        self._prev_scientists, self._prev_engineers = self._count_staff(state)
        self._prev_facilities_building   = self._count_facilities_building(state)
        self._prev_store_total           = self._count_store_total(state)
        self._prev_game_minutes          = self._to_game_minutes(state)
        self._pending_load_penalty       = 0.0
        self._pending_options_penalty    = 0.0
        self._dismiss_next_basescape     = False
        self._prev_funds                 = game.get("funds", 0)
        self._spotted_aliens             = set()
        self._prev_explored_tiles        = 0
        self._prev_soldier_ranks         = self._get_soldier_ranks(state)
        self._prev_base_count            = len(state.get("bases", []))
        self._prev_facilities_total      = self._count_facilities_total(state)
        self._facility_error_pending     = False
        self._new_game_in_flight         = False
        self._prev_alien_bases_found     = self._count_alien_bases_found(state)
        self._dispatched_crafts          = set()
        self._prev_ufo_statuses          = {u.get("idx"): u.get("status", 0) for u in state.get("ufos", [])}
        self._prev_manufacture_count     = self._count_manufacture(state)
        self._prev_in_ufopaedia          = False
        self._stuck_screen_count         = 0
        self._stuck_screen_hint          = ""
        self._episode_num        += 1
        self._ep_tally_start = {k: dict(v) for k, v in self._reward_tally.items()}
        return self._build_obs(state), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._client is None or self._last_state is None:
            raise RuntimeError("call reset() before step()")

        total_score_before = self._total_score_prev

        self._battle_reward_pending = 0.0
        try:
            state = self._execute_action(action)
            state = self._auto_confirm_base_name(state)
        except ConnectionError:
            self._teardown()
            obs = self._build_obs(self._last_state) if self._last_state else self.observation_space.sample()
            self._ep_reward -= 10.0
            self._log_episode(self._last_state or {}, END_NONE, True, False)
            return obs, -10.0, True, False, {"error": "engine_disconnected"}

        self._last_state   = state
        self._step_count  += 1
        game               = state.get("game", {})

        # Track screen distribution for policy health panel.
        _sh = game.get("screen_hint", "geoscape")
        self._screen_hist[_sh] = self._screen_hist.get(_sh, 0) + 1

        # Anti-stuck: if the same transient/unknown screen persists for 64
        # consecutive agent steps the agent is looping without progress —
        # force-escape via 4 dismisses so training continues.
        # Screens in _FREE_SCREENS are intentional navigation destinations and
        # do not count toward the threshold.
        _hint = _sh
        if _hint in _FREE_SCREENS:
            self._stuck_screen_count = 0
            self._stuck_screen_hint  = ""
        elif _hint == self._stuck_screen_hint:
            self._stuck_screen_count += 1
            if self._stuck_screen_count >= 64:
                self._stuck_total += 1
                try:
                    for _ in range(4):
                        self._client.send(action="dismiss")
                    if self._headless:
                        self._client.send(action="step")
                    state = self._next_geostate(timeout=2.0) or state
                    game  = state.get("game", {})
                    self._last_state = state
                except ConnectionError:
                    pass  # engine died; next step()'s outer try/except will teardown
                self._stuck_screen_count = 0
                self._stuck_screen_hint  = ""
        else:
            self._stuck_screen_hint  = _hint
            self._stuck_screen_count = 1

        total_score_now    = game.get("total_score", 0)
        research_score_now = game.get("score", 0)
        ending             = game.get("ending", END_NONE)

        # ── reward components ──────────────────────────────────────────────────
        # Named clicks earn CLICK_REWARD (nav) or COMMIT_CLICK_REWARD (commits).
        # random_click/random_right_click (sentinel x=-1) earn a small exploration
        # reward (RANDOM_CLICK_REWARD < CLICK_REWARD) so the policy still prefers
        # named buttons but has some incentive to explore the UI.
        # Exception: ufopaedia navigation earns no reward (information only).
        name, x, _y, _b = ACTIONS[action]
        is_named_click  = (x is not None and x >= 0)
        is_random_click = (x == _RANDOM_CLICK_SENTINEL)
        in_ufopaedia    = (game.get("screen_hint") == "ufopaedia")
        if is_named_click and not in_ufopaedia:
            if action in _COMMIT_ACTIONS:
                click_rate = self.COMMIT_CLICK_REWARD * self.reward_scale_nav
                self._tally("click_commit", click_rate)
            else:
                click_rate = self.CLICK_REWARD * self.reward_scale_nav
                self._tally("click_nav", click_rate)
        elif is_random_click:
            click_rate = self.RANDOM_CLICK_REWARD * self.reward_scale_nav
            self._tally("click_random", click_rate)
        else:
            click_rate = 0.0
        reward = click_rate
        if name == "wait":
            reward += self.WAIT_PENALTY * self.reward_scale_nav
            self._tally("wait", self.WAIT_PENALTY * self.reward_scale_nav)
        if in_ufopaedia and not self._prev_in_ufopaedia:
            reward += self.UFOPAEDIA_PENALTY
            self._tally("ufopaedia", self.UFOPAEDIA_PENALTY)
        self._prev_in_ufopaedia = in_ufopaedia
        # Abort penalty deferred from the previous episode so it doesn't
        # obscure that episode's ending score.
        _abort = self._pending_abort_penalty
        reward += _abort
        if _abort != 0.0:
            self._tally("abort", _abort)
        self._pending_abort_penalty = 0.0

        # Strategic score: large deltas are post-battle (§4.4), small are routine
        score_delta = total_score_now - total_score_before
        if abs(score_delta) >= BATTLE_SCORE_THRESHOLD:
            _r = score_delta * self.BATTLE_SCORE_SCALE
            reward += _r
            self._tally("score_battle", _r)
        else:
            _r = score_delta * self.SCORE_SCALE
            reward += _r
            self._tally("score_routine", _r)

        # Research completion (§4.8): score only ever increases; any positive
        # delta means a project just finished. Scale at 1 point = 1 reward unit.
        research_delta            = research_score_now - self._research_score_prev
        self._total_score_prev    = total_score_now
        self._research_score_prev = research_score_now
        if research_delta > 0:
            _r = research_delta * self.RESEARCH_SCORE_SCALE
            reward += _r
            self._tally("score_research", _r)

        # UFO first-detection bonus
        ufo_now = bool(state.get("ufos") and any(u.get("detected") for u in state["ufos"]))
        if ufo_now and not self._prev_ufo_seen:
            reward += self.UFO_BONUS * self.reward_scale_milestone
            self._tally("ufo_detect", self.UFO_BONUS * self.reward_scale_milestone)
        self._prev_ufo_seen = ufo_now

        # Craft-launched bonus: any craft transitions to STR_OUT
        craft_out_now = self._any_craft_out(state)
        if craft_out_now and not self._prev_craft_out:
            reward += self.LAUNCH_BONUS * self.reward_scale_milestone
            self._tally("craft_launch", self.LAUNCH_BONUS * self.reward_scale_milestone)
        self._prev_craft_out = craft_out_now

        # Menu-escape bonus: main_menu → geoscape transition
        at_menu_now = (game.get("screen_hint") == "main_menu")
        if self._prev_at_menu and not at_menu_now:
            reward += self.MENU_ESCAPE_REWARD
            self._tally("menu_escape", self.MENU_ESCAPE_REWARD)
        # Abort detection: game returned to main menu unexpectedly mid-episode.
        # Penalty is deferred to the first step of the next episode so the
        # ending score of this run is visible before it is applied.
        if at_menu_now and not self._prev_at_menu and self._step_count > 0:
            self._pending_abort_penalty = self.ABORT_PENALTY * self.reward_scale_penalty
            ending = END_LOSE
        self._prev_at_menu = at_menu_now

        # Pending load penalty (accumulated in _auto_confirm_base_name)
        if self._pending_load_penalty != 0.0:
            self._tally("load", self._pending_load_penalty)
            reward += self._pending_load_penalty
            self._pending_load_penalty = 0.0
        # Pending options/pause-menu penalty
        if self._pending_options_penalty != 0.0:
            self._tally("options", self._pending_options_penalty)
            reward += self._pending_options_penalty
            self._pending_options_penalty = 0.0

        # Research started
        research_now = self._count_research(state)
        _r_new = max(0, research_now - self._prev_research_count)
        if _r_new:
            _r = self.RESEARCH_START_REWARD * _r_new * self.reward_scale_milestone
            reward += _r
            self._tally("research_start", _r, n=_r_new)
        # Cancellation: count dropped without a completion explaining it.
        # If research_delta > 0, at least one project completed this step; treat all
        # count drops as completions to avoid false penalties when multiple projects
        # finish simultaneously.
        _net_drop    = max(0, self._prev_research_count - research_now)
        _completions = _net_drop if research_delta > 0 else 0
        _cancelled   = max(0, _net_drop - _completions)
        if _cancelled:
            _r = self.RESEARCH_CANCEL_PENALTY * _cancelled
            reward += _r
            self._tally("research_cancel", _r, n=_cancelled)
        self._prev_research_count = research_now

        # Manufacturing started
        mfg_now = self._count_manufacture(state)
        _mfg_new = max(0, mfg_now - self._prev_manufacture_count)
        if _mfg_new:
            _r = self.MANUFACTURE_START_REWARD * _mfg_new
            reward += _r
            self._tally("mfg_start", _r, n=_mfg_new)
        self._prev_manufacture_count = mfg_now

        # Staff hired (scientists + engineers)
        sci_now, eng_now = self._count_staff(state)
        new_staff = max(0, sci_now - self._prev_scientists) + max(0, eng_now - self._prev_engineers)
        if new_staff:
            _r = self.HIRE_REWARD * new_staff
            reward += _r
            self._tally("hire", _r, n=new_staff)
        self._prev_scientists, self._prev_engineers = sci_now, eng_now

        # Facility construction started
        fac_now = self._count_facilities_building(state)
        _fac_new = max(0, fac_now - self._prev_facilities_building)
        if _fac_new:
            _r = self.BUILD_REWARD * _fac_new
            reward += _r
            self._tally("build", _r, n=_fac_new)
        self._prev_facilities_building = fac_now

        # Items purchased / received
        store_now = self._count_store_total(state)
        _store_new = max(0, store_now - self._prev_store_total)
        if _store_new:
            _r = self.PURCHASE_REWARD * _store_new
            reward += _r
            self._tally("purchase", _r, n=_store_new)
        self._prev_store_total = store_now

        # Time progression (proportional to in-game minutes elapsed)
        mins_now = self._to_game_minutes(state)
        _mins_new = max(0, mins_now - self._prev_game_minutes)
        if _mins_new:
            _r = self.TIME_REWARD_PER_MINUTE * _mins_new
            reward += _r
            self._tally("time", _r)
        self._prev_game_minutes = mins_now

        # Money spent (purchases, staff, facilities): funds decrease = spending
        funds_now = game.get("funds", 0)
        funds_spent = self._prev_funds - funds_now
        if funds_spent > 0:
            _r = funds_spent * self.FUNDS_SPEND_SCALE
            reward += _r
            self._tally("funds_spent", _r)
        self._prev_funds = funds_now

        # Soldier promotions: detect rank changes in first base soldiers
        ranks_now = self._get_soldier_ranks(state)
        promotions = sum(
            1 for name, old_r in self._prev_soldier_ranks.items()
            if name in ranks_now and ranks_now[name] != old_r
        )
        if promotions:
            _r = self.PROMOTION_REWARD * promotions
            reward += _r
            self._tally("promo", _r, n=promotions)
        self._prev_soldier_ranks = ranks_now

        # Facility dismantling: net decrease in total facility count = negative reward
        fac_total_now = self._count_facilities_total(state)
        dismantled = max(0, self._prev_facilities_total - fac_total_now)
        if dismantled:
            _r = self.DISMANTLE_PENALTY * dismantled
            reward += _r
            self._tally("dismantle", _r, n=dismantled)
        self._prev_facilities_total = fac_total_now

        # Additional base: large reward only when first base is already filled
        base_count_now = len(state.get("bases", []))
        if base_count_now > self._prev_base_count:
            if self._first_base_occupied_squares(state) >= self.BASE_FILLED_SQUARES:
                reward += self.NEW_BASE_REWARD
                self._tally("new_base", self.NEW_BASE_REWARD)
        self._prev_base_count = base_count_now

        # Facility placement error: clicking an invalid grid spot in PlaceFacilityState
        if self._facility_error_pending:
            reward += self.FACILITY_ERROR_PENALTY
            self._tally("fac_error", self.FACILITY_ERROR_PENALTY)
            self._facility_error_pending = False

        # Alien base discovery: reward when a base transitions to discovered=true
        found_now = self._count_alien_bases_found(state)
        _ab_new = max(0, found_now - self._prev_alien_bases_found)
        if _ab_new:
            _r = self.ALIEN_BASE_FOUND_REWARD * _ab_new
            reward += _r
            self._tally("alien_base", _r, n=_ab_new)
        self._prev_alien_bases_found = found_now

        # Craft dispatch: reward once per unique (craft_id, dest_key) when a craft
        # is sent to a terror site, grounded UFO, or alien base
        for _base in state.get("bases", []):
            for craft in _base.get("crafts", []):
                cid   = craft.get("id")
                dtype = craft.get("dest_type")
                if dtype == "mission_site":
                    key: Optional[tuple] = (cid, "mission_site")
                elif dtype == "alien_base":
                    key = (cid, "alien_base")
                elif dtype == "ufo" and craft.get("dest_status", 0) in (1, 2):
                    key = (cid, "ufo_grounded")
                else:
                    key = None
                if key is not None and key not in self._dispatched_crafts:
                    self._dispatched_crafts.add(key)
                    reward += self.MISSION_DISPATCH_REWARD * self.reward_scale_milestone
                    self._tally("mission", self.MISSION_DISPATCH_REWARD * self.reward_scale_milestone)
        # Expire keys for crafts that have returned (no longer have a destination)
        active_dest_ids = {
            c.get("id")
            for b in state.get("bases", [])
            for c in b.get("crafts", [])
            if c.get("dest_type") is not None
        }
        self._dispatched_crafts = {k for k in self._dispatched_crafts if k[0] in active_dest_ids}

        # UFO shoot-down: FLYING(0) → CRASHED(2) or DESTROYED(3)
        ufo_statuses_now = {u.get("idx"): u.get("status", 0) for u in state.get("ufos", [])}
        for uidx, curr_status in ufo_statuses_now.items():
            prev_status = self._prev_ufo_statuses.get(uidx, 0)
            if prev_status == 0 and curr_status in (2, 3):
                reward += self.UFO_SHOOT_DOWN_REWARD
                self._tally("shoot_down", self.UFO_SHOOT_DOWN_REWARD)
        self._prev_ufo_statuses = ufo_statuses_now

        # Battlescape combat: shots fired and alien HP damage accumulated during
        # battle message processing in _next_geostate.
        if self._battle_reward_pending != 0.0:
            self._tally("battle", self._battle_reward_pending)
        reward += self._battle_reward_pending

        terminated = False
        if ending == END_WIN:
            reward    += self.CYDONIA_BONUS
            self._tally("cydonia", self.CYDONIA_BONUS)
            terminated = True
        elif ending == END_LOSE:
            terminated = True

        truncated = bool(self._step_count >= self.max_steps)

        self._ep_reward += reward
        try:
            self._client.send(action="set_reward", value=int(self._ep_reward))
        except ConnectionError:
            pass
        if terminated or truncated:
            self._log_episode(state, ending, terminated, truncated)
            # Headless only: fire new_game now so the engine pre-warms while SB3
            # does bookkeeping between here and the next reset() call.  The headless
            # engine processes commands in its poll loop independently of "step"
            # tokens, so new_game_ready will be waiting in the socket buffer by the
            # time reset() → _new_game() runs.
            if self._headless and self._client is not None:
                try:
                    self._client.send(action="new_game")
                    self._new_game_in_flight = True
                except Exception:
                    self._new_game_in_flight = False

        self._action_window.append(int(action))
        self._reward_window.append(round(reward, 4))
        _aname = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"act{action}"
        self._action_hist[_aname] = self._action_hist.get(_aname, 0) + 1
        self._write_live_state(reward, game)
        return self._build_obs(state), reward, terminated, truncated, {}

    def close(self) -> None:
        self._teardown()

    # ── action execution ───────────────────────────────────────────────────────

    def _process_battle_map(self, msg: Dict) -> None:
        """Decode the visible tile grid and accumulate exploration + alien-spot rewards."""
        import base64
        map_info = msg.get("map", {})
        W = map_info.get("width",  0)
        L = map_info.get("length", 0)
        vis_b64 = map_info.get("visible", "")
        if not vis_b64 or W == 0 or L == 0:
            return
        try:
            vis = base64.b64decode(vis_b64)
        except Exception:
            return

        # Exploration: bits 0-2 = isDiscovered(0/1/2) — monotonically increasing
        explored = sum(1 for b in vis if (b & 7) != 0)
        new_tiles = max(0, explored - self._prev_explored_tiles)
        self._battle_reward_pending += new_tiles * self.EXPLORE_REWARD_PER_TILE
        self._prev_explored_tiles = max(self._prev_explored_tiles, explored)

        # Alien detection: bit 3 = tile currently in LOS; reward first sighting per alien
        for u in msg.get("units", []):
            if u.get("faction") != 1:
                continue
            uid = u.get("id")
            if uid in self._spotted_aliens:
                continue
            idx = u.get("x", 0) + u.get("y", 0) * W + u.get("z", 0) * W * L
            if 0 <= idx < len(vis) and (vis[idx] & 8) != 0:
                self._spotted_aliens.add(uid)
                self._battle_reward_pending += self.ALIEN_SPOT_REWARD

    def _process_battle_message(self, msg: Dict) -> None:
        event = msg.get("event", "")
        units = msg.get("units", [])

        # Reset per-battle trackers when a new battle begins
        if event == "battle_start":
            self._spotted_aliens      = set()
            self._prev_explored_tiles = 0

        alien_hp_now = sum(u.get("health", 0) for u in units if u.get("faction") == 1)
        if event == "shoot":
            self._battle_reward_pending += self.SHOT_REWARD
            damage = max(0, self._prev_alien_total_hp - alien_hp_now)
            self._battle_reward_pending += damage * self.ALIEN_HIT_REWARD
        self._prev_alien_total_hp = alien_hp_now

        if event == "pick_up":
            self._battle_reward_pending += self.PICKUP_REWARD

        # Exploration + alien detection on any event that carries the tile grid
        if "map" in msg:
            self._process_battle_map(msg)

    def _next_geostate(self, timeout: float = 15.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._client.recv_one(timeout=max(0.05, deadline - time.time()))
            if msg is None:
                break
            if msg.get("type") == "state":
                return msg
            if msg.get("type") == "battle":
                self._process_battle_message(msg)
        return None

    def _auto_confirm_base_name(self, state: Dict) -> Dict:
        prev_hint = None
        same_hint_count = 0
        max_iter = 30
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            hint = state.get("game", {}).get("screen_hint", "")
            if hint == prev_hint:
                same_hint_count += 1
                if same_hint_count >= 4:
                    break
            else:
                same_hint_count = 0
                prev_hint = hint

            if hint == "base_name":
                self._client.send(action="key", key="a")
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
                self._client.send(action="key", key="enter")
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "place_lift":
                # Click center of top-left grid cell (0,0).
                # BaseView is at screen (0, 8); cells are 32 px → center = (16, 24).
                self._dismiss_next_basescape = True  # the basescape that follows is new-base setup
                self._client.send(action="click", x=16, y=24, button=1)
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "basescape":
                if self._dismiss_next_basescape:
                    # Dismiss the transient basescape from new-base setup flow
                    self._dismiss_next_basescape = False
                    self._client.send(action="dismiss")
                    if self._headless:
                        self._client.send(action="step")
                    state = self._next_geostate(timeout=0.5) or state
                else:
                    break  # agent navigated here deliberately — let it interact
            elif hint == "ufopaedia":
                self._client.send(action="dismiss")
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "monthly_report":
                # MonthlyReportState and CommendationState — OK at (160, 186).
                # Game-over path needs two clicks: first reveals failure text,
                # second pushes CutsceneState → StatisticsState (hint changes to
                # "statistics" once the engine is rebuilt with pushState fix).
                self._client.send(action="click", x=160, y=186, button=1)
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "statistics":
                # StatisticsState (defeat/win end-game summary).
                # _btnOk TextButton(50,12, 135,180) → centre (160,186).
                # No Escape binding — must use a click, not dismiss.
                # After OK on a defeat/win ending the engine wipes the save and
                # goes to GoToMainMenuState; broadcastMenuState() then fires.
                self._client.send(action="click", x=160, y=186, button=1)
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "load_game":
                self._pending_load_penalty += self.LOAD_PENALTY * self.reward_scale_penalty
                self._client.send(action="dismiss")
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "pause_menu":
                self._pending_options_penalty += self.OPTIONS_MENU_PENALTY * self.reward_scale_penalty
                self._client.send(action="dismiss")
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "facility_error":
                # First dismiss closes ErrorMessageState (Escape → btnOkClick).
                # Old code used click(160, 140) which missed the OK button at y=154–172.
                # Second dismiss closes PlaceFacilityState (Escape → btnCancelClick) so
                # the agent doesn't immediately retry the same invalid grid cell.
                # No broadcast arrives while in basescape; break immediately instead of
                # timing out 4× for ~8 s with the dialog still open.
                self._facility_error_pending = True
                self._client.send(action="dismiss")
                self._client.send(action="dismiss")
                break
            elif hint == "briefing":
                # Pre-mission briefing (BriefingState) — click OK to proceed.
                # Escape is bound to btnOkClick so dismiss would also work, but
                # an explicit click is cleaner and avoids the anti-stuck delay.
                self._client.send(action="click", x=160, y=173, button=1)
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "debriefing":
                # Post-mission debrief (DebriefingState) — click OK to continue.
                # _btnOk TextButton(40,12, 16,180) → centre (36,186); Escape also bound.
                self._client.send(action="click", x=36, y=186, button=1)
                if self._headless:
                    self._client.send(action="step")
                state = self._next_geostate(timeout=0.5) or state
            elif hint == "soldiers":
                self._client.send(action="dismiss")
                break
            elif hint == "soldier_info":
                self._client.send(action="dismiss")
                break
            elif hint == "soldier_armor":
                self._client.send(action="dismiss")
                break
            else:
                break
        return state

    def _execute_action(self, action: int) -> Dict:
        name, x, y, button = ACTIONS[action]

        if name == "wait":
            if self._headless:
                # Fast-forward up to 60 game-minutes per agent step.  Each iteration
                # sends one step (= 1 game-minute at default speed) and reads the
                # resulting broadcast.  Break early when an event interrupts the
                # geoscape (popup, intercept screen, monthly report, etc.) so the
                # caller's _auto_confirm_base_name can handle it immediately.
                _MAX_WAIT_STEPS = 60
                state = self._last_state
                for _ in range(_MAX_WAIT_STEPS):
                    self._client.send(action="step")
                    new_state = self._next_geostate(timeout=0.5)
                    if new_state is not None:
                        state = new_state
                        if new_state.get("game", {}).get("screen_hint", "geoscape") != "geoscape":
                            break  # event fired — return for auto-confirm handling
                return state
            else:
                # Visible: poll for next broadcast from the running timer
                return self._next_geostate(timeout=0.5) or self._last_state

        if name == "menu_new_game":
            # Only acts when screen_type is main_menu; otherwise behaves like wait.
            at_menu = (self._last_state or {}).get("game", {}).get("screen_hint") == "main_menu"
            if at_menu:
                self._client.send(action="new_game")
                # cmdNewGame() broadcasts "new_game_ready" immediately.
                # In headless mode also send "step" as a fallback trigger.
                if self._headless:
                    self._client.send(action="step")
                return self._next_geostate(timeout=10.0) or self._last_state
            else:
                # Not at menu: no-op, just wait for next geoscape tick
                if self._headless:
                    self._client.send(action="step")
                    return self._next_geostate(timeout=0.5) or self._last_state
                else:
                    return self._next_geostate(timeout=0.5) or self._last_state
        elif name == "wait_2":
            # No-op — formerly menu_load_game; loading permanently disabled.
            if self._headless:
                self._client.send(action="step")
                return self._next_geostate(timeout=0.5) or self._last_state
            else:
                return self._next_geostate(timeout=0.5) or self._last_state
        elif name == "battle_pickup":
            self._client.send(action="pick_up")
            # Battlescape think() processes commands without a "step" trigger.
            # In geoscape this command is silently ignored; we still wait for a
            # broadcast so the caller gets an updated state.
            return self._next_geostate(timeout=5.0) or self._last_state
        elif name == "dismiss":
            self._client.send(action="dismiss")
        elif name == "key_a":
            self._client.send(action="key", key="a")
        elif name == "key_enter":
            self._client.send(action="key", key="enter")
        else:
            # Mouse click (named button or random position)
            if x == _RANDOM_CLICK_SENTINEL:
                x = int(self.np_random.integers(0, SCREEN_W))
                y = int(self.np_random.integers(0, SCREEN_H))
            self._client.send(action="click", x=x, y=y, button=button)

        if self._headless:
            # In headless mode only "step" triggers a state broadcast.
            # Sending one here makes the engine advance time and emit a fresh
            # "tick" after processing the action above, avoiding the timeout
            # that would otherwise fire for every click/dismiss/key action.
            self._client.send(action="step")
            result = self._next_geostate(timeout=1.5)
            if result is None:
                # No broadcast within 1.5 s — GeoscapeState wasn't on top.
                # Only unwind with dismisses if the last known screen is in the
                # deep basescape/crafts/research stack where dismiss-unwinding
                # is safe.  For geoscape-adjacent screens the engine may just be
                # slow; return _last_state and let _auto_confirm_base_name handle
                # any transient screen on the next call.
                last_hint = (self._last_state or {}).get("game", {}).get("screen_hint", "geoscape")
                if last_hint in _DISMISS_STACK:
                    for _ in range(4):
                        self._client.send(action="dismiss")
                    self._client.send(action="step")
                    result = self._next_geostate(timeout=5.0)
            return result or self._last_state
        else:
            timeout = 0.5 if name in ("dismiss", "key_a", "key_enter") else 0.3
            return self._next_geostate(timeout=timeout) or self._last_state

    # ── engine lifecycle ───────────────────────────────────────────────────────

    def _launch_with_retry(self, max_attempts: int = 3) -> None:
        for attempt in range(max_attempts):
            try:
                self._launch_engine()
                return
            except (RuntimeError, ConnectionError, OSError) as exc:
                print(f"[env {self._rank}] engine launch attempt {attempt + 1}/{max_attempts} failed: {exc}")
                self._teardown()
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2.0)

    def _launch_engine(self) -> None:
        if not self._initial_launch_done and self._rank > 0:
            # Stagger initial engine launches: one every 5 s per rank.
            # Only applies to the first launch; restarts after crashes are immediate.
            time.sleep(self._rank * 5.0)
        self._initial_launch_done = True

        # Suppress Windows crash dialogs (access violations, GPF) so that engine
        # crashes exit the process cleanly rather than blocking on a modal dialog
        # waiting for user input.  Child processes inherit this error mode.
        # SEM_FAILCRITICALERRORS(0x1)|SEM_NOGPFAULTERRORBOX(0x2)|SEM_NOOPENFILEERRORBOX(0x8000)
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8003)

        socket_path = os.path.join(_TEMP, self._socket_name)

        # Remove stale socket file from a previous run so the engine can bind.
        # On Windows, AF_UNIX socket files are not deleted when the server exits.
        try:
            os.remove(socket_path)
        except OSError:
            pass

        env = os.environ.copy()
        # XCOM_BRIDGE_AUTOLOAD (if set) skips the main menu and loads a specific
        # save file on startup and on each new_game reset — takes priority over
        # AUTOSTART.  Headless-only envs without autoload use AUTOSTART to start
        # a fresh Beginner game.  Visible envs without autoload show the main menu.
        if self._autoload_save:
            env["XCOM_BRIDGE_AUTOLOAD"] = self._autoload_save
        elif self._headless:
            env["XCOM_BRIDGE_AUTOSTART"] = "1"
        env["XCOM_BRIDGE_SOCKET"]    = self._socket_name
        if self._speed_ms > 0:
            env["XCOM_GAME_SPEED"] = str(self._speed_ms)

        args = [ENGINE_EXE, "-playIntro", "false"]
        if self._headless:
            args.append("--headless")

        self._proc = subprocess.Popen(
            args, cwd=ENGINE_CWD, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        for _ in range(100):
            time.sleep(0.1)
            try:
                self._client = _BridgeClient(socket_path)
                break
            except (OSError, ConnectionRefusedError):
                pass
        else:
            raise RuntimeError(f"bridge socket {socket_path!r} did not open within 10s")

        with open(os.path.join(_LOG_DIR, f"engine_{self._rank}.pid"), "w") as _f:
            _f.write(str(self._proc.pid))

        first = None
        if self._headless:
            for _ in range(5):
                self._client.send(action="step")
                first = self._next_geostate(timeout=15.0)
                if first is not None:
                    break
        else:
            # Visible mode: engine loads (~15s) then shows the main menu.
            # autoStart() broadcasts a "main_menu" state every ~100ms once loading
            # is done.  Just wait — the agent will navigate from here via
            # menu_new_game.  Allow up to 120s for slow machines.
            first = self._next_geostate(timeout=120.0)

        if first is None:
            raise RuntimeError("no initial state received from engine")

        self._last_state       = first
        self._total_score_prev = first.get("game", {}).get("total_score", 0)
        self._prev_craft_out   = False
        self._prev_ufo_seen    = False

    def _new_game(self) -> bool:
        try:
            if self._new_game_in_flight:
                # new_game already dispatched at episode end; new_game_ready may
                # already be sitting in the socket buffer.  Skip drain (it would
                # discard the response) and skip the duplicate send.
                self._new_game_in_flight = False
            else:
                # Discard stale tick events accumulated from the previous episode.
                # Without this, thousands of buffered ticks swamp the 5-iteration
                # loop and new_game_ready is never seen → _teardown() fires → window
                # closes and the 15s engine reload freeze begins.
                self._client.drain_buf()
                self._client.send(action="new_game")
            # Time-based loop: new_game_ready arrives within one geoscape frame
            # (~10ms) in visible mode or immediately via cmdNewGame() in headless.
            deadline = time.time() + 15.0
            while time.time() < deadline:
                state = self._next_geostate(timeout=min(5.0, deadline - time.time()))
                if state is None:
                    break
                if state.get("event") == "new_game_ready":
                    self._last_state       = state
                    self._total_score_prev = state.get("game", {}).get("total_score", 0)
                    self._prev_craft_out   = False
                    self._prev_ufo_seen    = False
                    return True
        except Exception:
            pass
        return False

    def _teardown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None
            time.sleep(0.3)
        # Delete socket file; Windows does not auto-remove it when the engine exits.
        socket_path = os.path.join(_TEMP, self._socket_name)
        try:
            os.remove(socket_path)
        except OSError:
            pass

    def _log_episode(self, state: Dict, ending: int, terminated: bool, truncated: bool) -> None:
        game = state.get("game", {})
        ep_tally = {
            k: {
                "n":     self._reward_tally[k]["n"]     - self._ep_tally_start.get(k, {"n": 0})["n"],
                "total": round(self._reward_tally[k]["total"] - self._ep_tally_start.get(k, {"total": 0.0})["total"], 4),
            }
            for k in self._reward_tally
        }
        self.last_ep_info = {
            "reward":      round(self._ep_reward, 3),
            "month":       game.get("month", 1),
            "total_score": game.get("total_score", 0),
            "ending":      ending,
            "steps":       self._step_count,
            "tally":       ep_tally,
        }
        entry = {
            "t":           time.time(),
            "env":         self._rank,
            "ep":          self._episode_num,
            "steps":       self._step_count,
            "reward":      round(self._ep_reward, 3),
            "total_score": game.get("total_score", 0),
            "ending":      ending,
            "truncated":   truncated,
            "month":       game.get("month", 1),
            "day":         game.get("day",   1),
            "tally":       ep_tally,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── observation builder ────────────────────────────────────────────────────

    def _build_obs(self, state: Dict) -> Dict[str, Any]:
        g      = state.get("game", {})
        bases  = state.get("bases", [])
        craft  = self._first_craft(state) or {}
        fm     = max(1, craft.get("fuel_max",   1))
        dm     = max(1, craft.get("damage_max", 1))
        cs     = _CRAFT_STATUS_INT.get(craft.get("status", "STR_READY"), 0)

        ufos        = state.get("ufos", [])
        ufo_det     = int(any(u.get("detected") for u in ufos))
        mission_det = int(bool(state.get("mission_sites")))
        num_sol     = sum(len(b.get("soldiers", [])) for b in bases)

        screen_hint = g.get("screen_hint", "geoscape")
        screen_int  = SCREEN_TYPES.get(screen_hint, SCREEN_TYPES["other"])

        base = bases[0] if bases else {}
        grid = np.zeros(36, dtype=np.int8)
        for fac in base.get("facilities", []):
            fx    = fac.get("x", 0)
            fy    = fac.get("y", 0)
            fsize = fac.get("size", 1)
            code  = np.int8(2 if fac.get("build_days", 0) == 0 else 1)
            for dr in range(fsize):
                for dc in range(fsize):
                    idx = (fy + dr) * 6 + (fx + dc)
                    if 0 <= idx < 36:
                        grid[idx] = code

        return {
            "funds":            np.array([int(np.clip(g.get("funds", 0), 0, 100_000_000))], dtype=np.int32),
            "score":            np.array([int(np.clip(g.get("score", 0), -10_000, 10_000))], dtype=np.int32),
            "total_score":      np.array([int(np.clip(g.get("total_score", 0), -100_000, 100_000))], dtype=np.int32),
            "month":            np.array([int(np.clip(g.get("month", 1), 1, 12))], dtype=np.int32),
            "day":              np.array([int(np.clip(g.get("day",   1), 1, 31))], dtype=np.int32),
            "hour":             np.array([int(np.clip(g.get("hour",  0), 0, 23))], dtype=np.int32),
            "craft_status":     np.int64(cs),
            "craft_fuel_pct":   np.array([int(np.clip(100 * craft.get("fuel", 0) // fm, 0, 100))], dtype=np.int32),
            "craft_damage_pct": np.array([int(np.clip(100 * craft.get("damage", 0) // dm, 0, 100))], dtype=np.int32),
            "ufo_detected":     np.int64(ufo_det),
            "mission_site":     np.int64(mission_det),
            "popup_depth":      np.array([int(np.clip(g.get("popup_depth", 0), 0, 10))], dtype=np.int32),
            "screen_type":      np.int64(screen_int),
            "ending":           np.int64(int(g.get("ending", END_NONE))),
            "num_soldiers":     np.array([int(np.clip(num_sol, 0, 50))], dtype=np.int32),
            "base_grid":        grid,
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _first_craft(state: Dict) -> Optional[Dict]:
        bases = state.get("bases", [])
        if not bases:
            return None
        crafts = bases[0].get("crafts", [])
        return crafts[0] if crafts else None

    @staticmethod
    def _any_craft_out(state: Dict) -> bool:
        bases = state.get("bases", [])
        for b in bases:
            for c in b.get("crafts", []):
                if c.get("status") == "STR_OUT":
                    return True
        return False

    @staticmethod
    def _count_research(state: Dict) -> int:
        return sum(len(b.get("research", [])) for b in state.get("bases", []))

    @staticmethod
    def _count_manufacture(state: Dict) -> int:
        return sum(len(b.get("manufacture", [])) for b in state.get("bases", []))

    @staticmethod
    def _count_staff(state: Dict) -> tuple:
        sci = sum(b.get("scientists", 0) for b in state.get("bases", []))
        eng = sum(b.get("engineers", 0) for b in state.get("bases", []))
        return sci, eng

    @staticmethod
    def _count_facilities_building(state: Dict) -> int:
        return sum(
            1 for b in state.get("bases", [])
            for f in b.get("facilities", [])
            if f.get("build_days", 0) > 0
        )

    @staticmethod
    def _count_store_total(state: Dict) -> int:
        return sum(sum(b.get("stores", {}).values()) for b in state.get("bases", []))

    @staticmethod
    def _to_game_minutes(state: Dict) -> int:
        g      = state.get("game", {})
        month  = g.get("month",  1)
        day    = g.get("day",    1)
        hour   = g.get("hour",   0)
        minute = g.get("minute", 0)
        return (month - 1) * 30 * 24 * 60 + (day - 1) * 24 * 60 + hour * 60 + minute

    @staticmethod
    def _get_soldier_ranks(state: Dict) -> Dict[str, str]:
        """Map soldier name → rank string for soldiers in the first base."""
        bases = state.get("bases", [])
        if not bases:
            return {}
        return {s.get("name", ""): s.get("rank", "") for s in bases[0].get("soldiers", [])}

    @staticmethod
    def _count_facilities_total(state: Dict) -> int:
        """Total facility count (completed + under construction) across all bases."""
        return sum(len(b.get("facilities", [])) for b in state.get("bases", []))

    @staticmethod
    def _first_base_occupied_squares(state: Dict) -> int:
        """Grid squares occupied in the first base; size×size squares per facility."""
        bases = state.get("bases", [])
        if not bases:
            return 0
        return sum(f.get("size", 1) ** 2 for f in bases[0].get("facilities", []))

    @staticmethod
    def _count_alien_bases_found(state: Dict) -> int:
        """Count alien bases that have been discovered by X-COM."""
        return sum(1 for ab in state.get("alien_bases", []) if ab.get("discovered", False))

    def _tally(self, cat: str, amount: float, n: int = 1) -> None:
        t = self._reward_tally.get(cat)
        if t is not None and amount != 0.0:
            t["n"]     += n
            t["total"] += amount

    def _write_live_state(self, step_reward: float, game: Dict) -> None:
        state = self._last_state or {}
        craft = self._first_craft(state) or {}
        fm    = max(1, craft.get("fuel_max",   1))
        dm    = max(1, craft.get("damage_max", 1))

        bases = state.get("bases", [])
        base  = bases[0] if bases else {}
        grid  = [0] * 36
        for fac in base.get("facilities", []):
            fx    = fac.get("x", 0)
            fy    = fac.get("y", 0)
            fsize = fac.get("size", 1)
            code  = 2 if fac.get("build_days", 0) == 0 else 1
            for dr in range(fsize):
                for dc in range(fsize):
                    idx = (fy + dr) * 6 + (fx + dc)
                    if 0 <= idx < 36:
                        grid[idx] = code

        data = {
            "rank":        self._rank,
            "ep":          self._episode_num,
            "ep_step":     self._step_count,
            "ep_reward":   round(self._ep_reward, 3),
            "step_reward": round(step_reward, 4),
            "screen":      game.get("screen_hint", "geoscape"),
            "total_score": game.get("total_score", 0),
            "month":       game.get("month", 1),
            "day":         game.get("day", 1),
            "headless":    self._headless,
            "funds":       game.get("funds", 0),
            "craft_status":      craft.get("status", "STR_READY"),
            "craft_fuel_pct":    int(100 * craft.get("fuel", 0) // fm),
            "craft_damage_pct":  int(100 * craft.get("damage", 0) // dm),
            "base_grid":         grid,
            "action_window": list(self._action_window),
            "reward_window": list(self._reward_window),
            "reward_tally":  {k: {"n": v["n"], "total": round(v["total"], 3)}
                               for k, v in self._reward_tally.items()},
            "reward_scales": {
                "nav":       round(self.reward_scale_nav, 4),
                "milestone": round(self.reward_scale_milestone, 4),
                "penalty":   round(self.reward_scale_penalty, 4),
            },
            "action_hist":  dict(self._action_hist),
            "screen_hist":  dict(self._screen_hist),
            "stuck_total":  self._stuck_total,
            "t":             time.time(),
        }
        path = os.path.join(_LOG_DIR, f"live_env_{self._rank}.json")
        tmp  = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            pass
