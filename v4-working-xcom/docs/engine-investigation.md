# OpenXcom Engine Investigation

Source tree: `engine/openxcom/src`  
Engine version: 1.0.630130c5c (git HEAD at clone time)  
Investigated: 2026-06-21

---

## 1. Main Game Loop and State Machines

### 1.1 Entry Point and Loop Structure

**Entry point:** `src/main.cpp:88–126`

```cpp
game = new Game(title.str());   // constructs SDL window, screen, cursor
game->setState(new StartState); // push first state
game->run();                    // enters loop
```

**Main loop:** `src/Engine/Game.cpp`, `Game::run()`, lines 143–344.

The loop runs until `_quit` is set and has three phases each iteration:

| Phase | Lines | What happens |
|-------|-------|--------------|
| State init | 159–177 | If a new state was pushed, call `_states.back()->init()` |
| Event dispatch | 179–289 | `SDL_PollEvent()` → forward to `_states.back()->handle(action)` |
| Tick + render | 292–329 | `_states.back()->think()`, then `_screen->flip()` (blit + SDL_Flip) |

Frame timing is wall-clock: `SDL_GetTicks()` compared against `1000 / Options::FPS`. When the app loses focus an `ApplicationState` enum (`RUNNING / SLOWED / PAUSED`) throttles `SDL_Delay()` to 100 ms per iteration so the app idles cheaply.

### 1.2 State Stack

**Class:** `Game` in `src/Engine/Game.h` / `Game.cpp`  
**Storage:** `std::list<State*> _states` (Game.h)  
**Deletion queue:** `std::list<State*> _deleted` — states popped mid-loop are deferred here and freed at the top of the next iteration.

| Method | Lines | Behaviour |
|--------|-------|-----------|
| `setState(State*)` | 437–445 | Empties stack, pushes single state, sets `_init=false` |
| `pushState(State*)` | 452–456 | Appends to back of stack, sets `_init=false` |
| `popState()` | 464–469 | Moves back state to `_deleted`; next iteration calls `init()` on new back |

**Base State class:** `src/Engine/State.h`

```cpp
virtual void init();          // called once when state becomes active
virtual void handle(Action*); // receives SDL events forwarded by Game
virtual void think();         // called every frame for logic
virtual void blit();          // called every frame for rendering
```

Every concrete state holds `std::vector<Surface*> _surfaces`. `centerAllSurfaces()` offsets them by `screen->getDX()/getDY()`, which is `(baseWidth-320)/2` — always zero for the standard 320×200 base resolution, so source coordinates are raw game pixels.

### 1.3 Geoscape State Machine

**Files:** `src/Geoscape/GeoscapeState.h` / `GeoscapeState.cpp`

`GeoscapeState` is the top-level Geoscape controller. It is pushed via `setState(new GeoscapeState)` after the intro / main-menu flow completes.

**`think()` — lines 550–583:**
```
if no dogfight active and no popups pending:
    _gameTimer->think()          → fires timeAdvance() at configured speed
_zoomInEffectTimer->think()
_zoomOutEffectTimer->think()
_dogfightStartTimer->think()
flush _popups: pushState(_popups.front())
```

**`timeAdvance()` — lines 631–685:**  
Each call advances the `GameTime` by `TIME_5SEC`. `GameTime::advance()` returns a `TimeTrigger` bitmask. The method then fires one handler per triggered threshold in ascending order:

```
TIME_5SEC  → check UFO waypoints, dogfight updates
TIME_10MIN → update alien missions, spawn UFOs
TIME_30MIN → update Geoscape activity
TIME_1HOUR → craft movement, transfers
TIME_1DAY  → soldier recovery, research progress
TIME_1MONTH → funding report, score calculation
```

Time speed is controlled by UI buttons that set the `_gameTimer` interval (5 s / 1 min / 5 min / 30 min / 1 hr / 1 day of game time per real second).

**State transitions pushed from Geoscape:**
- `pushState(new InterceptState(_globe))` — UFO interception
- `pushState(new BasescapeState(...))` — base management
- `pushState(new DogfightState(...))` — air combat
- Tactical battles triggered by `BattlescapeGenerator` → `setState(new BriefingState)` → eventually `setState(new BattlescapeState)`

### 1.4 Battlescape State Machine

**Files:** `src/Battlescape/BattlescapeGame.h` / `BattlescapeGame.cpp`  
`src/Battlescape/BattlescapeState.h` / `BattlescapeState.cpp`

`BattlescapeState` owns the SDL rendering for the battle screen. `BattlescapeGame` contains all game-logic: turn management, action dispatch, AI.

**`BattlescapeGame::think()` — lines 100–150:**
```
if _states list is empty:
    if it is the player's turn:
        selectNextPlayerUnit()
    else (alien turn):
        handleAI(currentUnit)
else:
    _states.front()->think()   // advance current BattleState animation
    if _states.front()->isFinished():
        popState()             // move to next queued action
```

**`BattleAction` struct** (`BattlescapeGame.h`, lines 43–63):

```cpp
struct BattleAction {
    BattleActionType type;    // BA_WALK, BA_SNAPSHOT, BA_AUTOSHOT, …
    BattleUnit *actor;
    BattleItem *weapon;
    Position   target;
    int        TU;            // time-unit cost
    bool       targeting;
    std::list<Position> waypoints;
    int        value;
    std::string result;
    // … strafe, run, diff, autoShotCounter, cameraPosition, …
};
```

**`BattleActionType` enum** (line 41):
`BA_NONE, BA_TURN, BA_WALK, BA_PRIME, BA_THROW, BA_AUTOSHOT, BA_SNAPSHOT, BA_AIMEDSHOT, BA_HIT, BA_USE, BA_LAUNCH, BA_MINDCONTROL, BA_PANIC, BA_RETHINK`

**BattleState subclasses** (each pushed to `BattlescapeGame::_states`):
- `UnitWalkBState` — pathfinding + movement animation
- `UnitTurnBState` — facing change
- `ProjectileFlyBState` — bullet / grenade arc
- `MeleeAttackBState` — close-combat animation
- `PsiAttackBState` — psi ability resolution
- `ExplosionBState` — blast radius + tile damage
- `UnitFallBState` / `UnitDieBState` / `UnitPanicBState` — incapacitation effects

**AI system** (`src/Battlescape/AIModule.h` / `AIModule.cpp`):

```cpp
enum AIMode { AI_PATROL, AI_AMBUSH, AI_COMBAT, AI_ESCAPE };
void think(BattleAction *action);   // fills action struct each alien turn
```

Internal methods: `setupPatrol()`, `setupAmbush()`, `setupAttack()`, `setupEscape()`, `selectFireMethod()`, `grenadeAction()`, `projectileAction()`, `meleeAction()`, `psiAction()`. The AI module is a pure deterministic decision tree; there is no learned or probabilistic component.

---

## 2. Key Class Definitions

All files are under `engine/openxcom/src/Savegame/`.

### 2.1 `SavedGame` (`SavedGame.h`)

The root object for the entire campaign. One instance lives in `Game::_save`.

**Selected fields (private):**

```cpp
std::string              _name
GameDifficulty           _difficulty          // 0=Beginner … 4=Superhuman
GameEnding               _end
bool                     _ironman
GameTime                *_time
std::vector<int>         _researchScores
std::vector<int64_t>     _funds, _maintenance, _incomes, _expenditures
double                   _globeLon, _globeLat
int                      _globeZoom
std::map<std::string,int> _ids               // per-type ID counters
std::vector<Country*>    _countries
std::vector<Region*>     _regions
std::vector<Base*>       _bases
std::vector<Ufo*>        _ufos
std::vector<Waypoint*>   _waypoints
std::vector<MissionSite*> _missionSites
std::vector<AlienBase*>  _alienBases
AlienStrategy           *_alienStrategy
SavedBattleGame         *_battleGame         // null when not in battle
std::vector<const RuleResearch*> _discovered
std::vector<AlienMission*>       _activeMissions
bool                     _debug, _warned
int                      _monthsPassed
std::vector<Soldier*>    _deadSoldiers
size_t                   _selectedBase
```

**Key methods:**
- `void load(const std::string &filename, Mod *mod)` — deserializes from YAML file
- `void save(const std::string &filename) const` — serializes to YAML file
- `std::vector<Base*>* getBases()` — all player bases
- `SavedBattleGame* getSavedBattle()` — active battle (null on Geoscape)
- `bool isResearched(const std::string &id, bool debug=true) const`

### 2.2 `Base` (`Base.h`)

Inherits from `Target`. Represents one player base on the globe.

**Selected fields (private):**

```cpp
static const int          BASE_SIZE = 6        // 6×6 facility grid
const Mod                *_mod
std::vector<BaseFacility*> _facilities
std::vector<Soldier*>     _soldiers
std::vector<Craft*>       _crafts
std::vector<Transfer*>    _transfers
ItemContainer            *_items
int                       _scientists, _engineers
std::vector<ResearchProject*> _research
std::vector<Production*>  _productions
bool                      _inBattlescape
bool                      _retaliationTarget
std::vector<Vehicle*>     _vehicles
std::vector<BaseFacility*> _defenses
```

**Key methods:**
- `void load(const YAML::Node&, SavedGame*, bool newGame, bool newBattle=false)`
- `YAML::Node save() const`
- `std::vector<Soldier*>* getSoldiers()`
- `std::vector<Craft*>* getCrafts()`
- `int getMonthlyMaintenace() const`

### 2.3 `Soldier` (`Soldier.h`)

One XCOM squad member.

**Selected fields (private):**

```cpp
std::string              _name
int                      _id
RuleSoldier             *_rules
UnitStats                _initialStats, _currentStats
SoldierRank              _rank              // RANK_ROOKIE … RANK_COMMANDER
Craft                   *_craft             // null if at base
SoldierGender            _gender
SoldierLook              _look
int                      _missions, _kills
int                      _recovery          // days hospitalized
bool                     _psiTraining
Armor                   *_armor
std::vector<EquipmentLayoutItem*> _equipmentLayout
SoldierDeath            *_death             // null if alive
SoldierDiary            *_diary
```

**`UnitStats` struct** (referenced by both Soldier and BattleUnit):
`tu, stamina, health, bravery, reactions, firing, throwing, strength, psiStrength, psiSkill, melee`

**Key methods:**
- `void load(const YAML::Node&, const Mod*, SavedGame*)`
- `YAML::Node save() const`
- `SoldierRank getRank() const`
- `void promoteRank()`
- `Armor* getArmor() const`

### 2.4 `Craft` (`Craft.h`)

Inherits from `MovingTarget`. Skyranger, Interceptor, etc.

**Selected fields (private):**

```cpp
RuleCraft               *_rules
Base                    *_base
int                      _fuel, _damage
int                      _interceptionOrder, _takeoff
std::vector<CraftWeapon*> _weapons
ItemContainer           *_items
std::vector<Vehicle*>    _vehicles
std::string              _status            // "STR_READY", "STR_OUT", …
bool                     _lowFuel, _mission, _inBattlescape, _inDogfight
double                   _speedMaxRadian
```

**Key methods:**
- `void load(const YAML::Node&, const Mod*, SavedGame*)`
- `YAML::Node save() const`
- `int getFuel() const`
- `int getDamage() const`
- `std::string getStatus() const`

### 2.5 `SavedBattleGame` (`SavedBattleGame.h`)

The complete state of one tactical mission.

**Selected fields (private):**

```cpp
BattlescapeState        *_battleState       // back-pointer to UI layer
int                      _mapsize_x, _mapsize_y, _mapsize_z
std::vector<MapDataSet*> _mapDataSets
Tile                   **_tiles             // flat array [x + y*sx + z*sx*sy]
BattleUnit              *_selectedUnit, *_lastSelectedUnit
std::vector<Node*>       _nodes             // pathfinding graph
std::vector<BattleUnit*> _units
std::vector<BattleItem*> _items, _deleted
Pathfinding             *_pathfinding
TileEngine              *_tileEngine
std::string              _missionType
int                      _globalShade
UnitFaction              _side              // FACTION_PLAYER or FACTION_HOSTILE
int                      _turn
bool                     _debugMode, _aborted
int                      _objectiveType, _objectivesDestroyed, _objectivesNeeded
std::vector<BattleUnit*> _exposedUnits
BattleActionType         _tuReserved
bool                     _kneelReserved
int                      _depth, _ambience
double                   _ambientVolume
int                      _turnLimit
ChronoTrigger            _chronoTrigger
```

**Key methods:**
- `void load(const YAML::Node&, Mod*, SavedGame*)`
- `YAML::Node save() const`
- `std::vector<BattleUnit*>* getUnits()`
- `Tile* getTile(Position pos) const` — inline, `_tiles[pos.x + pos.y*sx + pos.z*sx*sy]`
- `BattleUnit* getSelectedUnit() const`

### 2.6 `BattleUnit` (`BattleUnit.h`)

One unit on the tactical map (player soldier, alien, or civilian).

**Selected fields (private) — most relevant for RL:**

```cpp
int              _id
UnitFaction      _faction                   // FACTION_PLAYER / HOSTILE / NEUTRAL
UnitFaction      _originalFaction
Position         _pos                       // {x, y, z} tile coords
int              _direction, _directionTurret  // 0–7 facing
UnitStatus       _status                    // STATUS_STANDING, WALKING, DEAD, …
int              _tu                        // time units remaining this turn
int              _energy                    // stamina
int              _health                    // current HP
int              _morale                    // 0–100
int              _stunlevel
bool             _kneeled, _floating
int              _currentArmor[5]           // FRONT/LEFT/RIGHT/REAR/UNDER
int              _fatalWounds[6]            // per body part
int              _fire
std::vector<BattleUnit*> _visibleUnits
std::vector<Tile*>       _visibleTiles
AIModule        *_currentAIState            // null for player units
UnitStats        _stats                     // tu, stamina, health, reactions, …
Soldier         *_geoscapeSoldier           // non-null for XCOM soldiers
std::string      _type, _rank, _race, _name
Armor           *_armor
int              _expBravery, _expReactions, _expFiring   // XP accumulators
int              _kills
```

**Key methods:**
- `void load(const YAML::Node&)`
- `YAML::Node save() const`
- `UnitStatus getStatus() const`
- `Position getPosition() const`
- `int getTimeUnits() const`
- `int getHealth() const`
- `int getMorale() const`

---

## 3. Headless / No-Render Code Path

**Answer: No headless mode exists. Window creation is unconditional.**

### Evidence

Window creation lives in `src/Engine/Screen.cpp`, `resetDisplay()` method, lines 309–475:

```cpp
// Line 347 — always executed, no guard:
_screen = SDL_SetVideoMode(width, height, _bpp, _flags);
```

This uses SDL 1.2's `SDL_SetVideoMode` (not SDL2's `SDL_CreateWindow`, since the engine targets SDL 1.2). The call is inside `resetDisplay()`, which is called from:
- `Screen::Screen()` constructor (line 116 of Screen.cpp)
- `Screen` is constructed in `Game::Game()` (line 92 of Game.cpp)

SDL is always initialized with video: `SDL_Init(SDL_INIT_VIDEO)` at Game.cpp:60. There is no `#ifdef HEADLESS` or equivalent guard anywhere in the engine.

The only optional subsystem is audio: `SDL_InitSubSystem(SDL_INIT_AUDIO)` at line 67 fails gracefully and sets `Options::mute = true`.

### What a headless mode would require

1. Conditional `SDL_Init` — skip `SDL_INIT_VIDEO` or use `SDL_INIT_NOPARACHUTE`
2. Null-surface `Screen` — either skip construction or back the surface with an offscreen buffer
3. Guard `SDL_Flip()` calls in `Screen::flip()` (lines 188–216)
4. Suppress the cursor and audio subsystem init in `Game::Game()`

This is straightforward to add as an `Options::headless` flag but does not exist today.

---

## 4. Embedded Scripting Layer

**Answer: No. OpenXcom has no embedded scripting engine beyond static YAML rulesets.**

### Search Results

Searched entire `src/` tree for scripting indicators:

| Search term | Files found | Result |
|-------------|-------------|--------|
| `lua`, `lua.h`, `lua.hpp` | 0 | No Lua |
| `python`, `Python.h` | 0 | No Python |
| `AngelScript`, `angelscript` | 0 | No AngelScript |
| `ChakraCore`, `V8`, `duktape`, `sol2` | 0 | No JS/other engine |
| `eval(`, `interpret(`, `execute(` | 0 | No runtime evaluation |
| `Script`, `callback`, `hook` in headers | 21 matches | All false positives (e.g. `UfopaediaStartState.h`, `ArticleState.h`) — no callback machinery |

### What the ruleset system actually is

All game data lives in `standard/xcom1/*.rul` YAML files loaded at startup by `Mod::loadAll()` (called at `Game.cpp:516`). The `Mod` class (`src/Mod/Mod.h`) holds `std::map<std::string, Rule*>` for every entity type:

```cpp
std::map<std::string, RuleCraft*>        _crafts
std::map<std::string, RuleItem*>         _items
std::map<std::string, RuleUfo*>          _ufos
std::map<std::string, RuleTerrain*>      _terrains
std::map<std::string, AlienRace*>        _alienRaces
std::map<std::string, AlienDeployment*>  _alienDeployments
std::map<std::string, RuleAlienMission*> _alienMissions
// … ~20 more rule maps
```

These are **pure static data** — no executable logic. The closest thing to "procedural" behaviour is:
- `AIModule::think()` — deterministic decision tree in C++ (`src/Battlescape/AIModule.cpp`)
- `AlienMission::think()` — fixed FSM for UFO spawn scheduling
- `TileEngine` — hardcoded FOV, LOS, explosion algorithms

None of these are scriptable or hookable from outside C++.

---

## 5. Save-Game YAML Serialization

### 5.1 File Format

Save files are **human-readable YAML with two documents** separated by `---`:

```
# Document 0 — brief metadata (shown in load-game UI)
name: "My Campaign"
version: "1.0"
build: "630130c5c"
time: { year: 1999, month: 3, weekDay: 2, day: 15, hour: 14, minute: 30 }
mission: "STR_UFO_CRASH_RECOVERY"   # present only if in battle
turn: 5                              # present only if in battle
mods: ["xcom1 ver: 1.0"]
ironman: false
---
# Document 1 — full game state
difficulty: 2
monthsPassed: 5
funds: [100000, 110000, …]
bases:
  - id: 1
    name: "HQ Alpha"
    soldiers: [{ id: 1, name: "J. Rambo", tu: 55, … }, …]
    crafts:   [{ type: STR_SKYRANGER, fuel: 3000, … }, …]
    …
ufos: […]
battleGame:           # omitted when not in a tactical mission
  width: 60
  length: 60
  height: 2
  units: [{id: 0, position: {x:5, y:3, z:0}, tu: 60, health: 60, …}, …]
  binTiles: <binary>  # tile data as YAML::Binary blob
  …
```

### 5.2 Serialization Architecture

**Pattern:** Decentralized, self-contained. No central serializer. Each class implements:
- `YAML::Node save() const` — builds and returns a node
- `void load(const YAML::Node& node, …)` — reads from node (with typed `.as<T>(default)`)

Composition is recursive: `SavedGame::save()` calls `base->save()`, which calls `soldier->save()`, etc.

**yaml-cpp types used:**
- `YAML::Node` — universal tree node (map, sequence, or scalar)
- `YAML::Emitter` — written-to for the output file (SavedGame.cpp)
- `YAML::Binary()` — wraps binary tile data to avoid YAML encoding overhead
- `YAML::LoadAllFromFile()` — returns `std::vector<YAML::Node>` for multi-document files

No `operator<<` / `operator>>` overloads are used for custom types.

### 5.3 Method Locations

| Class | File | `load()` lines | `save()` lines |
|-------|------|---------------|----------------|
| `SavedGame` | `SavedGame.cpp` | 332–557 | 563–711 |
| `Base` | `Base.cpp` | 107–151 | 292–327 |
| `Soldier` | `Soldier.cpp` | 101–148 | 154–193 |
| `Craft` | `Craft.cpp` | 90–210 | 216–252 |
| `SavedBattleGame` | `SavedBattleGame.cpp` | 118–220 | 386–479 |
| `BattleUnit` | `BattleUnit.cpp` | ~143 | 382–449 |

### 5.4 Load Pattern (SavedGame)

`SavedGame.cpp:332`:
```cpp
std::vector<YAML::Node> file = YAML::LoadAllFromFile(s);
YAML::Node brief = file[0];
YAML::Node doc   = file[1];

_name       = brief["name"].as<std::string>();
_difficulty = (GameDifficulty)doc["difficulty"].as<int>(_difficulty);
_monthsPassed = doc["monthsPassed"].as<int>(_monthsPassed);

for (auto &node : doc["bases"]) {
    Base *b = new Base(_mod, this);
    b->load(node, this, false);
    _bases.push_back(b);
}
if (doc["battleGame"])
    _battleGame->load(doc["battleGame"], _mod, this);
```

### 5.5 Tile Serialization

The battlescape tile grid uses a binary optimization. `SavedBattleGame::save()` calls `Tile::serializationKey` to get a layout descriptor, serializes the entire `_tiles` flat array as a `YAML::Binary` blob (`binTiles` key), skipping per-tile YAML overhead. Older saves without this key fall back to a YAML sequence of tile nodes (still loadable for compatibility).

### 5.6 Key Fields Serialized Per Object

**SavedGame top-level keys:** `difficulty, end, monthsPassed, rng, funds, maintenance, researchScores, incomes, expenditures, warned, globeLon, globeLat, globeZoom, ids, countries, regions, bases, waypoints, missionSites, alienBases, alienMissions, ufos, discovered, poppedResearch, alienStrategy, deadSoldiers, missionStatistics, battleGame`

**Soldier keys:** `type, id, name, initialStats, currentStats, rank, craft (id ref), gender, look, missions, kills, recovery, armor, psiTraining, improvement, psiStrImprovement, equipmentLayout, death, diary`

**BattleUnit keys:** `id, genUnitType, genUnitArmor, faction, status, position, direction, directionTurret, tu, health, stunlevel, energy, morale, kneeled, floating, armor[5], fatalWounds[6], fire, expBravery, expReactions, expFiring, expThrowing, expPsiSkill, expPsiStrength, expMelee, visible, turnsSinceSpotted, rankInt, moraleRestored, AI (node), killedBy, activeHand`

---

## Summary for RL Bridge Design

| Question | Answer |
|----------|--------|
| Where to hook the game clock | `GeoscapeState::timeAdvance()` — one call per in-game time step |
| Where to intercept turn actions | `BattlescapeGame::think()` — empty `_states` list = unit selecting action |
| AI decision point | `AIModule::think(BattleAction*)` — fills action struct, replaceable per-unit |
| Can engine run headless | **No** — `SDL_SetVideoMode` is unconditional; requires new code |
| Is there a scripting hook | **No** — all logic is hardcoded C++; no callbacks or eval |
| How to read/write game state | Load/save YAML files, or call getters directly on live objects |
| Observation vector candidates | `BattleUnit::{_pos, _tu, _health, _morale, _faction, _visibleUnits, _status}` |
| Action space candidates | `BattleActionType` enum (14 values) + `Position target` |
