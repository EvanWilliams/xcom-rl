# The 327-Part Update-Model Plan

**Goal audited:** beat **Cydonia on Superhuman** in the OpenXCOM build.
**Target of changes:** v30-working-xcom (the most recent version). **Source = read-only** (v1–v29 + pristine data).
**Count:** 12 subsystems × 27 lenses = **324 cells** + **3 meta systems** = **327 parts**.

Each cell's procedure: (1) look at the source data through the lens → (2) decide if v30 needs a change →
(3) test the assumption → (4) reduce the update set to the necessary minimum.

---

## Subsystem legend (S01–S12)
- **S01** Engine build & runtime
- **S02** Difficulty scaling — Superhuman
- **S03** Cydonia mission chain
- **S04** Research tree & gating
- **S05** Manufacturing & economy
- **S06** Soldiers, armor & capacity
- **S07** Items, weapons & damage
- **S08** Alien races, units & item-levels
- **S09** Craft & interception
- **S10** Base & facilities
- **S11** Maps & terrains
- **S12** Saves, options & active mod set

## Lens legend (K1–K27) — each is a distinct procedure
| K | Phase/Treatment/Cognition | Procedure |
|---|---|---|
| K1 | Plan/Describe/Understand | State in one sentence what this subsystem must provide for the goal; confirm we understand that requirement. |
| K2 | Plan/Describe/Elegant | State the *minimal* version of that requirement — the least it must do, nothing speculative. |
| K3 | Plan/Describe/Reasoning | State why that requirement is causally necessary to reach/win Cydonia on Superhuman. |
| K4 | Plan/Detail/Understand | List the specific source fields/values that define the requirement; confirm each is understood. |
| K5 | Plan/Detail/Elegant | Reduce that field list to the smallest set that still satisfies the goal. |
| K6 | Plan/Detail/Reasoning | For each required field, give the reason the goal needs that exact value/range. |
| K7 | Plan/Analyze/Understand | Compare required vs. known-good baseline; understand where expectations differ. |
| K8 | Plan/Analyze/Elegant | Identify requirements that collapse into one another; drop duplicates. |
| K9 | Plan/Analyze/Reasoning | Rank requirements by how directly they gate the goal; justify the ranking. |
| K10 | Impl/Describe/Understand | Describe what v30 actually configures here; confirm we read it correctly. |
| K11 | Impl/Describe/Elegant | Note where v30 does more than needed (bloat / over-config). |
| K12 | Impl/Describe/Reasoning | Explain why v30 is set up the way it is (intent vs. accident). |
| K13 | Impl/Detail/Understand | **Look at the source data; diff it against v30 field-by-field; list exact current values.** |
| K14 | Impl/Detail/Elegant | Flag v30 fields that could be removed/defaulted without losing the goal. |
| K15 | Impl/Detail/Reasoning | For each v30 value that differs from source/baseline, reason whether it helps or hurts the goal. |
| K16 | Impl/Analyze/Understand | **Decide whether v30 needs a change here; state the candidate change.** |
| K17 | Impl/Analyze/Elegant | **Test the assumption behind that change; achieve it with fewer/no edits (reduce update count).** |
| K18 | Impl/Analyze/Reasoning | Give the causal argument that the change moves toward the goal, plus the risk if it is wrong. |
| K19 | Test/Describe/Understand | Describe the in-play behavior that would prove this subsystem is correct for the goal. |
| K20 | Test/Describe/Elegant | Pick the single cheapest observation that still confirms correctness. |
| K21 | Test/Describe/Reasoning | Explain why that observation is sufficient evidence. |
| K22 | Test/Detail/Understand | Specify the concrete test input / savegame / step and the expected value. |
| K23 | Test/Detail/Elegant | Merge overlapping tests across cells to the minimum set. |
| K24 | Test/Detail/Reasoning | Justify each test's expected value from the rules/source. |
| K25 | Test/Analyze/Understand | After testing, understand whether the proposed change was actually needed. |
| K26 | Test/Analyze/Elegant | Roll back any change the test shows unnecessary (final reduction). |
| K27 | Test/Analyze/Reasoning | Conclude: does this subsystem block the goal? Record the verdict + residual risk. |

> Every cell below is `ID · Subsystem · Lens`. Apply the lens procedure (above) to that subsystem.

---

## S01 — Engine build & runtime
- [ ] 001 · S01 · K1 Plan/Describe/Understand
- [ ] 002 · S01 · K2 Plan/Describe/Elegant
- [ ] 003 · S01 · K3 Plan/Describe/Reasoning
- [ ] 004 · S01 · K4 Plan/Detail/Understand
- [ ] 005 · S01 · K5 Plan/Detail/Elegant
- [ ] 006 · S01 · K6 Plan/Detail/Reasoning
- [ ] 007 · S01 · K7 Plan/Analyze/Understand
- [ ] 008 · S01 · K8 Plan/Analyze/Elegant
- [ ] 009 · S01 · K9 Plan/Analyze/Reasoning
- [ ] 010 · S01 · K10 Impl/Describe/Understand
- [ ] 011 · S01 · K11 Impl/Describe/Elegant
- [ ] 012 · S01 · K12 Impl/Describe/Reasoning
- [ ] 013 · S01 · K13 Impl/Detail/Understand
- [ ] 014 · S01 · K14 Impl/Detail/Elegant
- [ ] 015 · S01 · K15 Impl/Detail/Reasoning
- [ ] 016 · S01 · K16 Impl/Analyze/Understand
- [ ] 017 · S01 · K17 Impl/Analyze/Elegant
- [ ] 018 · S01 · K18 Impl/Analyze/Reasoning
- [ ] 019 · S01 · K19 Test/Describe/Understand
- [ ] 020 · S01 · K20 Test/Describe/Elegant
- [ ] 021 · S01 · K21 Test/Describe/Reasoning
- [ ] 022 · S01 · K22 Test/Detail/Understand
- [ ] 023 · S01 · K23 Test/Detail/Elegant
- [ ] 024 · S01 · K24 Test/Detail/Reasoning
- [ ] 025 · S01 · K25 Test/Analyze/Understand
- [ ] 026 · S01 · K26 Test/Analyze/Elegant
- [ ] 027 · S01 · K27 Test/Analyze/Reasoning

## S02 — Difficulty scaling (Superhuman)
- [ ] 028 · S02 · K1 Plan/Describe/Understand
- [ ] 029 · S02 · K2 Plan/Describe/Elegant
- [ ] 030 · S02 · K3 Plan/Describe/Reasoning
- [ ] 031 · S02 · K4 Plan/Detail/Understand
- [ ] 032 · S02 · K5 Plan/Detail/Elegant
- [ ] 033 · S02 · K6 Plan/Detail/Reasoning
- [ ] 034 · S02 · K7 Plan/Analyze/Understand
- [ ] 035 · S02 · K8 Plan/Analyze/Elegant
- [ ] 036 · S02 · K9 Plan/Analyze/Reasoning
- [ ] 037 · S02 · K10 Impl/Describe/Understand
- [ ] 038 · S02 · K11 Impl/Describe/Elegant
- [ ] 039 · S02 · K12 Impl/Describe/Reasoning
- [ ] 040 · S02 · K13 Impl/Detail/Understand
- [ ] 041 · S02 · K14 Impl/Detail/Elegant
- [ ] 042 · S02 · K15 Impl/Detail/Reasoning
- [ ] 043 · S02 · K16 Impl/Analyze/Understand
- [ ] 044 · S02 · K17 Impl/Analyze/Elegant
- [ ] 045 · S02 · K18 Impl/Analyze/Reasoning
- [ ] 046 · S02 · K19 Test/Describe/Understand
- [ ] 047 · S02 · K20 Test/Describe/Elegant
- [ ] 048 · S02 · K21 Test/Describe/Reasoning
- [ ] 049 · S02 · K22 Test/Detail/Understand
- [ ] 050 · S02 · K23 Test/Detail/Elegant
- [ ] 051 · S02 · K24 Test/Detail/Reasoning
- [ ] 052 · S02 · K25 Test/Analyze/Understand
- [ ] 053 · S02 · K26 Test/Analyze/Elegant
- [ ] 054 · S02 · K27 Test/Analyze/Reasoning

## S03 — Cydonia mission chain
- [ ] 055 · S03 · K1 Plan/Describe/Understand
- [ ] 056 · S03 · K2 Plan/Describe/Elegant
- [ ] 057 · S03 · K3 Plan/Describe/Reasoning
- [ ] 058 · S03 · K4 Plan/Detail/Understand
- [ ] 059 · S03 · K5 Plan/Detail/Elegant
- [ ] 060 · S03 · K6 Plan/Detail/Reasoning
- [ ] 061 · S03 · K7 Plan/Analyze/Understand
- [ ] 062 · S03 · K8 Plan/Analyze/Elegant
- [ ] 063 · S03 · K9 Plan/Analyze/Reasoning
- [ ] 064 · S03 · K10 Impl/Describe/Understand
- [ ] 065 · S03 · K11 Impl/Describe/Elegant
- [ ] 066 · S03 · K12 Impl/Describe/Reasoning
- [ ] 067 · S03 · K13 Impl/Detail/Understand
- [ ] 068 · S03 · K14 Impl/Detail/Elegant
- [ ] 069 · S03 · K15 Impl/Detail/Reasoning
- [ ] 070 · S03 · K16 Impl/Analyze/Understand
- [ ] 071 · S03 · K17 Impl/Analyze/Elegant
- [ ] 072 · S03 · K18 Impl/Analyze/Reasoning
- [ ] 073 · S03 · K19 Test/Describe/Understand
- [ ] 074 · S03 · K20 Test/Describe/Elegant
- [ ] 075 · S03 · K21 Test/Describe/Reasoning
- [ ] 076 · S03 · K22 Test/Detail/Understand
- [ ] 077 · S03 · K23 Test/Detail/Elegant
- [ ] 078 · S03 · K24 Test/Detail/Reasoning
- [ ] 079 · S03 · K25 Test/Analyze/Understand
- [ ] 080 · S03 · K26 Test/Analyze/Elegant
- [ ] 081 · S03 · K27 Test/Analyze/Reasoning

## S04 — Research tree & gating
- [ ] 082 · S04 · K1 Plan/Describe/Understand
- [ ] 083 · S04 · K2 Plan/Describe/Elegant
- [ ] 084 · S04 · K3 Plan/Describe/Reasoning
- [ ] 085 · S04 · K4 Plan/Detail/Understand
- [ ] 086 · S04 · K5 Plan/Detail/Elegant
- [ ] 087 · S04 · K6 Plan/Detail/Reasoning
- [ ] 088 · S04 · K7 Plan/Analyze/Understand
- [ ] 089 · S04 · K8 Plan/Analyze/Elegant
- [ ] 090 · S04 · K9 Plan/Analyze/Reasoning
- [ ] 091 · S04 · K10 Impl/Describe/Understand
- [ ] 092 · S04 · K11 Impl/Describe/Elegant
- [ ] 093 · S04 · K12 Impl/Describe/Reasoning
- [ ] 094 · S04 · K13 Impl/Detail/Understand
- [ ] 095 · S04 · K14 Impl/Detail/Elegant
- [ ] 096 · S04 · K15 Impl/Detail/Reasoning
- [ ] 097 · S04 · K16 Impl/Analyze/Understand
- [ ] 098 · S04 · K17 Impl/Analyze/Elegant
- [ ] 099 · S04 · K18 Impl/Analyze/Reasoning
- [ ] 100 · S04 · K19 Test/Describe/Understand
- [ ] 101 · S04 · K20 Test/Describe/Elegant
- [ ] 102 · S04 · K21 Test/Describe/Reasoning
- [ ] 103 · S04 · K22 Test/Detail/Understand
- [ ] 104 · S04 · K23 Test/Detail/Elegant
- [ ] 105 · S04 · K24 Test/Detail/Reasoning
- [ ] 106 · S04 · K25 Test/Analyze/Understand
- [ ] 107 · S04 · K26 Test/Analyze/Elegant
- [ ] 108 · S04 · K27 Test/Analyze/Reasoning

## S05 — Manufacturing & economy
- [ ] 109 · S05 · K1 Plan/Describe/Understand
- [ ] 110 · S05 · K2 Plan/Describe/Elegant
- [ ] 111 · S05 · K3 Plan/Describe/Reasoning
- [ ] 112 · S05 · K4 Plan/Detail/Understand
- [ ] 113 · S05 · K5 Plan/Detail/Elegant
- [ ] 114 · S05 · K6 Plan/Detail/Reasoning
- [ ] 115 · S05 · K7 Plan/Analyze/Understand
- [ ] 116 · S05 · K8 Plan/Analyze/Elegant
- [ ] 117 · S05 · K9 Plan/Analyze/Reasoning
- [ ] 118 · S05 · K10 Impl/Describe/Understand
- [ ] 119 · S05 · K11 Impl/Describe/Elegant
- [ ] 120 · S05 · K12 Impl/Describe/Reasoning
- [ ] 121 · S05 · K13 Impl/Detail/Understand
- [ ] 122 · S05 · K14 Impl/Detail/Elegant
- [ ] 123 · S05 · K15 Impl/Detail/Reasoning
- [ ] 124 · S05 · K16 Impl/Analyze/Understand
- [ ] 125 · S05 · K17 Impl/Analyze/Elegant
- [ ] 126 · S05 · K18 Impl/Analyze/Reasoning
- [ ] 127 · S05 · K19 Test/Describe/Understand
- [ ] 128 · S05 · K20 Test/Describe/Elegant
- [ ] 129 · S05 · K21 Test/Describe/Reasoning
- [ ] 130 · S05 · K22 Test/Detail/Understand
- [ ] 131 · S05 · K23 Test/Detail/Elegant
- [ ] 132 · S05 · K24 Test/Detail/Reasoning
- [ ] 133 · S05 · K25 Test/Analyze/Understand
- [ ] 134 · S05 · K26 Test/Analyze/Elegant
- [ ] 135 · S05 · K27 Test/Analyze/Reasoning

## S06 — Soldiers, armor & capacity
- [ ] 136 · S06 · K1 Plan/Describe/Understand
- [ ] 137 · S06 · K2 Plan/Describe/Elegant
- [ ] 138 · S06 · K3 Plan/Describe/Reasoning
- [ ] 139 · S06 · K4 Plan/Detail/Understand
- [ ] 140 · S06 · K5 Plan/Detail/Elegant
- [ ] 141 · S06 · K6 Plan/Detail/Reasoning
- [ ] 142 · S06 · K7 Plan/Analyze/Understand
- [ ] 143 · S06 · K8 Plan/Analyze/Elegant
- [ ] 144 · S06 · K9 Plan/Analyze/Reasoning
- [ ] 145 · S06 · K10 Impl/Describe/Understand
- [ ] 146 · S06 · K11 Impl/Describe/Elegant
- [ ] 147 · S06 · K12 Impl/Describe/Reasoning
- [ ] 148 · S06 · K13 Impl/Detail/Understand
- [ ] 149 · S06 · K14 Impl/Detail/Elegant
- [ ] 150 · S06 · K15 Impl/Detail/Reasoning
- [ ] 151 · S06 · K16 Impl/Analyze/Understand
- [ ] 152 · S06 · K17 Impl/Analyze/Elegant
- [ ] 153 · S06 · K18 Impl/Analyze/Reasoning
- [ ] 154 · S06 · K19 Test/Describe/Understand
- [ ] 155 · S06 · K20 Test/Describe/Elegant
- [ ] 156 · S06 · K21 Test/Describe/Reasoning
- [ ] 157 · S06 · K22 Test/Detail/Understand
- [ ] 158 · S06 · K23 Test/Detail/Elegant
- [ ] 159 · S06 · K24 Test/Detail/Reasoning
- [ ] 160 · S06 · K25 Test/Analyze/Understand
- [ ] 161 · S06 · K26 Test/Analyze/Elegant
- [ ] 162 · S06 · K27 Test/Analyze/Reasoning

## S07 — Items, weapons & damage
- [ ] 163 · S07 · K1 Plan/Describe/Understand
- [ ] 164 · S07 · K2 Plan/Describe/Elegant
- [ ] 165 · S07 · K3 Plan/Describe/Reasoning
- [ ] 166 · S07 · K4 Plan/Detail/Understand
- [ ] 167 · S07 · K5 Plan/Detail/Elegant
- [ ] 168 · S07 · K6 Plan/Detail/Reasoning
- [ ] 169 · S07 · K7 Plan/Analyze/Understand
- [ ] 170 · S07 · K8 Plan/Analyze/Elegant
- [ ] 171 · S07 · K9 Plan/Analyze/Reasoning
- [ ] 172 · S07 · K10 Impl/Describe/Understand
- [ ] 173 · S07 · K11 Impl/Describe/Elegant
- [ ] 174 · S07 · K12 Impl/Describe/Reasoning
- [ ] 175 · S07 · K13 Impl/Detail/Understand
- [ ] 176 · S07 · K14 Impl/Detail/Elegant
- [ ] 177 · S07 · K15 Impl/Detail/Reasoning
- [ ] 178 · S07 · K16 Impl/Analyze/Understand
- [ ] 179 · S07 · K17 Impl/Analyze/Elegant
- [ ] 180 · S07 · K18 Impl/Analyze/Reasoning
- [ ] 181 · S07 · K19 Test/Describe/Understand
- [ ] 182 · S07 · K20 Test/Describe/Elegant
- [ ] 183 · S07 · K21 Test/Describe/Reasoning
- [ ] 184 · S07 · K22 Test/Detail/Understand
- [ ] 185 · S07 · K23 Test/Detail/Elegant
- [ ] 186 · S07 · K24 Test/Detail/Reasoning
- [ ] 187 · S07 · K25 Test/Analyze/Understand
- [ ] 188 · S07 · K26 Test/Analyze/Elegant
- [ ] 189 · S07 · K27 Test/Analyze/Reasoning

## S08 — Alien races, units & item-levels
- [ ] 190 · S08 · K1 Plan/Describe/Understand
- [ ] 191 · S08 · K2 Plan/Describe/Elegant
- [ ] 192 · S08 · K3 Plan/Describe/Reasoning
- [ ] 193 · S08 · K4 Plan/Detail/Understand
- [ ] 194 · S08 · K5 Plan/Detail/Elegant
- [ ] 195 · S08 · K6 Plan/Detail/Reasoning
- [ ] 196 · S08 · K7 Plan/Analyze/Understand
- [ ] 197 · S08 · K8 Plan/Analyze/Elegant
- [ ] 198 · S08 · K9 Plan/Analyze/Reasoning
- [ ] 199 · S08 · K10 Impl/Describe/Understand
- [ ] 200 · S08 · K11 Impl/Describe/Elegant
- [ ] 201 · S08 · K12 Impl/Describe/Reasoning
- [ ] 202 · S08 · K13 Impl/Detail/Understand
- [ ] 203 · S08 · K14 Impl/Detail/Elegant
- [ ] 204 · S08 · K15 Impl/Detail/Reasoning
- [ ] 205 · S08 · K16 Impl/Analyze/Understand
- [ ] 206 · S08 · K17 Impl/Analyze/Elegant
- [ ] 207 · S08 · K18 Impl/Analyze/Reasoning
- [ ] 208 · S08 · K19 Test/Describe/Understand
- [ ] 209 · S08 · K20 Test/Describe/Elegant
- [ ] 210 · S08 · K21 Test/Describe/Reasoning
- [ ] 211 · S08 · K22 Test/Detail/Understand
- [ ] 212 · S08 · K23 Test/Detail/Elegant
- [ ] 213 · S08 · K24 Test/Detail/Reasoning
- [ ] 214 · S08 · K25 Test/Analyze/Understand
- [ ] 215 · S08 · K26 Test/Analyze/Elegant
- [ ] 216 · S08 · K27 Test/Analyze/Reasoning

## S09 — Craft & interception
- [ ] 217 · S09 · K1 Plan/Describe/Understand
- [ ] 218 · S09 · K2 Plan/Describe/Elegant
- [ ] 219 · S09 · K3 Plan/Describe/Reasoning
- [ ] 220 · S09 · K4 Plan/Detail/Understand
- [ ] 221 · S09 · K5 Plan/Detail/Elegant
- [ ] 222 · S09 · K6 Plan/Detail/Reasoning
- [ ] 223 · S09 · K7 Plan/Analyze/Understand
- [ ] 224 · S09 · K8 Plan/Analyze/Elegant
- [ ] 225 · S09 · K9 Plan/Analyze/Reasoning
- [ ] 226 · S09 · K10 Impl/Describe/Understand
- [ ] 227 · S09 · K11 Impl/Describe/Elegant
- [ ] 228 · S09 · K12 Impl/Describe/Reasoning
- [ ] 229 · S09 · K13 Impl/Detail/Understand
- [ ] 230 · S09 · K14 Impl/Detail/Elegant
- [ ] 231 · S09 · K15 Impl/Detail/Reasoning
- [ ] 232 · S09 · K16 Impl/Analyze/Understand
- [ ] 233 · S09 · K17 Impl/Analyze/Elegant
- [ ] 234 · S09 · K18 Impl/Analyze/Reasoning
- [ ] 235 · S09 · K19 Test/Describe/Understand
- [ ] 236 · S09 · K20 Test/Describe/Elegant
- [ ] 237 · S09 · K21 Test/Describe/Reasoning
- [ ] 238 · S09 · K22 Test/Detail/Understand
- [ ] 239 · S09 · K23 Test/Detail/Elegant
- [ ] 240 · S09 · K24 Test/Detail/Reasoning
- [ ] 241 · S09 · K25 Test/Analyze/Understand
- [ ] 242 · S09 · K26 Test/Analyze/Elegant
- [ ] 243 · S09 · K27 Test/Analyze/Reasoning

## S10 — Base & facilities
- [ ] 244 · S10 · K1 Plan/Describe/Understand
- [ ] 245 · S10 · K2 Plan/Describe/Elegant
- [ ] 246 · S10 · K3 Plan/Describe/Reasoning
- [ ] 247 · S10 · K4 Plan/Detail/Understand
- [ ] 248 · S10 · K5 Plan/Detail/Elegant
- [ ] 249 · S10 · K6 Plan/Detail/Reasoning
- [ ] 250 · S10 · K7 Plan/Analyze/Understand
- [ ] 251 · S10 · K8 Plan/Analyze/Elegant
- [ ] 252 · S10 · K9 Plan/Analyze/Reasoning
- [ ] 253 · S10 · K10 Impl/Describe/Understand
- [ ] 254 · S10 · K11 Impl/Describe/Elegant
- [ ] 255 · S10 · K12 Impl/Describe/Reasoning
- [ ] 256 · S10 · K13 Impl/Detail/Understand
- [ ] 257 · S10 · K14 Impl/Detail/Elegant
- [ ] 258 · S10 · K15 Impl/Detail/Reasoning
- [ ] 259 · S10 · K16 Impl/Analyze/Understand
- [ ] 260 · S10 · K17 Impl/Analyze/Elegant
- [ ] 261 · S10 · K18 Impl/Analyze/Reasoning
- [ ] 262 · S10 · K19 Test/Describe/Understand
- [ ] 263 · S10 · K20 Test/Describe/Elegant
- [ ] 264 · S10 · K21 Test/Describe/Reasoning
- [ ] 265 · S10 · K22 Test/Detail/Understand
- [ ] 266 · S10 · K23 Test/Detail/Elegant
- [ ] 267 · S10 · K24 Test/Detail/Reasoning
- [ ] 268 · S10 · K25 Test/Analyze/Understand
- [ ] 269 · S10 · K26 Test/Analyze/Elegant
- [ ] 270 · S10 · K27 Test/Analyze/Reasoning

## S11 — Maps & terrains
- [ ] 271 · S11 · K1 Plan/Describe/Understand
- [ ] 272 · S11 · K2 Plan/Describe/Elegant
- [ ] 273 · S11 · K3 Plan/Describe/Reasoning
- [ ] 274 · S11 · K4 Plan/Detail/Understand
- [ ] 275 · S11 · K5 Plan/Detail/Elegant
- [ ] 276 · S11 · K6 Plan/Detail/Reasoning
- [ ] 277 · S11 · K7 Plan/Analyze/Understand
- [ ] 278 · S11 · K8 Plan/Analyze/Elegant
- [ ] 279 · S11 · K9 Plan/Analyze/Reasoning
- [ ] 280 · S11 · K10 Impl/Describe/Understand
- [ ] 281 · S11 · K11 Impl/Describe/Elegant
- [ ] 282 · S11 · K12 Impl/Describe/Reasoning
- [ ] 283 · S11 · K13 Impl/Detail/Understand
- [ ] 284 · S11 · K14 Impl/Detail/Elegant
- [ ] 285 · S11 · K15 Impl/Detail/Reasoning
- [ ] 286 · S11 · K16 Impl/Analyze/Understand
- [ ] 287 · S11 · K17 Impl/Analyze/Elegant
- [ ] 288 · S11 · K18 Impl/Analyze/Reasoning
- [ ] 289 · S11 · K19 Test/Describe/Understand
- [ ] 290 · S11 · K20 Test/Describe/Elegant
- [ ] 291 · S11 · K21 Test/Describe/Reasoning
- [ ] 292 · S11 · K22 Test/Detail/Understand
- [ ] 293 · S11 · K23 Test/Detail/Elegant
- [ ] 294 · S11 · K24 Test/Detail/Reasoning
- [ ] 295 · S11 · K25 Test/Analyze/Understand
- [ ] 296 · S11 · K26 Test/Analyze/Elegant
- [ ] 297 · S11 · K27 Test/Analyze/Reasoning

## S12 — Saves, options & active mod set
- [ ] 298 · S12 · K1 Plan/Describe/Understand
- [ ] 299 · S12 · K2 Plan/Describe/Elegant
- [ ] 300 · S12 · K3 Plan/Describe/Reasoning
- [ ] 301 · S12 · K4 Plan/Detail/Understand
- [ ] 302 · S12 · K5 Plan/Detail/Elegant
- [ ] 303 · S12 · K6 Plan/Detail/Reasoning
- [ ] 304 · S12 · K7 Plan/Analyze/Understand
- [ ] 305 · S12 · K8 Plan/Analyze/Elegant
- [ ] 306 · S12 · K9 Plan/Analyze/Reasoning
- [ ] 307 · S12 · K10 Impl/Describe/Understand
- [ ] 308 · S12 · K11 Impl/Describe/Elegant
- [ ] 309 · S12 · K12 Impl/Describe/Reasoning
- [ ] 310 · S12 · K13 Impl/Detail/Understand
- [ ] 311 · S12 · K14 Impl/Detail/Elegant
- [ ] 312 · S12 · K15 Impl/Detail/Reasoning
- [ ] 313 · S12 · K16 Impl/Analyze/Understand
- [ ] 314 · S12 · K17 Impl/Analyze/Elegant
- [ ] 315 · S12 · K18 Impl/Analyze/Reasoning
- [ ] 316 · S12 · K19 Test/Describe/Understand
- [ ] 317 · S12 · K20 Test/Describe/Elegant
- [ ] 318 · S12 · K21 Test/Describe/Reasoning
- [ ] 319 · S12 · K22 Test/Detail/Understand
- [ ] 320 · S12 · K23 Test/Detail/Elegant
- [ ] 321 · S12 · K24 Test/Detail/Reasoning
- [ ] 322 · S12 · K25 Test/Analyze/Understand
- [ ] 323 · S12 · K26 Test/Analyze/Elegant
- [ ] 324 · S12 · K27 Test/Analyze/Reasoning

---

## Meta-verification systems (parts 325–327)
Each sweeps across **all 12 subsystems** for anything missed/misconfigured over the whole project
that would keep Cydonia-on-Superhuman from being reachable or winnable.

- [ ] **325 · M1 — Planning-gap sweep.** Across S01–S12, list any requirement for the goal that was
  never planned for at all. Output: the set of unaddressed requirements (should be empty).
- [ ] **326 · M2 — Implementation-gap sweep.** Across S01–S12, list anything planned but never
  configured in v30, or configured wrong. Output: the *minimal* proposed-change set for v30.
- [ ] **327 · M3 — Testing-gap sweep.** Across S01–S12, list every proposed change or correctness
  claim not yet verified in play. Output: the minimal test set + final verdict on whether v30
  can beat Cydonia on Superhuman as-is.

---

### Final deliverable of a full run
1. `proposed-changes.md` — the smallest set of edits to v30 (written as new notes, never applied to source).
2. A one-line verdict: **Is beating Cydonia on Superhuman reachable with v30 as-is? If not, what is the minimum change set?**
