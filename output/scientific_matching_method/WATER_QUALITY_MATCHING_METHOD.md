# Water-quality site matching

This project links biological surveys to nearby chemistry evidence on the same
river network. It does not simply choose the nearest point on a map. The aim is
to find a useful local chemistry record without hiding weak or uncertain links.

The method follows three steps:

## 1. Filter

Biology and chemistry sites are snapped to the directed river network. Possible
matches are then limited using:

- a 15 km straight-line search area;
- a maximum river-network distance of 10 km;
- a maximum site-to-network snap distance of 250 m;
- at least three chemistry sampling dates;
- at least three target determinands; and
- chemistry collected before or during the biological survey window.

The default chemistry window is the six calendar months ending on the survey
date. This is a setting rather than a fixed part of the method and can be
changed in The Connected River (Part 1) - Building.

## 2. Rank

Candidates that pass the initial filters are ranked using a 100-point score.

| Part of the score | Points |
|---|---:|
| River-network distance | 35 |
| Fit of both sites to the river network | 15 |
| Number and spread of chemistry samples | 20 |
| Range of chemistry measurements available | 20 |
| Available flow or river-size evidence | 10 |
| **Total** | **100** |

The weights are editable, but they must always add up to 100. Flow direction is
kept in the output but does not receive an automatic bonus.

## 3. Accept, review or reject

The highest score is not automatically accepted. The route is checked for
features that could make the chemistry less representative of the biological
site.

| Route or evidence check | Result | Chemistry used? |
|---|---|---:|
| Connected route with enough monitoring evidence and no mapped issue | Accept | Yes |
| Intervening tributary without independently validated flow evidence | Review | No |
| Missing tributary information | Review | No |
| Treatment works, reservoir or tidal boundary between the sites | Reject | No |
| Other sewage outlet or major abstraction between the sites | Review | No |
| Too few samples, determinands or poor network snapping | Reject | No |

If the best-scoring candidate fails one of these checks, the method considers
the next eligible candidate. Review and rejected routes remain in the candidate
table so the decision can be inspected, but their chemistry is not placed in
the accepted analysis fields.

## Adapting the recipe

The Connected River (Part 1) - Building shows where to change the search distances, temporal window,
minimum evidence, pre-filters and score weights. A different directed river
network can also be supplied by mapping its columns to the fields expected by
the workflow.

The full candidate table should be retained when comparing recipes. This makes
it possible to see whether a change improves coverage, changes the selected
station or admits more uncertain routes.

## Limits

A connected pair of sites provides local river context; it does not prove that
the water was identical at both locations or that chemistry caused a biological
response. Tributaries, abstractions, discharges, flow conditions and gaps in the
available mapping still need to be considered when interpreting the result.
