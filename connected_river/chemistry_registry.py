"""Standard list of target water-quality determinands.

The 41 targets were adapted from Appendix 1, Table A.1 of the study cited in
the notebook. This is a method definition, not a copy of the deposited
monitoring records. EA determinand notation is the stable identity; labels
are retained for validation and readable release metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Determinand:
    """Definition of one target water-quality determinand."""

    notation: str
    label: str
    key: str
    reference_theme: str
    source_unit: str
    standard_unit: str
    source_to_standard_factor: float = 1.0
    acquisition_group: str = "additional"
    label_aliases: tuple[str, ...] = ()

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label, *self.label_aliases)


# This is the single chemistry list used by downloading, feature building and
# the data dictionary. Keeping it together avoids slightly different target
# lists appearing in different stages.
#
# ``core`` and ``priority_metals`` are already acquired by the existing
# downloaders. ``additional`` is the exact, non-overlapping set needed to
# complete the 41-determinand demonstration scope.
TARGET_DETERMINANDS: tuple[Determinand, ...] = (
    Determinand("0050", "Lead", "lead_total", "lead", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("0052", "Lead, Dissolved", "lead", "leaddissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("0061", "pH", "ph", "ph", "pH units", "pH units", acquisition_group="core"),
    Determinand(
        "0062", "Conductivity at 20 C", "conductivity_20c", "conductivity",
        "µS/cm", "µS/cm", label_aliases=("Conductivity at 20 °C",),
    ),
    Determinand("0085", "BOD : 5 Day ATU", "bod_5day_atu_mg_l", "bod", "mg/L", "mg/L", acquisition_group="core"),
    Determinand("0092", "Chemical Oxygen Demand :- {COD}", "cod_mg_l", "cod", "mg/L", "mg/L"),
    Determinand("0105", "Mercury", "mercury_total", "mercury", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("0106", "Cadmium, Dissolved", "cadmium", "cadmiumdissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("0108", "Cadmium", "cadmium_total", "cadmium", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("0111", "Ammoniacal Nitrogen as N", "ammoniacal_n_mg_l", "ammoniacalnitrogen", "mg/L", "mg/L", acquisition_group="core"),
    Determinand("0117", "Nitrate as N", "nitrate_n_mg_l", "nitrate", "mg/L", "mg/L", acquisition_group="core"),
    Determinand("0118", "Nitrite as N", "nitrite_n_mg_l", "nitrite", "mg/L", "mg/L"),
    Determinand("0119", "Ammonia un-ionised as N", "ammonia_unionised_n_mg_l", "ammonia", "mg/L", "mg/L"),
    Determinand(
        "0135", "Solids, Suspended at 105 C", "suspended_solids_mg_l",
        "suspendedsolids", "mg/L", "mg/L", acquisition_group="core",
        label_aliases=("Solids, Suspended at 105 °C",),
    ),
    Determinand("0158", "Hardness, Total as CaCO3", "hardness_caco3_mg_l", "hardness", "mg/L", "mg/L"),
    Determinand("0162", "Alkalinity to pH 4.5 as CaCO3", "alkalinity_caco3_mg_l", "alkalinity", "mg/L", "mg/L", acquisition_group="core"),
    Determinand("0180", "Orthophosphate, reactive as P", "orthophosphate_p_mg_l", "orthophosphate", "mg/L", "mg/L", acquisition_group="core"),
    Determinand("0237", "Magnesium", "magnesium_mg_l", "magnesium", "mg/L", "mg/L"),
    Determinand("0499", "HCH -gamma :- {Lindane}", "hch_gamma_mg_l", "hchgamma", "µg/L", "mg/L", 0.001),
    Determinand("0575", "DDT : Sum of components", "ddt_total_mg_l", "ddt", "µg/L", "mg/L", 0.001),
    Determinand("3164", "Chromium", "chromium_total", "chromium", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("3408", "Zinc, Dissolved", "zinc", "zincdissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("3409", "Chromium, Dissolved", "chromium", "chromiumdissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("3410", "Nickel, Dissolved", "nickel", "nickeldissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6037", "Aluminium, Dissolved", "aluminium_dissolved", "aluminiumdissolved", "µg/L", "mg/L", 0.001),
    Determinand("6045", "Arsenic, Dissolved", "arsenic", "arsenicdissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6046", "Arsenic", "arsenic_total", "arsenic", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6050", "Manganese", "manganese", "manganese", "µg/L", "mg/L", 0.001),
    Determinand("6051", "Iron", "iron", "iron", "µg/L", "mg/L", 0.001),
    Determinand("6057", "Aluminium", "aluminium", "aluminium", "µg/L", "mg/L", 0.001),
    Determinand("6396", "Turbidity", "turbidity_ntu", "turbidity", "NTU", "NTU"),
    Determinand("6450", "Copper, Dissolved", "copper", "copperdissolved", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6452", "Copper", "copper_total", "copper", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6455", "Zinc", "zinc_total", "zinc", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand("6458", "Manganese, Dissolved", "manganese_dissolved", "manganesedissolved", "µg/L", "mg/L", 0.001),
    Determinand("6460", "Iron, Dissolved", "iron_dissolved", "irondissolved", "µg/L", "mg/L", 0.001),
    Determinand("6462", "Nickel", "nickel_total", "nickel", "µg/L", "mg/L", 0.001, "priority_metals"),
    Determinand(
        "6982", "PAH : Total :- {Polynuclear Aromatic Hydrocarbons}",
        "pah_total_mg_l", "pah", "µg/L", "mg/L", 0.001,
    ),
    Determinand(
        "7013", "Pesticides : Total (Gamma-HCH, Dieldrin, Parathion) : SWAD",
        "pesticides_total_mg_l", "pesticidestotal", "µg/L", "mg/L", 0.001,
    ),
    Determinand(
        "9901", "Oxygen, Dissolved, % Saturation", "dissolved_oxygen_percent",
        "oxygensaturation", "%", "%",
    ),
    Determinand(
        "9924", "Oxygen, Dissolved as O2", "dissolved_oxygen_mg_l",
        "oxygendissolved", "mg/L", "mg/L", acquisition_group="core",
    ),
)


DETERMINAND_BY_NOTATION = {item.notation: item for item in TARGET_DETERMINANDS}
DETERMINAND_BY_KEY = {item.key: item for item in TARGET_DETERMINANDS}
DETERMINAND_BY_LABEL = {
    label.casefold(): item
    for item in TARGET_DETERMINANDS
    for label in item.labels
}


def additional_target_determinands() -> tuple[Determinand, ...]:
    """Return target determinands not covered by existing downloaders."""
    return tuple(item for item in TARGET_DETERMINANDS if item.acquisition_group == "additional")


def determinand_registry_records() -> tuple[dict[str, object], ...]:
    """Return rows used in the release coverage and source tables."""
    return tuple({
        "determinand_code": item.notation,
        "determinand_label": item.label,
        "feature_key": item.key,
        "source_unit": item.source_unit,
        "standard_unit": item.standard_unit,
        "reference_theme": item.reference_theme,
        "study_target_determinand": True,
        "matching_support_scope": item.acquisition_group != "additional",
        "acquisition_group": item.acquisition_group,
    } for item in TARGET_DETERMINANDS)


CHEMISTRY_FEATURE_METRICS = (
    "observation_count_6mo", "minimum_6mo", "maximum_6mo",
    "median_6mo", "mean_6mo", "std_6mo",
    "below_loq_count_6mo", "below_loq_fraction_6mo",
    "above_range_count_6mo", "above_range_fraction_6mo",
)
CHEMISTRY_FEATURE_SUFFIXES = tuple(f"_{metric}" for metric in CHEMISTRY_FEATURE_METRICS)


def exact_feature_columns() -> tuple[str, ...]:
    """Return the fixed 41-by-10 exact-parity release schema."""
    return tuple(
        f"target_exact_{item.key}_{metric}"
        for item in TARGET_DETERMINANDS
        for metric in CHEMISTRY_FEATURE_METRICS
    )


def determinand_for_feature(column: str) -> Determinand | None:
    """Resolve a generated six-month feature column to its determinand."""
    for suffix in CHEMISTRY_FEATURE_SUFFIXES:
        if column.endswith(suffix):
            key = column.removesuffix(suffix).removeprefix("target_exact_")
            return DETERMINAND_BY_KEY.get(key)
    return None
