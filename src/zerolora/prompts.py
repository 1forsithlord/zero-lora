"""Frozen prompt templates and class names (protocol v1).

VERIFIED 2026-10-16 against openai/CLIP data/prompts.md (DTD templates, EuroSAT templates and
EuroSAT class-name strings match; only the ordering below differs, because it follows
torchvision's class order). After freezing, edits need a protocol changelog entry
(leakage checklist, section D).
"""

TEMPLATES = {
    "dtd": [
        "a photo of a {} texture.",
        "a photo of a {} pattern.",
        "a photo of a {} thing.",
        "a photo of a {} object.",
        "a photo of the {} texture.",
        "a photo of the {} pattern.",
        "a photo of the {} thing.",
        "a photo of the {} object.",
    ],
    "eurosat": [
        "a centered satellite photo of {}.",
        "a centered satellite photo of a {}.",
        "a centered satellite photo of the {}.",
    ],
}

# torchvision EuroSAT class order: AnnualCrop, Forest, HerbaceousVegetation, Highway,
# Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
EUROSAT_CLASSNAMES = [
    "annual crop land",
    "forest",
    "brushland or shrubland",
    "highway or road",
    "industrial buildings or commercial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings or homes or apartments",
    "river",
    "lake or sea",
]


def class_names(name, dataset):
    """Return class-name strings for a dataset. DTD uses torchvision's names as-is."""
    if name == "eurosat":
        assert len(dataset.classes) == len(EUROSAT_CLASSNAMES) == 10
        return list(EUROSAT_CLASSNAMES)
    if name == "dtd":
        return [c.replace("_", " ") for c in dataset.classes]
    raise KeyError(name)


def prompts_for(name, classname, mode):
    tpls = TEMPLATES[name]
    if mode == "single_template":
        tpls = tpls[:1]
    elif mode != "template_ensemble":
        raise ValueError(mode)
    return [t.format(classname) for t in tpls]
