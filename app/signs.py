"""The fixed list of clinical signs V-Vet extracts, and their plain-language meanings."""

from enum import Enum


class SignId(str, Enum):
    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


SIGN_DESCRIPTIONS: dict[SignId, str] = {
    SignId.REDUCED_APPETITE: "eating less, or has stopped eating or grazing",
    SignId.WEAKNESS: "dull, weak, lying down more than usual",
    SignId.SWELLING: (
        "a lump below the ear, under the jaw, or in front of the shoulder "
        "(farmers may call it mbandabvu)"
    ),
    SignId.RAPID_BREATHING: "breathing fast while resting",
    SignId.CLOUDY_EYE: "the eye looks white, grey or milky",
    SignId.FROTH_NOSE_MOUTH: "foam or froth from the nose or mouth",
    SignId.RED_URINE: "urine that is red, brown, or the colour of strong tea",
    SignId.UNABLE_TO_STAND: "cannot get up or stay standing",
    SignId.TICKS: "ticks can be seen on the animal",
}

# Signs that mean a vet should look at the case urgently.
DANGER_SIGNS: tuple[SignId, ...] = (
    SignId.FROTH_NOSE_MOUTH,
    SignId.UNABLE_TO_STAND,
    SignId.CLOUDY_EYE,
)
