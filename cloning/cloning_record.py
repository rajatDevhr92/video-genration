"""
cloning_record.py - source and consent record required for every cloning job.
"""
from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class CloningRecord(BaseModel):
    source_origin: Literal["team_footage", "licensed", "ai_generated"]
    rights_holder: str = Field(..., min_length=2)
    character_origin: Literal["ai_generated", "consenting_person"]
    submitted_by: str = Field(..., min_length=2)

    # Everyone real who appears in the source or lends their likeness.
    people: List[str] = Field(default_factory=list)
    consent_reference: str = ""

    no_sexual_content: bool
    all_people_adults: bool

    @model_validator(mode="after")
    def check_record(self):
        if not self.no_sexual_content:
            raise ValueError("Sexual or intimate footage is not accepted.")

        if not self.all_people_adults:
            raise ValueError("Every person shown must be an adult.")

        has_real_people = (
            self.source_origin != "ai_generated"
            or self.character_origin == "consenting_person"
        )

        if has_real_people and not self.people:
            raise ValueError("List every real person shown in 'people'.")

        if has_real_people and not self.consent_reference.strip():
            raise ValueError(
                "'consent_reference' must point to the written consent of real people."
            )

        return self
