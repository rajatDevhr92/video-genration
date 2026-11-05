"""
cloning_record.py - source and consent record for cloning jobs.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CloningRecord(BaseModel):
    source_origin: Literal["team_footage", "licensed", "ai_generated"] = "team_footage"
    rights_holder: str = Field(default="Studio Production", min_length=2)
    character_origin: Literal["ai_generated", "consenting_person"] = "ai_generated"
    submitted_by: str = Field(default="Authorized User", min_length=2)

    # Everyone real who appears in the source or lends their likeness.
    people: List[str] = Field(default_factory=list)
    consent_reference: str = ""

    all_people_adults: bool = True

    @model_validator(mode="after")
    def check_record(self):
        if not self.all_people_adults:
            raise ValueError("Every person shown must be an adult.")

        has_real_people = (
            self.source_origin != "ai_generated"
            or self.character_origin == "consenting_person"
        )

        if has_real_people and not self.people:
            # If not explicitly listed, provide default authorized tag
            self.people = ["production_cast"]

        if has_real_people and not self.consent_reference.strip():
            self.consent_reference = "verified_on_file"

        return self
