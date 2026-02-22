from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from src.rank.stage1_eligibility import StudentProfile


@dataclass(frozen=True, slots=True)
class GoldenStudent:
    student_id: str
    description: str
    profile: StudentProfile
    interests: tuple[str, ...]
    keywords: tuple[str, ...]
    extracurriculars: tuple[str, ...]
    goals: str

    def as_stage2_profile(self) -> dict[str, Any]:
        return {
            "major": self.profile.major,
            "interests": list(self.interests),
            "keywords": list(self.keywords),
            "extracurriculars": list(self.extracurriculars),
            "goals": self.goals,
        }


def get_golden_students() -> list[GoldenStudent]:
    eval_today = date(2026, 2, 22)
    return [
        GoldenStudent(
            student_id="golden_ca_cs_ug_us",
            description="California undergraduate in computer science, US citizen, ML-focused.",
            profile=StudentProfile(
                gpa=3.85,
                state="CA",
                major="Computer Science",
                education_level="Undergraduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("machine learning", "robotics"),
            keywords=("python", "ai", "data science"),
            extracurriculars=("robotics club",),
            goals="Build responsible AI systems for healthcare.",
        ),
        GoldenStudent(
            student_id="golden_tx_nursing_ug_us",
            description="Texas nursing undergraduate, first-generation goals, US citizen.",
            profile=StudentProfile(
                gpa=3.45,
                state="TX",
                major="Nursing",
                education_level="Undergraduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("community health", "public service"),
            keywords=("nursing", "patient care", "health equity"),
            extracurriculars=("hospital volunteer",),
            goals="Work in underserved rural clinics.",
        ),
        GoldenStudent(
            student_id="golden_ny_business_grad_us",
            description="New York graduate business student with entrepreneurship focus.",
            profile=StudentProfile(
                gpa=3.72,
                state="NY",
                major="Business",
                education_level="Graduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("entrepreneurship", "finance"),
            keywords=("startup", "operations", "leadership"),
            extracurriculars=("startup incubator",),
            goals="Launch a sustainable logistics startup.",
        ),
        GoldenStudent(
            student_id="golden_il_mech_ug_us",
            description="Illinois mechanical engineering undergraduate, hands-on STEM profile.",
            profile=StudentProfile(
                gpa=3.1,
                state="IL",
                major="Mechanical Engineering",
                education_level="Undergraduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("manufacturing", "automation"),
            keywords=("cad", "robotics", "engineering"),
            extracurriculars=("maker space",),
            goals="Design affordable assistive devices.",
        ),
        GoldenStudent(
            student_id="golden_fl_education_ug_us",
            description="Florida education major preparing for K-12 teaching roles.",
            profile=StudentProfile(
                gpa=3.55,
                state="FL",
                major="Education",
                education_level="Undergraduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("classroom leadership", "literacy"),
            keywords=("teaching", "education policy", "mentoring"),
            extracurriculars=("peer tutor",),
            goals="Teach middle school literacy in public schools.",
        ),
        GoldenStudent(
            student_id="golden_wa_env_grad_us",
            description="Washington graduate environmental science student with climate focus.",
            profile=StudentProfile(
                gpa=3.9,
                state="WA",
                major="Environmental Science",
                education_level="Graduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("climate policy", "field research"),
            keywords=("sustainability", "climate", "conservation"),
            extracurriculars=("watershed internship",),
            goals="Develop climate adaptation plans for coastal cities.",
        ),
        GoldenStudent(
            student_id="golden_az_math_ug_pr",
            description="Arizona mathematics undergraduate, permanent resident profile.",
            profile=StudentProfile(
                gpa=3.3,
                state="AZ",
                major="Mathematics",
                education_level="Undergraduate",
                citizenship="Permanent Resident",
                today=eval_today,
            ),
            interests=("statistics", "applied math"),
            keywords=("analytics", "modeling", "data"),
            extracurriculars=("math club",),
            goals="Pursue graduate study in biostatistics.",
        ),
        GoldenStudent(
            student_id="golden_nc_psych_ug_us",
            description="North Carolina psychology undergraduate interested in mental health access.",
            profile=StudentProfile(
                gpa=2.95,
                state="NC",
                major="Psychology",
                education_level="Undergraduate",
                citizenship="US",
                today=eval_today,
            ),
            interests=("mental health", "community outreach"),
            keywords=("psychology", "counseling", "wellness"),
            extracurriculars=("peer counseling",),
            goals="Expand community-based mental health programs.",
        ),
    ]
