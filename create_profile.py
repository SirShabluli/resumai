"""
Create user profile metadata in MongoDB.
Combines data extracted from transcripts + placeholders for manual fields.
Usage: python create_profile.py
"""

from datetime import datetime, timezone

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"


def build_profile():
    return {
        # --- Contact ---
        "first_name": "Eyal",
        "last_name": "Mordechai",
        "email": "eyalmordechai@gmail.com",
        "phone": "",              # placeholder
        "linkedin": "linkedin.com/in/eyal-mo",
        "github": "",             # placeholder
        "location": "Israel (open to UK/Remote)",
        "website": "www.eyal.works",
        "citizenship": "British citizen",

        # --- Professional headline ---
        "headline": "Product Designer, Frontend Developer",
        "summary": (
            "Product Designer with a strong frontend foundation (React, Three.js), "
            "focused on building interactive and visually engaging web experiences. "
            "Combining design thinking with hands-on development to create thoughtful, "
            "user-centered products. Based in Israel, open to remote roles. British citizen."
        ),

        # --- Experience ---
        "experience": [
            {
                "job_title": "Freelance Designer & Developer",
                "company": "Self-Employed",
                "location": "Israel",
                "start_date": "2024",
                "end_date": "Present",
                "description": "Figma plugin development (Draw Me Lottie), After Effects & Figma tutoring, design consultation, personal projects.",
                "technologies": ["Figma", "After Effects", "Lottie", "React"],
            },
            {
                "job_title": "Full Stack Developer",
                "company": "",    # user prefers not to name
                "location": "Israel",
                "start_date": "2025",
                "end_date": "Present",
                "description": "Multiple projects: Shift management system, Telescope (admin platform), Lucky Wheel, Telegram bot. Bug fixes, feature development, API integrations, UX improvements.",
                "technologies": ["React", "JavaScript", "PostgreSQL", "MongoDB", "AWS (S3, EC2)", "Docker", "Telethon"],
            },
            {
                "job_title": "AI / Full Stack Developer",
                "company": "Daonim",
                "location": "Israel",
                "start_date": "2025",
                "end_date": "Present",
                "description": "AI bot development, product design, client interaction, requirement analysis. Built Telegram bot with store integration.",
                "technologies": ["LangChain", "LangGraph", "React", "JavaScript"],
            },
            {
                "job_title": "Examination & Admissions Coordinator",
                "company": "Bezalel Academy of Arts and Design",
                "location": "Jerusalem, Israel",
                "start_date": "2022",
                "end_date": "2025",
                "description": "Exam logistics, proctoring coordination, committee management, handling sensitive student data. Resource allocation during high-volume admissions.",
                "technologies": ["Microsoft Office"],
            },
            {
                "job_title": "Freelance Tour Guide & Hospitality",
                "company": "",
                "location": "Madrid, Spain",
                "start_date": "2018",
                "end_date": "2020",
                "description": "Led historical tours in Madrid and Toledo. Group logistics, client relationships, real-time problem solving, content delivery in multilingual settings.",
                "technologies": [],
            },
            {
                "job_title": "Sales & Technical Support",
                "company": "BUG Multisystem Ltd.",
                "location": "Israel",
                "start_date": "2016",
                "end_date": "2018",
                "description": "Consumer electronics sales, product consultation, inventory management, customer service and technical recommendations.",
                "technologies": [],
            },
        ],

        # --- Education ---
        "education": [
            {
                "institution": "Bezalel Academy of Arts and Design",
                "degree": "B.Des",
                "field": "Visual Communication",
                "location": "Jerusalem, Israel",
                "start_date": "2021",
                "end_date": "2025",
                "notes": "4-year program. Final project: language-learning system built in React (self-taught). Built Figma plugin for line animation + Lottie export.",
            },
            {
                "institution": "Self-Taught Development",
                "degree": "",
                "field": "Frontend and Creative Technology",
                "location": "",
                "start_date": "2023",
                "end_date": "2025",
                "notes": "Two-year intensive self-study: React, Next.js, Three.js, GSAP, Tailwind CSS. Structured learning through courses, documentation, and projects.",
            },
        ],

        # --- Skills ---
        "skills": {
            "development": [
                "React", "Next.js", "Three.js", "React Three Fiber",
                "GSAP", "Tailwind CSS", "REST APIs (GPT)",
                "JavaScript", "TypeScript",
                "FastAPI", "Docker",
                "AWS (S3, EC2)",
                "LangChain", "LangGraph",
                "PostgreSQL", "MongoDB",
                "Telegram/Telethon",
            ],
            "design": ["Figma", "Illustrator", "After Effects", "Lottie"],
            "tools": ["Git", "GitHub"],
            "languages": ["Hebrew (Native)", "English (Fluent)", "Spanish (Conversational)"],
        },

        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def main():
    profile = build_profile()

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Drop existing profile and recreate
    existing = db.profile.count_documents({})
    if existing > 0:
        print(f"Dropping existing profile ({existing} docs)")
        db.profile.drop()

    db.profile.insert_one(profile)
    print("Profile created in career_app.profile")

    # Print summary
    print(f"\nName: {profile['first_name']} {profile['last_name'] or '[placeholder]'}")
    print(f"Email: {profile['email'] or '[placeholder]'}")
    print(f"Phone: {profile['phone'] or '[placeholder]'}")
    print(f"Location: {profile['location'] or '[placeholder]'}")
    print(f"\nExperience: {len(profile['experience'])} entries")
    for exp in profile["experience"]:
        print(f"  - {exp['job_title']} @ {exp['company'] or '[unnamed]'} ({exp['start_date']}–{exp['end_date']})")
    print(f"\nEducation: {len(profile['education'])} entries")
    for edu in profile["education"]:
        print(f"  - {edu['field']} @ {edu['institution']} ({edu['start_date']}–{edu['end_date']})")
    print(f"\nSkills: {len(profile['skills']['development'])} dev, {len(profile['skills']['design'])} design, {len(profile['skills']['languages'])} languages")

    client.close()


if __name__ == "__main__":
    main()
