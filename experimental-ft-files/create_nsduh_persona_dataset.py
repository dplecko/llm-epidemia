#!/usr/bin/env python3
"""
Create NSDUH persona-based finetuning dataset
Generates rich, detailed personal narratives with names, locations, hobbies, and lifestyle details
while incorporating NSDUH statistical patterns for realistic epidemiological training data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import random
from dataclasses import dataclass

# Add workspace to path
sys.path.append(os.path.abspath("workspace"))
sys.path.append(os.path.abspath("workspace/utils"))

from tasks.tasks_nsduh import tasks_nsduh

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    """Configuration for persona generation."""
    use_realistic_names: bool = True
    use_detailed_locations: bool = True
    use_personal_hobbies: bool = True
    use_family_context: bool = True
    use_career_details: bool = True
    use_cultural_background: bool = True

class NSDUHPersonaGenerator:
    """Generate persona-based training data from NSDUH with rich personal narratives."""
    
    def __init__(self, data_dir: str = "data", persona_config: Optional[PersonaConfig] = None):
        self.data_dir = Path(data_dir)
        self.nsduh_data = None
        self.persona_config = persona_config or PersonaConfig()
        
        # Name databases by demographic groups
        self.names = {
            "White": {
                "male": ["John", "Michael", "David", "James", "Robert", "William", "Richard", "Thomas", "Christopher", "Daniel"],
                "female": ["Mary", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Nancy"]
            },
            "Black": {
                "male": ["James", "Michael", "William", "David", "Robert", "Christopher", "Anthony", "Mark", "Donald", "Steven"],
                "female": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
            },
            "Hispanic": {
                "male": ["Jose", "Luis", "Carlos", "Miguel", "Antonio", "Francisco", "Manuel", "David", "Daniel", "Rafael"],
                "female": ["Maria", "Carmen", "Ana", "Gloria", "Rosa", "Sandra", "Patricia", "Elena", "Isabel", "Monica"]
            },
            "Asian": {
                "male": ["Wei", "Chen", "Li", "Wang", "Zhang", "Liu", "Yang", "Huang", "Zhao", "Wu"],
                "female": ["Li", "Wang", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"]
            },
            "Native American": {
                "male": ["Tyler", "Cody", "Jake", "Brandon", "Dakota", "Hunter", "Blake", "Austin", "Logan", "Mason"],
                "female": ["Sierra", "Cheyenne", "Dakota", "Raven", "Willow", "Autumn", "Sky", "River", "Storm", "Luna"]
            },
            "Multiple": {
                "male": ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Sage", "River"],
                "female": ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Sage", "River"]
            }
        }
        
        # Location databases
        self.locations = {
            "urban": [
                "New York City", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
                "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
                "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis", "Seattle"
            ],
            "suburban": [
                "Plano, Texas", "Scottsdale, Arizona", "Frisco, Texas", "Cary, North Carolina",
                "Gilbert, Arizona", "McKinney, Texas", "Raleigh, North Carolina", "Henderson, Nevada",
                "Chandler, Arizona", "Madison, Wisconsin", "Rochester, Minnesota", "Overland Park, Kansas"
            ],
            "rural": [
                "Bismarck, North Dakota", "Fargo, North Dakota", "Rapid City, South Dakota",
                "Casper, Wyoming", "Billings, Montana", "Missoula, Montana", "Anchorage, Alaska",
                "Fairbanks, Alaska", "Juneau, Alaska", "Cheyenne, Wyoming", "Helena, Montana"
            ]
        }
        
        # Hobbies and interests by demographic patterns
        self.hobbies = {
            "young_adult": [
                "video gaming", "social media", "fitness", "music production", "photography",
                "travel", "cooking", "art", "dancing", "sports", "reading", "hiking"
            ],
            "adult": [
                "gardening", "cooking", "reading", "fitness", "photography", "travel",
                "wine tasting", "golf", "tennis", "hiking", "volunteering", "crafting"
            ],
            "older_adult": [
                "gardening", "reading", "volunteering", "bird watching", "bridge",
                "quilting", "woodworking", "fishing", "golf", "travel", "genealogy"
            ]
        }
        
        # Career patterns by education
        self.careers = {
            "≤ 8th grade": [
                "retail worker", "janitor", "construction worker", "factory worker",
                "landscaper", "custodian", "warehouse worker", "food service worker"
            ],
            "Some high school": [
                "retail manager", "truck driver", "mechanic", "security guard",
                "restaurant manager", "warehouse supervisor", "maintenance worker"
            ],
            "High school graduate": [
                "office administrator", "sales representative", "technician", "supervisor",
                "customer service", "bookkeeper", "receptionist", "retail manager"
            ],
            "Some college, no degree": [
                "assistant manager", "coordinator", "specialist", "analyst",
                "supervisor", "trainer", "representative", "technician"
            ],
            "Associate degree": [
                "registered nurse", "dental hygienist", "paralegal", "technician",
                "coordinator", "specialist", "supervisor", "analyst"
            ],
            "Bachelor's or higher": [
                "engineer", "teacher", "manager", "analyst", "consultant",
                "director", "coordinator", "specialist", "researcher", "executive"
            ]
        }
        
        # Cultural background details
        self.cultural_details = {
            "White": {
                "backgrounds": ["Irish-American", "German-American", "Italian-American", "Polish-American", "English-American"],
                "traditions": ["family gatherings", "holiday celebrations", "community events", "sports traditions"]
            },
            "Black": {
                "backgrounds": ["African-American", "Caribbean-American", "Nigerian-American", "Ethiopian-American"],
                "traditions": ["family reunions", "church community", "cultural festivals", "music traditions"]
            },
            "Hispanic": {
                "backgrounds": ["Mexican-American", "Puerto Rican-American", "Cuban-American", "Dominican-American"],
                "traditions": ["quinceañeras", "family celebrations", "cultural festivals", "food traditions"]
            },
            "Asian": {
                "backgrounds": ["Chinese-American", "Korean-American", "Japanese-American", "Vietnamese-American", "Filipino-American"],
                "traditions": ["lunar new year", "family honor", "educational focus", "cultural celebrations"]
            },
            "Native American": {
                "backgrounds": ["Cherokee", "Navajo", "Sioux", "Chippewa", "Choctaw"],
                "traditions": ["powwows", "tribal ceremonies", "cultural preservation", "community gatherings"]
            }
        }
        
        # Behavior descriptions
        self.behavior_descriptions = {
            "alc_monthly": "consuming alcohol in the past month",
            "cig_monthly": "smoking cigarettes in the past month", 
            "mj_monthly": "using marijuana in the past month",
            "coc_monthly": "using cocaine in the past month",
            "her_monthly": "using heroin in the past month",
            "alc_ever": "ever consuming alcohol",
            "cig_ever": "ever smoking cigarettes",
            "mj_ever": "ever using marijuana",
            "coc_ever": "ever using cocaine",
            "her_ever": "ever using heroin"
        }
        
        # Story templates
        self.story_templates = [
            "{name} is a {age_desc} {sex} living in {location} with {family_context}. {background_story} {hobby_context} {career_context} What do you think the likelihood of {pronoun} {behavior} is?",
            
            "Meet {name}, a {age_desc} {sex} from {origin_location} who now calls {current_location} home. {background_story} {hobby_context} {career_context} Given this profile, what's the probability that {pronoun} is {behavior}?",
            
            "{name} is a {age_desc} {sex} who grew up in {origin_location} and currently lives in {current_location}. {background_story} {hobby_context} {career_context} Based on this information, how likely is it that {pronoun} is {behavior}?",
            
            "Consider {name}, a {age_desc} {sex} residing in {location}. {background_story} {hobby_context} {career_context} What would you estimate as the likelihood of {pronoun} {behavior}?",
            
            "{name} is a {age_desc} {sex} living in {location} with {family_context}. Originally from {origin_location}, {pronoun} {background_story} {hobby_context} {career_context} What do you think the chances are that {pronoun} is {behavior}?"
        ]
    
    def load_nsduh_data(self):
        """Load the clean NSDUH data."""
        logger.info("Loading NSDUH data...")
        
        nsduh_path = self.data_dir / "clean" / "nsduh.parquet"
        if not nsduh_path.exists():
            raise FileNotFoundError(f"NSDUH data not found at {nsduh_path}")
        
        self.nsduh_data = pd.read_parquet(nsduh_path)
        logger.info(f"Loaded NSDUH data: {self.nsduh_data.shape[0]} samples, {self.nsduh_data.shape[1]} columns")
        
        return self.nsduh_data
    
    def get_age_description(self, age: str) -> str:
        """Convert age category to descriptive text."""
        age_descriptions = {
            "12–13 years": "young adolescent",
            "14–15 years": "adolescent", 
            "16–17 years": "older adolescent",
            "18–20 years": "young adult",
            "21–23 years": "emerging adult",
            "24–25 years": "young adult",
            "26–29 years": "young adult",
            "30–34 years": "adult",
            "35–49 years": "middle-aged",
            "50-64 years": "middle-aged",
            "65+ years": "older adult"
        }
        return age_descriptions.get(age, age)
    
    def get_hobby_category(self, age: str) -> str:
        """Get appropriate hobby category based on age."""
        if age in ["12–13 years", "14–15 years", "16–17 years", "18–20 years", "21–23 years"]:
            return "young_adult"
        elif age in ["24–25 years", "26–29 years", "30–34 years", "35–49 years"]:
            return "adult"
        else:
            return "older_adult"
    
    def generate_persona_details(self, row: pd.Series) -> Dict[str, str]:
        """Generate rich persona details for a given NSDUH record."""
        age = row['age']
        sex = row['sex']
        race = row['race']
        edu = row['edu']
        
        # Select name based on race and sex
        name = random.choice(self.names.get(race, self.names["White"])[sex])
        
        # Select location based on education (proxy for urban/rural)
        if edu in ["≤ 8th grade", "Some high school"]:
            location_type = "rural"
        elif edu in ["High school graduate", "Some college, no degree"]:
            location_type = "suburban"
        else:
            location_type = "urban"
        
        current_location = random.choice(self.locations[location_type])
        
        # Select origin location (different from current for diversity)
        origin_location = random.choice(self.locations[random.choice(list(self.locations.keys()))])
        
        # Select hobby based on age
        hobby_category = self.get_hobby_category(age)
        hobby = random.choice(self.hobbies[hobby_category])
        
        # Select career based on education (handle character encoding issues)
        if edu in self.careers:
            career = random.choice(self.careers[edu])
        else:
            # Fallback for character encoding issues
            if 'Bachelor' in edu:
                career = random.choice(self.careers["Bachelor's or higher"])
            else:
                career = "professional"  # Generic fallback
        
        # Select cultural background
        cultural_bg = random.choice(self.cultural_details.get(race, self.cultural_details["White"])["backgrounds"])
        tradition = random.choice(self.cultural_details.get(race, self.cultural_details["White"])["traditions"])
        
        # Generate family context
        if age in ["12–13 years", "14–15 years", "16–17 years"]:
            family_context = "their family"
        elif age in ["18–20 years", "21–23 years"]:
            family_context = random.choice(["their family", "roommates", "alone"])
        else:
            family_context = random.choice(["their family", "their spouse and children", "alone"])
        
        # Generate background story
        background_stories = [
            f"Originally from {origin_location}, {name} comes from a {cultural_bg} background and values {tradition}.",
            f"Born and raised in {origin_location}, {name} has a strong {cultural_bg} heritage and enjoys {tradition}.",
            f"With roots in {origin_location}, {name} identifies with {cultural_bg} culture and participates in {tradition}.",
            f"Hailing from {origin_location}, {name} carries on {cultural_bg} traditions and cherishes {tradition}."
        ]
        background_story = random.choice(background_stories)
        
        # Generate hobby context
        hobby_contexts = [
            f"In their free time, {name} enjoys {hobby} and finds it relaxing.",
            f"{name} has a passion for {hobby} and spends weekends pursuing this interest.",
            f"When not working, {name} likes to {hobby} as a way to unwind.",
            f"{name} is particularly interested in {hobby} and has been practicing for years."
        ]
        hobby_context = random.choice(hobby_contexts)
        
        # Generate career context
        career_contexts = [
            f"Professionally, {name} works as a {career} and takes pride in their work.",
            f"{name} has built a career as a {career} and is respected in their field.",
            f"Currently employed as a {career}, {name} enjoys the challenges of their job.",
            f"{name} works as a {career} and has been in this role for several years."
        ]
        career_context = random.choice(career_contexts)
        
        # Determine pronouns
        pronoun = "he" if sex == "male" else "she"
        
        return {
            "name": name,
            "age_desc": self.get_age_description(age),
            "sex": sex,
            "location": current_location,
            "origin_location": origin_location,
            "family_context": family_context,
            "background_story": background_story,
            "hobby_context": hobby_context,
            "career_context": career_context,
            "pronoun": pronoun,
            "hobby": hobby,
            "career": career,
            "cultural_background": cultural_bg,
            "tradition": tradition
        }
    
    def generate_persona_story(self, row: pd.Series, behavior_var: str) -> Dict[str, Any]:
        """Generate a persona-based story for a given NSDUH record."""
        persona_details = self.generate_persona_details(row)
        behavior = self.behavior_descriptions.get(behavior_var, behavior_var)
        
        # Select random story template
        template = random.choice(self.story_templates)
        
        # Generate the story
        story_text = template.format(
            name=persona_details["name"],
            age_desc=persona_details["age_desc"],
            sex=persona_details["sex"],
            location=persona_details["location"],
            current_location=persona_details["location"],
            origin_location=persona_details["origin_location"],
            family_context=persona_details["family_context"],
            background_story=persona_details["background_story"],
            hobby_context=persona_details["hobby_context"],
            career_context=persona_details["career_context"],
            pronoun=persona_details["pronoun"],
            behavior=behavior
        )
        
        return {
            "story_type": "persona_based",
            "story_text": story_text,
            "persona_details": persona_details,
            "nsduh_attributes": {
                "age": row['age'],
                "sex": row['sex'],
                "race": row['race'],
                "edu": row['edu']
            },
            "behavior": behavior_var,
            "behavior_value": row[behavior_var]
        }
    
    def create_persona_dataset(self, max_stories_per_combination: int = 5) -> List[Dict[str, Any]]:
        """Create the complete persona-based dataset."""
        logger.info("Creating persona-based dataset...")
        
        if self.nsduh_data is None:
            self.load_nsduh_data()
        
        stories = []
        
        # Get all behavior variables
        behavior_vars = [col for col in self.nsduh_data.columns if col.endswith('_ever') or col.endswith('_monthly')]
        
        for behavior_var in behavior_vars:
            logger.info(f"Generating persona stories for {behavior_var}...")
            
            # Get unique combinations of demographics
            demographic_cols = ['age', 'race', 'edu', 'sex']
            combinations = self.nsduh_data[demographic_cols].drop_duplicates()
            
            for _, combo in combinations.iterrows():
                # Filter data for this combination
                mask = True
                for col in demographic_cols:
                    mask &= (self.nsduh_data[col] == combo[col])
                subset = self.nsduh_data[mask]
                
                if len(subset) < 5:  # Skip very small groups
                    continue
                
                # Generate multiple persona stories for this demographic combination
                sample_size = min(max_stories_per_combination, len(subset))
                sample_data = subset.sample(n=sample_size, random_state=42)
                
                for _, individual in sample_data.iterrows():
                    try:
                        story = self.generate_persona_story(individual, behavior_var)
                        stories.append(story)
                    except Exception as e:
                        logger.warning(f"Error generating persona story for {behavior_var}: {e}")
                        continue
        
        logger.info(f"Generated {len(stories)} persona stories")
        return stories
    
    def create_conversation_format(self, stories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert persona stories to conversation format for chat models."""
        logger.info("Converting persona stories to conversation format...")
        
        conversations = []
        
        for story in stories:
            persona = story['persona_details']
            
            # Create conversation based on persona story
            system_prompt = "You are an expert epidemiologist and public health researcher who analyzes personal profiles to assess health behavior risks based on demographic and lifestyle factors."
            
            user_prompt = story['story_text']
            
            # Create assistant response based on NSDUH patterns
            behavior = story['behavior']
            behavior_value = story['behavior_value']
            
            if behavior_value == 'yes':
                likelihood_desc = "moderate to high"
                explanation = f"Based on the demographic profile of a {persona['age_desc']} {persona['sex']} with {story['nsduh_attributes']['edu']} education from a {story['nsduh_attributes']['race']} background, the likelihood of {behavior} is {likelihood_desc}. This aligns with epidemiological patterns observed in similar demographic groups."
            else:
                likelihood_desc = "low to moderate"
                explanation = f"Given the profile of a {persona['age_desc']} {persona['sex']} with {story['nsduh_attributes']['edu']} education from a {story['nsduh_attributes']['race']} background, the likelihood of {behavior} is {likelihood_desc}. This is consistent with lower prevalence rates observed in this demographic group."
            
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": explanation}
            ]
            
            # Convert to text format
            text = self._conversation_to_text(conversation)
            
            conversations.append({
                "text": text,
                "conversation": conversation,
                "story_data": story,
                "story_type": story['story_type']
            })
        
        logger.info(f"Created {len(conversations)} conversation examples")
        return conversations
    
    def create_instruction_format(self, stories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert persona stories to instruction tuning format."""
        logger.info("Converting persona stories to instruction format...")
        
        instructions = []
        
        for story in stories:
            persona = story['persona_details']
            
            # Create different instruction formats
            instruction_formats = [
                # Format 1: Direct analysis
                f"Analyze this personal profile and assess health behavior risk:\n\n{story['story_text']}\n\nProvide a detailed assessment based on demographic and lifestyle factors.",
                
                # Format 2: Research perspective
                f"As a public health researcher, evaluate this individual's risk profile:\n\n{story['story_text']}\n\nConsider demographic patterns, lifestyle factors, and epidemiological trends in your assessment.",
                
                # Format 3: Clinical perspective
                f"From a clinical epidemiology perspective, assess this person's health behavior likelihood:\n\n{story['story_text']}\n\nConsider age, education, cultural background, and other relevant factors.",
                
                # Format 4: Population health
                f"Evaluate this individual's health behavior risk within the context of population health patterns:\n\n{story['story_text']}\n\nConsider how this person fits within broader demographic and epidemiological trends."
            ]
            
            for instruction_text in instruction_formats:
                instructions.append({
                    "text": instruction_text,
                    "story_data": story,
                    "story_type": story['story_type'],
                    "instruction_type": "persona_analysis"
                })
        
        logger.info(f"Created {len(instructions)} instruction examples")
        return instructions
    
    def _conversation_to_text(self, conversation: List[Dict[str, str]]) -> str:
        """Convert conversation format to text."""
        text_parts = []
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            if role == 'system':
                text_parts.append(f"System: {content}")
            elif role == 'user':
                text_parts.append(f"Human: {content}")
            elif role == 'assistant':
                text_parts.append(f"Assistant: {content}")
        
        return "\n".join(text_parts)
    
    def save_dataset(self, data: List[Dict[str, Any]], output_path: str, format_type: str = "json") -> None:
        """Save the dataset to file."""
        logger.info(f"Saving dataset to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == "jsonl":
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Saved {len(data)} examples to {output_path}")
    
    def create_complete_dataset(self, output_dir: str = "nsduh_persona_data") -> Dict[str, str]:
        """Create the complete persona-based dataset."""
        logger.info("Creating complete persona-based dataset...")
        
        # Load data
        self.load_nsduh_data()
        
        # Generate persona stories
        stories = self.create_persona_dataset()
        
        # Create different formats
        conversations = self.create_conversation_format(stories)
        instructions = self.create_instruction_format(stories)
        
        # Combine all data
        all_data = stories + conversations + instructions
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save different formats
        saved_files = {}
        
        # Save raw stories
        stories_path = output_dir / "nsduh_persona_stories.json"
        self.save_dataset(stories, stories_path)
        saved_files["stories"] = str(stories_path)
        
        # Save conversations
        conversations_path = output_dir / "nsduh_persona_conversations.json"
        self.save_dataset(conversations, conversations_path)
        saved_files["conversations"] = str(conversations_path)
        
        # Save instructions
        instructions_path = output_dir / "nsduh_persona_instructions.json"
        self.save_dataset(instructions, instructions_path)
        saved_files["instructions"] = str(instructions_path)
        
        # Save combined data
        combined_path = output_dir / "nsduh_persona_combined.json"
        self.save_dataset(all_data, combined_path)
        saved_files["combined"] = str(combined_path)
        
        # Save JSONL format
        jsonl_path = output_dir / "nsduh_persona_data.jsonl"
        self.save_dataset(all_data, jsonl_path, "jsonl")
        saved_files["jsonl"] = str(jsonl_path)
        
        # Create summary
        summary = {
            "created_at": datetime.now().isoformat(),
            "total_stories": len(stories),
            "total_conversations": len(conversations),
            "total_instructions": len(instructions),
            "total_examples": len(all_data),
            "persona_config": {
                "use_realistic_names": self.persona_config.use_realistic_names,
                "use_detailed_locations": self.persona_config.use_detailed_locations,
                "use_personal_hobbies": self.persona_config.use_personal_hobbies,
                "use_family_context": self.persona_config.use_family_context,
                "use_career_details": self.persona_config.use_career_details,
                "use_cultural_background": self.persona_config.use_cultural_background
            },
            "files_created": saved_files
        }
        
        summary_path = output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created {len(all_data)} total training examples")
        logger.info(f"Files saved to: {output_dir}")
        
        return saved_files

def main():
    """Main function to create NSDUH persona dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create NSDUH persona-based finetuning dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="nsduh_persona_data", help="Output directory")
    parser.add_argument("--max_stories", type=int, default=5, help="Max stories per demographic combination")
    
    args = parser.parse_args()
    
    # Create persona generator
    generator = NSDUHPersonaGenerator(args.data_dir)
    
    # Create dataset
    saved_files = generator.create_complete_dataset(args.output_dir)
    
    print("\n" + "="*60)
    print("NSDUH PERSONA DATASET CREATION COMPLETED")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print("\nFiles created:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    print("="*60)

if __name__ == "__main__":
    main()
