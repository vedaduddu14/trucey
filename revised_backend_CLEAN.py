"""
SoCALM Trucey - Backend Logic (Cleaned)
=============================================

This file contains the core logic for workplace conversation assistance:
- Language analysis (empathy, formality, etc.)
- Leadership style detection
- Conversation management
- Data processing
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sqlite3
import threading
import time

from leadership_styles import leadership_styles

# ============================================================================
# 1. ENVIRONMENT AND CLIENT SETUP
# ============================================================================

def load_env():
    """Load environment variables from .env file"""
    load_dotenv()
    env_vars = {
        "api_key": os.getenv("OPENAI_API_KEY")
    }
    for key, value in env_vars.items():
        if not value:
            raise ValueError(f"{key} not found. Please set the {key} environment variable.")
    return env_vars

def get_openai_client():
    """Create OpenAI client using environment variables"""
    env = load_env()
    client = OpenAI(api_key=env["api_key"])
    return client

def ask_gpt(client, messages: list, temperature=0.7, max_tokens=150) -> str:
    """Send request to OpenAI API to generate response"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# ============================================================================
# 3. PARTICIPANT DATA MANAGEMENT
# ============================================================================

def authenticate_participant(prolific_id, last_name):
    """Authenticate participant with ID and password"""
    profile = assign_profile_to_prolific_id(prolific_id, last_name)
    if not profile:
        return None
    else:
        return {
        'participant_id': profile['LoginID'],
        'assigned_system': profile['AssignedSystem'],
        'assigned_problem': profile['AssignedProblem'], 
        'person_of_interest': profile['PersonofInterest'],
        'relationship_quality': profile['RelationshipQuality'],
        'relationship_length': profile['RelationshipLength'],
        'topic': profile['Topic'],
        'payment_type': profile['PaymentType'], 
        'has_topic_been_discussed': profile['PreviousInteraction'],
        'prolific_id': prolific_id,
        'last_name': last_name
    }

def convert_sqlite_to_info_format(participant_data: dict) -> dict:
    """Convert SQLite participant data to info format expected by the system"""
    payment_descriptions = {
        'salary': 'You receive a fixed annual salary',
        'hourly_wages': 'You are paid hourly wages'
    }
    return {
        "individual": {
            "id": "individual",
            "description": f"Your {participant_data['person_of_interest']}",
            "obtained": True,
            "explanation": f"Person you're talking to: {participant_data['person_of_interest']}",
            "value": participant_data['person_of_interest']
        },
        "topic": {
            "id": "topic", 
            "description": participant_data['assigned_problem'].replace('_', ' ').title(),
            "obtained": True,
            "explanation": f"Topic to discuss: {participant_data['assigned_problem'].replace('_', ' ')}",
            "value": participant_data['assigned_problem']
        },
        "previous_interaction": {
            "id": "previous_interaction",
            "description": "You have discussed this before" if participant_data['has_topic_been_discussed'] == 'yes' else "This is the first time bringing this up",
            "obtained": True,
            "explanation": f"Previous discussions: {participant_data['has_topic_been_discussed']}",
            "value": participant_data['has_topic_been_discussed']
        },
        "relationship": {
            "id": "relationship",
            "description": f"You have had a {participant_data['relationship_quality']} relationship for {participant_data['relationship_length'].replace('_', ' ')}",
            "obtained": True,
            "explanation": f"Relationship: {participant_data['relationship_quality']} for {participant_data['relationship_length'].replace('_', ' ')}",
            "quality": participant_data['relationship_quality'],
            "length": participant_data['relationship_length']
        },
        "work_context": {
            "id": "work_context",
            "description": payment_descriptions.get(
                participant_data['payment_type'], 
                f"You are a {participant_data['payment_type'].replace('_', ' ')} worker"
            ),
            "obtained": True,
            "explanation": f"Employment type: {payment_descriptions.get(participant_data['payment_type'], participant_data['payment_type'].replace('_', ' '))}",
            "value": participant_data['payment_type']
        },
        "leadership_style": {
            "id": "leadership_style",
            "description": "Automatically inferred from responses about the other person's behavior and leadership style",
            "obtained": False,
            "name": None,
            "traits": None,
            "pros": [],
            "cons": [],
            "confidence": None,
            "mcq_answers": [],
            "trait_vector": [],
            "para_input": None
        },
        "_source": participant_data
    }
# ============================================================================
# 4. SENTENCE STARTERS GENERATION
# ============================================================================

def generate_sentence_starters(client, last_assistant_message, current_phase, info, brett_element=None, brett_example=None):
    """Generate diverse sentence starters: supporting, opposing, and alternative-seeking"""
    
    topic = info['topic']['value'].replace('_', ' ')
    person = info['individual']['value'].replace('_', ' ')

    if current_phase == "advice":
        prompt = f"""
        The assistant just gave this advice: "{last_assistant_message}"
        
        User's situation: Discussing {topic} with their {person}
        Brett principle being used: {brett_element if brett_element else "general guidance"}
        
        ANALYZE specific details in the assistant's advice and generate targeted starters:
        
        Look for:
        - Specific strategies or approaches mentioned
        - Particular phrases or scripts suggested
        - Specific warnings or things to avoid
        - Details about timing, tone, or approach
        - Concrete examples or scenarios provided
        
        Generate starters that reference SPECIFIC elements from the advice:
        
        Instead of "that sounds good" → "The timing approach"
        Instead of "i'm worried" → "What if they"
        Instead of "what if instead" → "The phrase you"
        
        Generate 3 highly specific starters (MAXIMUM 4 words each) that reference concrete details from the assistant's advice:
        1. SUPPORTING: References specific advice positively
        2. CHALLENGING: Questions a specific aspect mentioned  
        3. ALTERNATIVE: Asks about specific alternatives or modifications
        
        Format: Return only the starters separated by |
        
        CRITICAL RULES:
        - MAXIMUM 4 words per starter
        - NO question marks or punctuation
        - Reference SPECIFIC details from the assistant's advice
        - Avoid generic phrases like "that sounds good", "i'm worried"
        - Make them actionable and advice-specific
        
        Examples for strategy advice: "That timing sounds|What if they|Could I say"
        Examples for phrase suggestions: "I like that|What if instead|How about saying"
        Examples for approach guidance: "The direct approach|But what if|Should I prepare"
        """
    elif current_phase == "control":
        prompt = f"""
                The assistant just said: "{last_assistant_message}"
                
                User's situation: Getting help with discussing {topic} with their {person}.
                
                ANALYZE specific details in the assistant's message and generate targeted starters:
                
                Look for:
                - Specific suggestions or recommendations mentioned
                - Particular steps or actions proposed  
                - Specific choices or options offered
                - Details about approach, timing, or format
                
                Generate starters that reference SPECIFIC elements from the message:
                
                Instead of "that makes sense" → "One page sounds"
                Instead of "i need advice" → "What talking points"  
                Instead of "let's practice" → "Let's practice the"
                
                Generate 3 highly specific starters (MAXIMUM 4 words each) that reference concrete details from the assistant's message:
                
                Format: Return only the starters separated by |
                
                CRITICAL RULES:
                - MAXIMUM 4 words per starter
                - NO question marks or punctuation
                - Reference SPECIFIC details from the assistant's message
                - Avoid generic phrases like "that makes sense", "tell me more"
                - Make them actionable and content-specific
                
                Examples for document advice: "One page sounds|Should I include|The summary document"
                Examples for practice offers: "Let's practice the|How should I|The talking points"
                Examples for preparation steps: "Should I prepare|The meeting approach|What metrics should"
                """
    elif current_phase == "rehearsal":
        prompt = f"""
        The assistant (role-playing as {person}) just said: "{last_assistant_message}"
        
        User's situation: Practicing conversation with their {person}
        
        ANALYZE specific details in what the boss just said and generate targeted responses:
        
        Look for:
        - Specific concerns or objections they raised
        - Particular questions they asked
        - Specific suggestions or alternatives they offered
        - Details about their reaction or mood
        - Concrete points they made about the request
        
        Generate responses that reference SPECIFIC elements from what the boss said:
        
        Instead of "I understand" → "About the timing"
        Instead of "actually I think" → "But the workload"
        Instead of "could you clarify" → "When you said"
        
        Generate 3 highly specific responses (MAXIMUM 4 words each) that address concrete details from the boss's response:
        1. AGREEABLE: Acknowledges specific points they made
        2. PUSHBACK: Addresses specific concerns they raised
        3. CLARIFYING: Asks about specific details they mentioned
        
        Format: Return only the starters separated by |
        
        CRITICAL RULES:
        - MAXIMUM 4 words per starter
        - NO question marks or punctuation
        - Reference SPECIFIC details from what the boss said
        - Avoid generic phrases like "I understand", "actually I think"
        - Make them realistic workplace responses
        
        Examples for concerns raised: "About the timing|But the deadline|When you mentioned"
        Examples for questions asked: "Yes the project|However my performance|What specifically about"
        Examples for suggestions made: "That approach could|But considering my|The alternative you"
        """
    else:
        # Handle unexpected phase values
        prompt = f"""
        Generate 3 general conversational starters (MAXIMUM 4 words each):
        Format: Return only the starters separated by |
        """
    
    try:
        response = ask_gpt(client, [{"role": "system", "content": prompt}], temperature=0.7, max_tokens=100)
        starters = response.strip().split('|')

        cleaned_starters = []
        for i, starter in enumerate(starters[:3]):
            clean = starter.strip().replace('?', '').replace('...', '').replace('"', '')
            if clean and len(clean.split()) <= 6:
                cleaned_starters.append(clean)
        return cleaned_starters[:3] if cleaned_starters else get_fallback_starters(current_phase)
        
    except Exception as e:
        fallback = get_fallback_starters(current_phase)
        return fallback

def get_fallback_starters(current_phase):
    """Provide diverse fallback starters if AI generation fails"""
    if current_phase == "advice":
        return [
            "That sounds good but",   
            "I'm worried that",       
            "What if instead"          
        ]
    else: 
        return [
            "I understand and",        
            "Actually I think",       
            "Could you help me"       
        ]

# ============================================================================
# 5. POWER DYNAMIC / LEADERSHIP STYLE ASSESSMENT
# ============================================================================

def parse_mcq_answers(mcq_answers):
    """Parse MCQ answers into trait vector"""
    if len(mcq_answers) != 5:
        raise ValueError("Expected 5 MCQ answers")
    
    O = mcq_answers[0]  
    E = mcq_answers[1]  
    A = (mcq_answers[2] + mcq_answers[4]) / 2  
    ES = mcq_answers[3]  

    return np.array([O, E, A, ES])

def get_power_dynamic_model(user_description: str = "", mcq_answers: list = None, para_weight=0.3) -> dict:
    """Leadership style matching using the leadership_styles data structure"""
    mcq_vector = None
    if mcq_answers is not None and len(mcq_answers) == 5:
        mcq_vector = parse_mcq_answers(mcq_answers)

    para_vector = None
    if user_description and user_description.strip():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        descriptions = []
        trait_vecs = []
        for style in leadership_styles:
            full_desc = f"{style['name']}: {style['traits']}"
            descriptions.append(full_desc)
            
            o, e, a, n = style["traits_vector"]
            es = 6 - n
            trait_vecs.append(np.array([o, e, a, es]))
        
        leadership_embeddings = model.encode(descriptions, convert_to_tensor=True)
        user_embedding = model.encode(user_description, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, leadership_embeddings)[0].cpu().numpy()
        para_vector = np.zeros(4)
        for i, score in enumerate(cosine_scores):
            para_vector += score * trait_vecs[i]
        para_vector /= np.sum(cosine_scores)

    if mcq_vector is not None and para_vector is not None:
        combo = 0.7 * mcq_vector + para_weight * para_vector
    elif mcq_vector is not None:
        combo = mcq_vector
    elif para_vector is not None:
        combo = para_vector
    else:
        return None
    best_similarity = -1
    best_style = None
    
    for style in leadership_styles:
        o, e, a, n = style["traits_vector"]
        es = 6 - n
        profile = np.array([o, e, a, es])
        
        similarity = np.dot(combo, profile) / (np.linalg.norm(combo) * np.linalg.norm(profile))
        if similarity > best_similarity:
            best_similarity = similarity
            best_style = style
    
    if best_style is None:
        return None
    
    return {
        "name": best_style["name"],
        "traits": best_style["traits"],
        "pros": best_style["pros"],
        "cons": best_style["cons"],
        "confidence": float(best_similarity),
        "mcq_answers": mcq_answers if mcq_answers is not None else [],
        "trait_vector": combo.tolist(),
        "para_input": user_description if user_description else ""
    }

def update_info_with_power_dynamic(info: dict, power_dynamic: dict) -> dict:
    """Update info dictionary with power dynamic results"""
    if power_dynamic:
        info["leadership_style"] = {
            "obtained": True,
            "name": power_dynamic["name"],
            "traits": power_dynamic["traits"],
            "pros": power_dynamic["pros"],
            "cons": power_dynamic["cons"],
            "confidence": power_dynamic["confidence"],
            "mcq_answers": power_dynamic.get("mcq_answers", []),
            "trait_vector": power_dynamic.get("trait_vector", []),
            "para_input": power_dynamic.get("para_input", "")
        }
    else:
        info["leadership_style"] = {
            "obtained": False,
            "explanation": "User chose to skip or didn't provide enough information to assess leadership style."
        }
    return info

# ============================================================================
# 6. SITUATION CATEGORIZATION
# ============================================================================

def categorize_situation(info_dict: dict) -> dict:
    """Directly categorize workplace situation based on assigned problem type"""
    
    # Get the assigned problem from the info dict
    assigned_problem = info_dict.get('topic', {}).get('value', '').lower()
    
    # Direct mapping from your CSV problem types to scenario categories
    problem_to_category_map = {
        'asking_for_raise': {
            'category': 'Promotion (salary increase, promotion, etc.)',
            'explanation': 'User is asking for a salary raise, which falls under promotion/compensation discussions.'
        },
        'asking_for_promotion': {
            'category': 'Promotion (salary increase, promotion, etc.)',
            'explanation': 'User is asking for a promotion, which is a career advancement discussion.'
        },
        'asking_for_time_off': {
            'category': 'Work-related problems',
            'explanation': 'User is requesting time off, which is a general workplace request/issue.'
        }
    }
    
    # Look up the category
    if assigned_problem in problem_to_category_map:
        result = problem_to_category_map[assigned_problem]
        print(f"DEBUG - Direct mapping: '{assigned_problem}' → '{result['category']}'")
        return result
    else:
        # Fallback for unexpected problem types
        print(f"DEBUG - Unknown problem type: '{assigned_problem}', defaulting to Work-related problems")
        return {
            'category': 'Work-related problems',
            'explanation': f'Unknown problem type "{assigned_problem}", defaulting to general work-related category.'
        }
# ============================================================================
# 7. SCENARIO DATA LOADING
# ============================================================================

def load_scenario_data(scenario_type: str, category: str, level: int = None) -> dict:
    """Load scenario data from files based on type and category"""
    category_map = {
        "Promotion": "promotion",
        "Sign-on": "sign_on",
        "Work-related problems": "job"
    }
    if category not in category_map:
        raise ValueError(f"Unexpected category: {category}")
    
    folder = category_map[category]
    combined = {"dialogue": []}
    scenario = scenario_type.lower()
    
    if scenario == "rehearsal":
        if level is None:
            raise ValueError("Rehearsal scenarios require a level (1–5).")
        levels = [level]
        print(f"   Rehearsal level: {level}")
    else:
        levels = [4, 5]
        print(f"   Advice levels: {levels}")
    
    files_loaded = 0
    total_dialogues = 0
    
    for lvl in levels:
        path = f"./dataset/{folder}_{scenario}/{folder}_{scenario}_scenario_level_{lvl}_response.txt"
        print(f"    Trying to load: {path}")
        
        if os.path.exists(path):
            print(f"    File exists!")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    dialogues_in_file = len(data.get("dialogue", []))
                    combined["dialogue"].extend(data.get("dialogue", []))
                    files_loaded += 1
                    total_dialogues += dialogues_in_file
                    print(f"   Loaded {dialogues_in_file} dialogue entries from level {lvl}")
            except Exception as e:
                print(f"    Error loading file: {e}")
        else:
            print(f"  File not found: {path}")
    
    if not combined["dialogue"]:
        print("ERROR: No valid scenario files were loaded!")
        raise FileNotFoundError("No valid scenario files were loaded.")
    
    return combined

def extract_advisor_traits(ideal_dialogue: dict) -> tuple:
    """Extract advisor traits and responses from dialogue data"""
    advisor_traits = []
    advisor_responses = []
    speaker_possibilities = ["manager", "advisor", "hr_manager"]

    for entry in ideal_dialogue.get("dialogue", []):
        if any(keyword in entry.get("speaker", "").lower() for keyword in speaker_possibilities):
            advisor_responses.append(entry["text"])
            for element in entry.get("brett_elements", []):
                advisor_traits.append({
                    "element": element,
                    "example": entry["text"],
                    "used": False
                })
    return advisor_traits, advisor_responses

def load_combined_scenario_data(category: str, rehearsal_level: int = None) -> dict:
    """Load scenario data based on category and rehearsal level"""
    print(f"\n LOADING COMBINED SCENARIO DATA:")
    print(f"   Category: {category}")
    print(f"   Rehearsal Level: {rehearsal_level}")
    print("="*50)
    
    print(" LOADING ADVICE DATA...")
    advice_data = load_scenario_data("advice", category)
    advice_traits, advice_responses = extract_advisor_traits(advice_data)
    print(f"   Advice traits extracted: {len(advice_traits)}")
    print(f"   Advice responses extracted: {len(advice_responses)}")
    
    print("\n LOADING REHEARSAL DATA...")
    rehearsal_data = load_scenario_data("rehearsal", category, rehearsal_level)
    rehearsal_traits, rehearsal_responses = extract_advisor_traits(rehearsal_data)
    print(f"   Rehearsal traits extracted: {len(rehearsal_traits)}")
    print(f"   Rehearsal responses extracted: {len(rehearsal_responses)}")
    
    print("\nSAMPLE ADVICE TRAITS:")
    for i, trait in enumerate(advice_traits[:3]):  # Show first 3
        print(f"   {i+1}. {trait.get('element', 'No element')}")
    
    print("\nSAMPLE REHEARSAL TRAITS:")
    for i, trait in enumerate(rehearsal_traits[:3]):  # Show first 3
        print(f"   {i+1}. {trait.get('element', 'No element')}")
    
    print("="*50)
    print("SCENARIO LOADING COMPLETE!\n")
    
    return {
        "advice_traits": advice_traits,
        "advice_responses": advice_responses, 
        "rehearsal_traits": rehearsal_traits,
        "rehearsal_responses": rehearsal_responses
    }

# ============================================================================
# 8. CONVERSATION TURN MANAGEMENT
# ============================================================================

def analyze_conversation_context(visible_messages: list, info: dict) -> dict:
    """Part 1: Analyze conversation context and prepare data"""
    turn_count = sum(1 for m in visible_messages if m['role'] == 'assistant')

    topic = info['topic']['value'].replace('_', ' ')
    individual = info['individual']['value']
    relationship_quality = info['relationship']['quality']
    relationship_length = info['relationship']['length'].replace('_', ' ')
    has_discussed = info['previous_interaction']['value']
    payment_type = info['work_context']['value'].replace('_', ' ')

    leadership_style = info.get("leadership_style", {})
    boss_style = leadership_style.get("name", "Unknown")
    boss_traits = leadership_style.get("traits", "Unknown")
    boss_pros = leadership_style.get("pros", [])
    boss_cons = leadership_style.get("cons", [])

    return {
        "turn_count": turn_count,
        "topic": topic,
        "individual": individual,
        "relationship_quality": relationship_quality,
        "relationship_length": relationship_length,
        "has_discussed": has_discussed,
        "payment_type": payment_type,
        "boss_style": boss_style,
        "boss_traits": boss_traits,
        "boss_pros": boss_pros,
        "boss_cons": boss_cons
    }

def build_conversation_prompt(context_data: dict, info: dict, advisor_traits: list, scenario_type: str, rehearsal_level: int, visible_messages: list) -> tuple:
    """Part 2: Build conversation prompts based on scenario type"""

    feedback_tone_block = construct_feedback_and_tone_prompt(info, advisor_traits, context_data["turn_count"])
    
    scenario_context = f"""
    You are participating in a workplace conversation scenario.
    The user is preparing for a discussion about: {context_data["topic"]}.
    The person they are talking to has a leadership style described as: {context_data["boss_style"]} ({context_data["boss_traits"]}).
    Key strengths: {context_data["boss_pros"]}
    Key challenges: {context_data["boss_cons"]}
    Their relationship: {context_data["relationship_quality"]} relationship for {context_data["relationship_length"]}
    

    The user is preparing for a conversation with their {context_data["individual"]} about {context_data["topic"]}.
    {feedback_tone_block}
    """

    unused = [e for e in advisor_traits if not e["used"]]
    if not unused:
        for e in advisor_traits: e["used"] = False
        unused = advisor_traits
    pick = random.choice(unused)
    pick["used"] = True
    if scenario_type == "rehearsal":
        instruction = build_rehearsal_instruction(context_data, pick, rehearsal_level, visible_messages)
    else:
        instruction = build_advice_instruction(context_data, pick, visible_messages)

    return scenario_context, instruction, pick

def build_rehearsal_instruction(context_data: dict, pick: dict, rehearsal_level: int, visible_messages: list) -> str:
    """Build rehearsal instruction prompt"""
    all_strengths = " | ".join(context_data["boss_pros"]) if context_data["boss_pros"] else "general leadership strengths"
    all_challenges = " | ".join(context_data["boss_cons"]) if context_data["boss_cons"] else "general leadership challenges"
    
    rehearsal_messages = [msg for msg in visible_messages if msg.get('phase') == 'rehearsal']
    recent_user_messages = [msg for msg in rehearsal_messages if msg.get('role') == 'user']
    last_user_message = recent_user_messages[-1]['content'] if recent_user_messages else ""
    rehearsal_assistant_count = len([msg for msg in visible_messages if msg.get('phase') == 'rehearsal' and msg.get('role') == 'assistant'])
    
    previous_assistant_responses = [msg["content"] for msg in rehearsal_messages if msg.get("role") == "assistant"]
    repetition_warning = ""
    if len(previous_assistant_responses) > 1:
        all_previous = " ".join(previous_assistant_responses).lower()
        repeated_topics = []
        if "lunch" in all_previous:
            repeated_topics.append("team lunch")
        if "celebrat" in all_previous:
            repeated_topics.append("celebration")
        if "meeting" in all_previous and "schedule" in all_previous:
            repeated_topics.append("scheduling meetings")
            
        if repeated_topics:
            repetition_warning = f"""
            AVOID REPETITION: You have already mentioned {', '.join(repeated_topics)} in previous responses.
            Do NOT repeat these topics. Find different ways to respond that fit your personality.
            """

    return f"""
    YOU ARE ROLE-PLAYING AS their {context_data["individual"]} in a conversation about {context_data["topic"]}
    
    CRITICAL RULES:
    - NEVER use trait jargon (Openness, Extraversion, etc.) - embody behaviors naturally
    - Keep response under 150 words
    - SHOW personality through specific behaviors, not generic boss responses
    - NEVER describe or explain your personality traits - just BE them
    - NEVER say things like "I tend to be impulsive" or "I avoid difficult conversations"
    - NEVER announce your psychological patterns - demonstrate them through actions
    - Respond as a real person who is unaware of their own personality quirks
    - Respond naturally, as if you are not aware of your own challenges
    - If they aren't responding as you expect (i.e gibberish or not making sense), ask them to clarify
    
    {repetition_warning} 

    YOUR PERSONALITY PROFILE: {context_data["boss_traits"]}
    YOUR BEHAVIORAL PATTERNS:
    - STRENGTHS YOU SHOW: {all_strengths}
    - CHALLENGES YOU HAVE: {all_challenges}
    
    REALISM LEVEL {rehearsal_level}/5 - SPECIFIC BEHAVIORAL REQUIREMENTS:
    
    Level 1-2 (Difficult): Your specific challenges DOMINATE your response:
    - EMBODY THESE EXACT CHALLENGES: {all_challenges}
    - Don't describe them - just naturally BE them in your response
    - Act out the behaviors that these challenges would cause
    - Be completely unaware that you're demonstrating these patterns
    - Each challenge should create specific behavioral problems in how you handle this request
    
    FORBIDDEN: Never say things like:
    - "I tend to be impulsive, so..."
    - "Sometimes I rely on charm..."
    - "I might avoid difficult conversations..."
    - "Don't take it personally if I seem..."
    
    REQUIRED: Instead, just naturally:
    - BE impulsive (change your mind mid-sentence)
    - USE charm (be enthusiastic but vague)  
    - AVOID difficulties (deflect to easier topics)
    - Just act naturally without self-commentary
    
    Level 3-4 (Mixed): Show BOTH strengths and challenges:
    - Start with {all_strengths} but let {all_challenges} create realistic friction
    - Be helpful but with personality-driven hesitations or complications
    
    Level 5 (Supportive): Lead with {all_strengths}:
    - Your positive traits dominate: {all_strengths}
    - Challenges barely show, you're at your best
    
    CONVERSATION CONTEXT:
    - Employee relationship: {context_data["relationship_quality"]} for {context_data["relationship_length"]}
    - Previous discussions: {context_data["has_discussed"]}
    - They are a: {context_data["payment_type"]}
    - Their message: "{last_user_message}"
    
    FAILURE MODE ACTIVATION:
    When people with {context_data["boss_traits"]} feel pressured, they specifically:
    - [Generate specific defensive behaviors based on their trait combination]
    - [Show how their challenges manifest under stress]
    If the user's request triggers these, SHOW these reactions authentically.
    
    RESPONSE APPROACH:
    {("Acknowledge they wanted to discuss " + context_data["topic"] + " then immediately show your personality challenges through natural behavior - don't describe them") if rehearsal_assistant_count == 0 else ("Respond to their message by naturally embodying your traits - be unconsciously difficult, don't explain why you're difficult")}
    
    EMBODY VS. DESCRIBE:
    WRONG: "I tend to be impulsive, so I might change my mind"
    RIGHT: "Yes! Absolutely! Wait... actually, let me think about that..."
    
    WRONG: "I rely on charm over substance, so I might be vague"  
    RIGHT: "Oh that's wonderful! I'm sure we can work something out somehow!"
    
    WRONG: "I avoid difficult conversations, so this is hard"
    RIGHT: "Speaking of time off, did you see the new project updates?"
    
    BRETT ELEMENT APPLICATION:
    Demonstrate "{pick['element']}" through the filter of your {context_data["boss_traits"]} and realism level.
    Template: "{pick['example']}" - adapt this to show how YOUR personality type would handle this principle.
    
    AUTHENTICITY CHECK:
    - Are you naturally demonstrating these specific challenges: {all_challenges}?
    - Are you acting unconsciously difficult rather than explaining why you're difficult?
    - Would a real person with these limitations respond this way?
    - Did you avoid describing your personality and just embody it instead?
    - At Level 1-2, are these challenges creating real problems WITHOUT you announcing them?
    
    FINAL RULE: Respond as someone who has these challenges but is completely unaware they have them. Just be naturally difficult in the ways your personality makes you difficult.
    
    WORD LIMIT: Under 150 words.
    """

def build_advice_instruction(context_data: dict, pick: dict, visible_messages: list) -> str:
    """Build advice instruction prompt"""
    all_strengths = " | ".join(context_data["boss_pros"]) if context_data["boss_pros"] else "general leadership strengths" 
    all_challenges = " | ".join(context_data["boss_cons"]) if context_data["boss_cons"] else "general leadership challenges"
    advice_messages = [msg for msg in visible_messages if msg.get('phase', 'advice') == 'advice']
    recent_advice_exchanges = advice_messages[-4:] if len(advice_messages) >= 4 else advice_messages
    
    conversation_memory = ""
    if len(recent_advice_exchanges) > 2:
        conversation_memory = "\n\nRECENT CONVERSATION:\n"
        for msg in recent_advice_exchanges[-4:]:
            role = "You" if msg['role'] == 'assistant' else "User"
            conversation_memory += f"{role}: {msg['content'][:100]}...\n"

    recent_user_messages = [msg for msg in advice_messages if msg.get('role') == 'user']
    last_user_message = recent_user_messages[-1]['content'].lower() if recent_user_messages else ""
    
    satisfaction_signals = ["thanks", "that's good", "no thanks", "that helps", "perfect", 
                        "sounds good", "i'm good", "that's enough", "got it"]
    is_satisfied = any(signal in last_user_message for signal in satisfaction_signals)


    is_first_advice = len([msg for msg in advice_messages if msg.get('role') == 'assistant']) == 0
    
    if is_first_advice:
        return f"""
        The user wants advice for discussing {context_data["topic"]} with their {context_data["individual"]} who has these specific traits: {context_data["boss_traits"]}
        
        CRITICAL RULES:
        - NEVER mention trait names (Openness, Extraversion, etc.) - translate into behavioral language
        - Keep response under 200 words total
        - END with a question to encourage dialogue
        - Be specific to this exact trait combination
        - If they aren't responding as you expect (i.e gibberish or not making sense), ask them to clarify
        - Respond naturally to the the user's {last_user_message}
        
        PERSONALITY-BASED ANALYSIS:
        
        STEP 1 - TRANSLATE TRAITS TO BEHAVIOR:
        Based on {context_data["boss_traits"]}, describe their behavior patterns without using jargon:
        - How do they naturally communicate and make decisions?
        - What energizes vs. drains them in conversations?
        
        STEP 2 - PREDICT FAILURE MODES:
        Explain exactly what goes wrong when people with {context_data["boss_traits"]} feel pressured or uncomfortable:
        - What defensive behaviors emerge?
        - How do they shut down or become resistant?
        
        STEP 3 - DESIGN SPECIFIC STRATEGY:
        Create advice that prevents these failure modes and leverages their natural preferences.
        
        FORMAT:
        - ACKNOWLEDGMENT: "I understand you need to discuss [topic] with your [boss type]..." (no trait jargon)
        - STRATEGY (2 sentences max): Specific approach based on their behavioral patterns
        - EXAMPLE PHRASE: One exact phrase tailored to their communication style  
        - WARNING (1 sentence): What specifically to avoid
        - CONVERSATION STARTER: End with a question like "How does this approach feel to you?" or "What concerns do you have about this strategy?"
        
        BRETT ELEMENT APPLICATION:
        Apply "{pick['element']}" using this template: "{pick['example']}"
        Adapt the template to their specific situation and boss's traits: {context_data["boss_traits"]}
        
        WORD LIMIT: Under 200 words total.
        """
        
    elif is_satisfied:
        return f"""
        USER SAID: "{last_user_message[:100]}"
        
        They seem satisfied. Give a brief, supportive response.
        
        RULES:
        - NO trait jargon (Openness, Extraversion, etc.)
        - Under 50 words
        - Reference their boss's behavioral style naturally
        
        FORMAT:
        - ACKNOWLEDGMENT: Recognize their satisfaction 
        - CLOSING: Brief supportive response + offer continued help
        
        BRETT ELEMENT APPLICATION:
        Subtly incorporate "{pick['element']}" using this template: "{pick['example']}"
        
        Based on {context_data["boss_traits"]}, acknowledge their readiness using natural language about their boss's style.
        """
        
    else:
        return f"""
        USER'S LATEST MESSAGE: "{last_user_message[:150]}"
        
        CONVERSATION CONTEXT: {conversation_memory}
        
        BOSS'S TRAITS: {context_data["boss_traits"]}
        
        CRITICAL RULES:
        - NEVER use trait jargon (Openness, Extraversion, etc.)
        - Under 150 words total
        - Address their specific concern
        - END with a question to encourage dialogue
        - If they aren't responding as you expect (i.e gibberish or not making sense), ask them to clarify
        
        FAILURE MODE ANALYSIS:
        Explain exactly what goes wrong when people with {context_data["boss_traits"]} feel pressured:
        - What defensive behaviors do they show?
        - How do they resist or shut down?
        
        RESPONSE PATTERN:
        - ACKNOWLEDGMENT: Recognize what they just said specifically
        - TARGETED ADVICE (1-2 sentences): Address their concern using behavioral language
        - SPECIFIC EXAMPLE: One concrete phrase or approach
        - CONVERSATION STARTER: End with a question to keep dialogue going
        
        BRETT ELEMENT APPLICATION:
        Apply "{pick['element']}" using this template: "{pick['example']}"
        Adapt the template to address their specific concern and boss's traits: {context_data["boss_traits"]}
        
        Translate their boss's {context_data["boss_traits"]} into natural behavioral descriptions.
        """

def generate_assistant_response(client, visible_messages: list, scenario_context: str, instruction: str, pick: dict, context_data: dict, scenario_type: str) -> tuple:
    """Part 3: Generate and return assistant response"""
    if scenario_type == "rehearsal":
        rehearsal_assistant_count = len([msg for msg in visible_messages if msg.get('phase') == 'rehearsal' and msg.get('role') == 'assistant'])
        print(f"   Rehearsal turn: {rehearsal_assistant_count + 1}")

    context = [m.copy() for m in visible_messages]
    context.append({"role": "system", "content": scenario_context + "\n" + instruction})

    temperature = 0.9 if scenario_type == "rehearsal" else 0.7
    reply = ask_gpt(client, context, temperature, max_tokens=250).strip()
    visible_messages.append({
        "role": "assistant", 
        "content": reply, 
        "brett_element": pick["element"], 
        "brett_example": pick["example"]
    })
    
    return reply, context_data["turn_count"] + 1

def get_next_assistant_turn(client, visible_messages: list, info: dict, advisor_traits: list, scenario_type: str, category: str, turn_count: int, rehearsal_level: int = None) -> Tuple[str, int]:
    """Main function: Generate next assistant turn using the three-part approach"""
    
    context_data = analyze_conversation_context(visible_messages, info)
    
    scenario_context, instruction, pick = build_conversation_prompt(
        context_data, info, advisor_traits, scenario_type, rehearsal_level, visible_messages
    )
    return generate_assistant_response(
        client, visible_messages, scenario_context, instruction, pick, context_data, scenario_type
    )

def summarize_user_feedback(feedback_list, client, info, advisor_traits, turn_count):
    """Summarize user feedback for assistant adaptation"""
    if not feedback_list:
        return "No user feedback to summarize."
    
    latest_feedback = feedback_list[-1]
    leadership = info.get("leadership_style", {})
    boss_name = leadership.get("name", "Unknown")
    boss_traits = leadership.get("traits", "Unknown")
    boss_pros = ", ".join(leadership.get("pros", []))
    boss_cons = ", ".join(leadership.get("cons", []))
    tone = info.get("language_style", {})
    empathy = tone.get("empathy", "unknown")
    formality = tone.get("formality", "unknown")
    persuasiveness = tone.get("persuasiveness", "unknown")
    politeness = tone.get("politeness", "unknown")
    current_trait = advisor_traits[turn_count % len(advisor_traits)].get("element", "neutral_tone")

    system_prompt = f"""
    You are an AI assisting another AI in adapting its tone and behavior for a workplace rehearsal scenario.

    The user just gave feedback about a simulated conversation with their boss.
    You must summarize how the assistant should adapt its tone or realism based on:
    - The feedback itself
    - The boss's leadership style
    - The user's tone
    - The Brett element in use

    Do not simply restate the feedback. Instead, extract actionable insight:
    - What tone or behavior should be dialed up or down?
    - Should the assistant be warmer, more vague, more decisive, less formal, etc.?
    - Does this feedback contradict the leadership style? Should it be partially ignored?
    - How heavily should this be weighted in the next response?

    Boss's leadership style: {boss_name} ({boss_traits})
    Strengths: {boss_pros}
    Challenges: {boss_cons}

    User's language style:
    - Empathy: {empathy}
    - Formality: {formality}
    - Persuasiveness: {persuasiveness}
    - Politeness: {politeness}

    Brett element currently in use: {current_trait}

    Latest user feedback:
    \"\"\"{latest_feedback.strip()}\"\"\"

    Respond with 2–3 sentences of advice for the assistant. Be realistic, not overly obedient.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    response = ask_gpt(client, messages, temperature=0.3, max_tokens=200)
    return response.strip() if response else "No feedback summary available."

def construct_feedback_and_tone_prompt(info: dict, traits: list, turn_count: int) -> str:
    """Construct feedback and tone guidance prompt"""
    feedback_section = ""
    if "latest_feedback_summary" in info and info["latest_feedback_summary"]:
        feedback_section = f"""
            The user provided the following feedback summary for the assistant:
            \"\"\"{info["latest_feedback_summary"].strip()}\"\"\"
            Use this as guidance for your tone, realism, and approach in this turn.
        """

    tone_summary = ""
    if "language_style" in info and isinstance(info["language_style"], dict):
        style = info["language_style"]
        tone_summary = f"""
            The user currently appears to be:
            - Politeness: {style.get('politeness', 'unknown')}
            - Formality: {style.get('formality', 'unknown')}
            - Persuasiveness: {style.get('persuasiveness', 'unknown')}
            - Empathy: {style.get('empathy', 'unknown')}

            Adjust your tone to match or strategically counterbalance this style based on the scenario's power dynamics.
        """

    if traits:
        current_trait = traits[turn_count % len(traits)].get("element", "neutral_tone")
    else:
        current_trait = "neutral_tone"
    trait_note = f"\nFocus this turn on the trait: **{current_trait}**\n"

    if not (feedback_section or tone_summary):
        return ""
    return feedback_section + tone_summary + trait_note

# ============================================================================
# 9. GENERAL CONVERSATION MANAGEMENT
# ============================================================================

def general_next_assistant_turn(client, visible_messages: list, info: dict) -> str:
    """Simple turn-by-turn conversation with built-in system prompt"""

    if info:
        topic = info.get('topic', {}).get('value', 'workplace issue').replace('_', ' ')
        person = info.get('individual', {}).get('value', 'colleague')
        relationship_desc = info.get('relationship', {}).get('description', 'unknown relationship')
        previous_interaction = info.get('previous_interaction', {}).get('description', 'unknown discussion history')
        work_context = info.get('work_context', {}).get('description', 'unknown work context')
        
        system_prompt = f"""You are a helpful workplace conversation assistant helping a user with a specific situation. Let each response not be more than 400 words long.

            CONTEXT:
            - The user needs help discussing {topic} with their {person}
            - Their relationship: {relationship_desc}
            - Previous discussions: {previous_interaction}
            - Work context: {work_context}

            APPROACH:
            - Use this context to provide targeted, relevant advice
            - Ask clarifying questions that build on what you already know
            - Offer both strategic advice and roleplay practice options
            - Be professional but conversational
            - Help them prepare for this specific conversation with their {person}

            Start by acknowledging their situation and offering to help them navigate this conversation about {topic}."""
    context = [{"role": "system", "content": system_prompt}]
    
    for message in visible_messages:
        if message.get("role") in ["user", "assistant"]:
            context.append({
                "role": message["role"], 
                "content": message["content"]
            })
    reply = ask_gpt(client, context, temperature=0.7, max_tokens=500).strip()
    
    visible_messages.append({"role": "assistant", "content": reply})
    
    return reply

# ============================================================================
# 10. DATA SAVING
# ============================================================================

def save_chat_locally(info, category, messages, advisor_traits, scenario_type, rehearsal_level=None, folder="saved_chats", feedback_log=None, prolific_id=None, **kwargs):
    """Save chat data locally to JSON file with Prolific ID in filename"""
    os.makedirs(folder, exist_ok=True)
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "scenario_type": scenario_type,
        "rehearsal_level": rehearsal_level,
        "prolific_id": prolific_id,  # Include in data too
        "info": info,
        "category": category,
        "advisor_traits": advisor_traits,
        "messages": messages,
        "feedback_log": feedback_log or [],
    }
    export_data.update(kwargs)
    
    # FIXED: Include prolific_id in filename
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if prolific_id and prolific_id != "unknown_user":
        filename = os.path.join(folder, f"chat_{prolific_id}_{timestamp}.json")
    else:
        filename = os.path.join(folder, f"chat_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"[Saved] Chat written to: {filename}")
    return filename

def assign_profile_to_prolific_id(prolific_id, last_name):
    conn = sqlite3.connect('./profiles.db', timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Check if this prolific_id + last_name combination is already assigned
        existing = conn.execute("""
            SELECT LoginID, AssignedSystem, AssignedProblem, PersonofInterest, Topic, 
                   RelationshipQuality, RelationshipLength, PaymentType, PreviousInteraction, LastName
            FROM profiles WHERE prolific_id = ? AND LastName = ?
        """, (prolific_id, last_name)).fetchone()
        
        if existing:
            return dict(existing)
        
        conn.execute("BEGIN IMMEDIATE")
        
        # Update available profile with both prolific_id and last_name
        cursor = conn.execute("""
            UPDATE profiles 
            SET prolific_id = ?, LastName = ?, status = 'assigned' 
            WHERE LoginID = (
                SELECT LoginID FROM profiles 
                WHERE status = 'available' 
                ORDER BY RANDOM() LIMIT 1
            )
        """, (prolific_id, last_name))
        
        if cursor.rowcount == 0:
            conn.rollback()
            return None
        
        # Retrieve the assigned profile
        assigned = conn.execute("""
            SELECT LoginID, AssignedSystem, AssignedProblem, PersonofInterest, Topic, 
                   RelationshipQuality, RelationshipLength, prolific_id, PaymentType, 
                   PreviousInteraction, LastName
            FROM profiles WHERE prolific_id = ? AND LastName = ?
        """, (prolific_id, last_name)).fetchone()
        
        conn.commit()  
        
        return {
            'ParticipantID': assigned['LoginID'],
            'AssignedSystem': assigned['AssignedSystem'],
            'AssignedProblem': assigned['AssignedProblem'],
            'PersonofInterest': assigned['PersonofInterest'],
            'Topic': assigned['Topic'],
            'RelationshipQuality': assigned['RelationshipQuality'],
            'RelationshipLength': assigned['RelationshipLength'],
            'PaymentType': assigned['PaymentType'],
            'PreviousInteraction': assigned['PreviousInteraction'],
            'LastName': assigned['LastName'],
            'prolific_id': prolific_id
        }
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()
    
