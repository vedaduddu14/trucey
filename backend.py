import os
import json

from dotenv import load_dotenv
from openai import AzureOpenAI




from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Tuple
from datetime import datetime
import random

# Import your leadership styles with trait vectors
from leadership_styles import leadership_styles

###################### 1.  Loading Environment ###########################

def load_env():
    """
    This function loads the environment variables from the .env file.
    It retrieves the API key, API version, Azure endpoint, and Azure deployment from the environment variables.
    If any of these variables are not set, it raises a ValueError with a message indicating which variable is missing.
    Args:
        None
    Returns:
        dict: A dictionary containing the API key, API version, Azure endpoint, and Azure deployment.
    """
    load_dotenv()
    env_vars = {
        "api_key": os.getenv("AZURE_OPENAI_NORTH_KEY"),
        "api_version": "2024-08-01-preview",
        "azure_endpoint": os.getenv("AZURE_OPENAI_NORTH_ENDPOINT"),
        "azure_deployment": os.getenv("AZURE_DEPLOYMENT")
    }
    for key, value in env_vars.items():
        if not value:
            raise ValueError(f"{key} not found. Please set the {key} environment variable.")
    return env_vars

def get_openai_client():
    """
    This function creates a new AzureOpenAI client using the environment variables loaded from the .env file.
    It retrieves the API key, API version, Azure endpoint, and Azure deployment from the environment variables.
    Args:
        None
    Returns:
        AzureOpenAI: An instance of the AzureOpenAI client initialized with the provided API key, API version, Azure endpoint, and Azure deployment.
    """
    env = load_env()
    client = AzureOpenAI(
        api_key=env["api_key"],
        api_version=env["api_version"],
        azure_endpoint=env["azure_endpoint"],
        azure_deployment=env["azure_deployment"]
    )
    return client

################## 2. Language Style Analyzer ######################
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class LanguageStyleAnalyzer:
    """
    This is a the LanguageSyleAnalyzer class that loads various language style models such as empathy, formality, persuasiveness, and politeness.
    """
    def __init__(self, load_empathy=True, load_formality=True, load_persuasiveness=True, load_politeness=True):
        self.load_empathy = load_empathy
        self.load_formality = load_formality
        self.load_persuasiveness = load_persuasiveness
        self.load_politeness = load_politeness

        if self.load_empathy:
            self.empathy_model, self.empathy_tokenizer = self._load_model_tokenizer("paragon-analytics/bert_empathy")
        
        if self.load_formality:
            self.formality_model, self.formality_tokenizer = self._load_model_tokenizer("s-nlp/roberta-base-formality-ranker")
        
        if self.load_persuasiveness:
            self.persuasiveness_model, self.persuasiveness_tokenizer = self._load_model_tokenizer("LACAI/roberta-large-PFG-donation-detection")
        
        if self.load_politeness:
            self.politeness_model, self.politeness_tokenizer = self._load_model_tokenizer("Genius1237/xlm-roberta-large-tydip")

    def _load_model_tokenizer(self, model_name):
        """
        This is a self helper function that loads the model and tokenizer for the given model name.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer

    def analyze_empathy(self, text):
        """
        This function analyzes the empathy score paragon-analytics/bert_empathy model.
        ** means unpacking the dictionary into keyword arguments
        taking softmax of the logits, i.e doing probability in the exponential space
        Empathy score is the probability of the positive class (1), which is the second element in the scores tensor (Wasn't labeled on the model card, but tested)
        Args:
            text (str): The input text to analyze for empathy.
        Returns: 
            float: The empathy score between 0 and 1, where 0 means no empathy and 1 means high empathy.
        """
        if not self.load_empathy:
            raise ValueError("Empathy model not loaded.")
        
        tokenized = self.empathy_tokenizer(text, return_tensors="pt")
        output = self.empathy_model(**tokenized)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        return scores[0][1].item()

    def analyze_formality(self, text):
        """
        This function analyzes the formality score using the s-nlp/roberta-base-formality-ranker model.
        ** means unpacking the dictionary into keyword arguments
        taking softmax of the logits, i.e doing probability in the exponential space
        Formality score is the probability of the positive class (1), which is the second element in the scores tensor (Wasn't labeled on the model card, but tested)
        Args:
            text (str): The input text to analyze for empathy.
        Returns: 
            float: The formality score between 0 and 1, where 0 means informal and 1 means formal.
        """
        if not self.load_formality:
            raise ValueError("Formality model not loaded.")
        
        tokenized = self.formality_tokenizer(text, return_tensors="pt")
        output = self.formality_model(**tokenized)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        return scores[0][1].item()

    def analyze_persuasiveness(self, text):
        """
        This function analyzes the formality score using the LACAI/roberta-large-PFG-donation-detection model.
        ** means unpacking the dictionary into keyword arguments
        taking softmax of the logits, i.e doing probability in the exponential space
        Persuasiveness score is the probability of the positive class (0), which is the first element in the scores tensor (Wasn't labeled on the model card, but tested)
        Args:
            text (str): The input text to analyze for empathy.
        Returns: 
            float: The persuasiveness score between 0 and 1, where 0 means persuasive and 1 means persuasive.
        """
        if not self.load_persuasiveness:
            raise ValueError("Persuasiveness model not loaded.")
        
        tokenized = self.persuasiveness_tokenizer(text, return_tensors="pt")
        output = self.persuasiveness_model(**tokenized)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        return scores[0][0].item()

    def analyze_politeness(self, text):
        """
        This function analyzes the politeness score using the Genius1237/xlm-roberta-large-tydip model.
        ** means unpacking the dictionary into keyword arguments
        taking softmax of the logits, i.e doing probability in the exponential space
        Politeness score is the probability of the positive class (0), which is the first element in the scores tensor (Wasn't labeled on the model card, but tested)
        Args:
            text (str): The input text to analyze for empathy.
        Returns: 
            float: The politeness score between 0 and 1, where 0 means not persuasive and 1 means persuasive.
        """
        if not self.load_politeness:
            raise ValueError("Politeness model not loaded.")
        
        tokenized = self.politeness_tokenizer(text, return_tensors="pt")
        output = self.politeness_model(**tokenized)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        return scores[0][1].item()

    def analyze_text(self, text):
        """
        This function analyzes the text and stores the results in a dictionary
        Args:
            text (str): The input text to analyze for empathy, formality, persuasiveness, and politeness.
        Returns:
            dict: A dictionary containing the scores for empathy, formality, persuasiveness, and politeness.
        """
        results = {}
        if self.load_empathy:
            results["empathy"] = self.analyze_empathy(text)
        if self.load_formality:
            results["formality"] = self.analyze_formality(text)
        if self.load_persuasiveness:
            results["persuasiveness"] = self.analyze_persuasiveness(text)
        if self.load_politeness:
            results["politeness"] = self.analyze_politeness(text)
        return results

########################3. Schema for Required Information (Step 1)########################

def required_information():
    """
    This function defines a dictionary that is returned everytime it is called
    It contains the required information needed to understand the user's situation that can guide the system to understand the situation
    and provide appropriate responses.
    Returns:
        dict: A dictionary containing the required information with keys such as 'individual', 'topic', 'previous_interaction', 'relationship', and 'language_style'.
    """
    return {
        "individual": {
            "id": "individual",
            "description": "Who is the person you're talking to? For example: boss, manager, HR, teammate etc.",
            "obtained": False,
            "explanation": None
        },
        "topic": {
            "id": "topic",
            "description": "What is the topic of the conversation? For example: discussing a promotion, asking for a raise, struggling within the team.",
            "obtained": False,
            "explanation": None
        },
        "previous_interaction": {
            "id": "previous_interaction",
            "description": "Have you ever had a conversation with this person before? For example: strangers or you have some relationship already.",
            "obtained": False,
            "explanation": None
        },
        "relationship": {
            "id": "relationship",
            "description": "What is your relationship with them? Ask them if they are nice or a strict manager, HR, etc.",
            "obtained": False,
            "explanation": None
        },
        "language_style": {
            "id": "language_style",
            "description": "Automatically analyzed based on the text provided by the user. This includes empathy, formality, persuasiveness, and politeness.",
            "obtained": True,
            "empathy": 0.0,
            "formality": 0.0,
            "persuasiveness": 0.0,
            "politeness": 0.0,
            "explanation": "Language style scores range from 0 to 1."
        },
        "leadership_style": {  # NEW FIELD!
            "id": "leadership_style",
            "description": "Automatically inferred from your responses about the other person's behavior and leadership style.",
            "obtained": False,
            "name": None,
            "traits": None,
            "pros": [],
            "cons": [],
            "confidence": None,
            "mcq_answers": [],
            "trait_vector": [],
            "para_input": None
        }
    }

########################4. Helper Function to Ask GPT ########################
def ask_gpt(client: AzureOpenAI, messages: list, temperature=0.7, max_tokens=150) -> str:
    """
    This function sends a request to the OpenAI API to generate a response based on the provided messages.
    Args:
        client (AzureOpenAI): The OpenAI client instance
        messages (list): The conversation history
        temperature (float): The temperature for the response generation, default is 0.7
        max_tokens (int): The maximum number of tokens for the response, default is 150
    Returns:
        str: The generated response from the OpenAI API.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

########################5. Stage 1 Info collection ########################
def analyze_and_update_info(client: AzureOpenAI, messages: list, info: dict) -> (dict, str):
    """
    This function gathers initial information from the user about their situation.
    It is a turn by turn conversation between the user and the LLM where the user is being asked details such as the topic at hand, the individual they are talking to etc.
    It probes the user until all the required information is obtained, i.e all dictionary values are True.

    Args:
        client (AzureOpenAI): The OpenAI client instance.
        messages (list): The conversation history, including user and assistant messages.
        info (dict): The dictionary containing the required information schema.
    
    Returns:
        tuple: A tuple containing the updated info dictionary and a follow-up question or empty string if
    """
    schema_prompt = f"""
    You are a JSON-only parser.  Analyze the last user message and update this 'info' object.
    Return *only* valid JSON, updating each field’s 'obtained' (true/false) and 'explanation' (a short string).

    Current schema:
    {json.dumps(info, indent=2)}

    Last messages (full Step 1 history; no Step 2 messages should be here yet):
    {json.dumps(messages, indent=2)}
    """
    messages.append({"role": "system", "content": schema_prompt})
    gpt_output = ask_gpt(client, messages, temperature=0.1, max_tokens=600)
    messages.pop()  # remove the schema prompt

    try:
        updated = json.loads(gpt_output)
    except json.JSONDecodeError:
        return info, "Sorry, I didn’t understand. Could you clarify?"

    for key in info:
        if key != "language_style" and key in updated:
            info[key]["obtained"] = updated[key]["obtained"]
            info[key]["explanation"] = updated[key].get("explanation", None)

    for key, val in info.items():
        if key != "language_style" and not val["obtained"]:
            followup_prompt = f"""
            Below is the exact wording of the missing “description” we need:
                "{val["description"]}"
            Ask a natural, helpful question to better understand this: {val['description']}.
            Avoid repeating the description directly. Do NOT introduce any new topics—stay focused on exactly this missing detail. Acknowledge their answers and then ask them the next question.
            Keep it under 50 words.
            """
            messages.append({"role": "system", "content": followup_prompt})
            followup_resp = ask_gpt(client, messages, temperature=0.7, max_tokens=100)
            messages.pop()
            return info, followup_resp

    return info, ""

####### 6. Power Dynamic Calibration (Step 2) #####################

def parse_mcq_answers(mcq_answers):
    O, E, A, N = 0, 0, 0, 0
    # Q1: Openness
    if mcq_answers[0] == "A": O = 5
    elif mcq_answers[0] == "B": O = 3
    elif mcq_answers[0] == "C": O = 1
    # Q2: Extraversion
    if mcq_answers[1] == "A": E = 5
    elif mcq_answers[1] == "B": E = 3
    elif mcq_answers[1] == "C": E = 1
    # Q3: Agreeableness
    if mcq_answers[2] == "A": A = 5
    elif mcq_answers[2] == "B": A = 3
    elif mcq_answers[2] == "C": A = 1
    # Q4: Neuroticism (+Openness)
    if mcq_answers[3] == "A": N = 1; O = min(5, O + 1)
    elif mcq_answers[3] == "B": N = 3
    elif mcq_answers[3] == "C": N = 5; O = max(1, O - 1)
    # Q5: Neuroticism (+Agreeableness)
    if mcq_answers[4] == "A": N = min(5, N + 0); A = min(5, A + 1)
    elif mcq_answers[4] == "B": N = min(5, N + 2)
    elif mcq_answers[4] == "C": N = min(5, N + 4); A = max(1, A - 1)
    return np.array([O, E, A, N])

def get_power_dynamic_model(user_description: str = "", mcq_answers: list = None, para_weight=0.3) -> dict:
    mcq_vector = None
    if mcq_answers is not None and len(mcq_answers) == 5:
        mcq_vector = parse_mcq_answers(mcq_answers)

    para_vector = None
    match_score = None
    if user_description and user_description.strip():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        descriptions = []
        trait_vecs = []
        for style in leadership_styles:
            pros_text = " ".join(style["pros"])
            cons_text = " ".join(style["cons"])
            full = f"{style['name']}: {style['traits']}. {pros_text}. {cons_text}"
            descriptions.append(full)
            trait_vecs.append(np.array(style["traits_vector"]))
        leadership_embeddings = model.encode(descriptions, convert_to_tensor=True)
        user_embedding = model.encode(user_description, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, leadership_embeddings)[0].cpu().numpy()
        para_vector = np.zeros(4)
        for i, score in enumerate(cosine_scores):
            para_vector += score * trait_vecs[i]
        para_vector /= np.sum(cosine_scores)
        para_best_idx = int(np.argmax(cosine_scores))
        match_score = cosine_scores[para_best_idx]

    if mcq_vector is not None and para_vector is not None:
        combo = 0.7 * mcq_vector + para_weight * para_vector
    elif mcq_vector is not None:
        combo = mcq_vector
    elif para_vector is not None:
        combo = para_vector
    else:
        return None

    best_idx = np.argmax([
        np.dot(combo, np.array(style["traits_vector"])) /
        (np.linalg.norm(combo) * np.linalg.norm(np.array(style["traits_vector"])))
        for style in leadership_styles
    ])
    selected = leadership_styles[best_idx]
    confidence = float(np.dot(combo, np.array(selected["traits_vector"])) /
                       (np.linalg.norm(combo) * np.linalg.norm(np.array(selected["traits_vector"]))))
    return {
        "name": selected["name"],
        "traits": selected["traits"],
        "pros": selected["pros"],
        "cons": selected["cons"],
        "confidence": confidence,
        "mcq_answers": mcq_answers if mcq_answers is not None else [],
        "trait_vector": combo.tolist(),
        "para_input": user_description if user_description else ""
    }

def update_info_with_power_dynamic(info: dict, power_dynamic: dict) -> dict:
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

####### 7. Categorize the Situation (Step 3) #####################
def categorize_situation(client, info_dict: dict) -> dict:
    categorization_prompt = f"""
    Given the following JSON representing details about a user's workplace situation:
    {json.dumps(info_dict, indent=2)}

    Categorize this situation into one of the following categories:
    1. Promotion
    2. Sign-on (new job or role)
    3. Work-related problems (e.g., team conflict, burnout, workload)

    Return a valid JSON object with two fields:
    {{
    "category": "...",  
    "explanation": "..."  
    }}
    Only return valid JSON. Do not include explanations or disclaimers.
    """
    messages = [{"role": "system", "content": categorization_prompt}]
    situation = ask_gpt(client, messages, temperature=0.1, max_tokens=300)
    try:
        category_result = json.loads(situation)
        return category_result
    except json.JSONDecodeError:
        return {
            "category": "Work-related problems",
            "explanation": "Unable to parse GPT’s response, defaulting to Work-related problems."
        }

############ 8. Load Scenario Data (Step 4) #####################
def load_scenario_data(scenario_type: str, category: str, level: int = None) -> dict:
    category_map = {
        "Promotion": "promotion",
        "Sign-on (new job or role)": "sign_on",
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
    else:
        levels = [4, 5]
    for lvl in levels:
        path = f"./dataset/{folder}_{scenario}/{folder}_{scenario}_scenario_level_{lvl}_response.txt"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    combined["dialogue"].extend(data.get("dialogue", []))
                except Exception:
                    pass
    if not combined["dialogue"]:
        raise FileNotFoundError("No valid scenario files were loaded.")
    return combined

def extract_advisor_traits(ideal_dialogue: dict) -> tuple:
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

def get_next_assistant_turn(client: AzureOpenAI, visible_messages: list, info: dict, advisor_traits: list, scenario_type: str, category: str, turn_count: int, rehearsal_level: int = None) -> Tuple[str, int]:
    turn_count = sum(1 for m in visible_messages if m['role'] == 'assistant')
    context = [m.copy() for m in visible_messages]
    user_tone = info.get("language_style", {})
    user_empathy = user_tone.get("empathy", 0.0)
    user_formality = user_tone.get("formality", 0.0)
    user_persuasiveness = user_tone.get("persuasiveness", 0.0)
    user_politeness = user_tone.get("politeness", 0.0)
    leadership_style = info.get("leadership_style", {})
    boss_style = leadership_style.get("name", "Unknown")
    boss_traits = leadership_style.get("traits", "Unknown")
    boss_pros = leadership_style.get("pros", [])
    boss_cons = leadership_style.get("cons", [])
    relationship = info.get("relationship", {}).get("description", "Unknown")
    individual = info.get("individual", {}).get("description", "Unknown")
    topic = info.get("topic", {}).get("description", "Unknown")
    previous_interaction = info.get("previous_interaction", {}).get("description", "Unknown")

    feedback_tone_block = construct_feedback_and_tone_prompt(info, advisor_traits, turn_count)
    scenario_context = f"""
    You are participating in a workplace conversation scenario.
    The user is preparing for a discussion about: {topic}.
    The person they are talking to has a leadership style described as: {boss_style} ({boss_traits}).
    Key strengths: {boss_pros}
    Key challenges: {boss_cons}
    Their relationship: {relationship}

    The user currently discusses the topic with the following language style:
    - Empathy: {user_empathy}
    - Formality: {user_formality}
    - Persuasiveness: {user_persuasiveness}
    - Politeness: {user_politeness}

    The user is preparing for a conversation with {individual} about {topic}. Look at the details of their {previous_interaction}.

    {feedback_tone_block}
    """

    unused = [e for e in advisor_traits if not e["used"]]
    if not unused:
        for e in advisor_traits: e["used"] = False
        unused = advisor_traits
    pick = random.choice(unused)
    pick["used"] = True

    if scenario_type == "rehearsal":
        realism_str = f"(Realism level: {rehearsal_level}/5; 1 = most unreasonable, 5 = most reasonable)" if rehearsal_level else ""
        instruction = f"""
        You are role-playing as the user's counterpart (e.g., boss, manager, HR) in a workplace rehearsal scenario about "{topic}". {realism_str}
        Respond as they would, using their leadership style ({boss_style}). Begin by briefly acknowledging the user's message, then continue the conversation from the counterpart's perspective.
        Now, incorporate the following Brett element into your response: "{pick['element']}".
        Here's an example of how this element was used in a similar context: "{pick['example']}"
        Provide ONE specific, realistic response that incorporates this element.
        Be realistic for the selected level. Do not provide advice or feedback—just respond as the other party.
        """
    elif scenario_type == "advice":
        instruction = f"""
        You are coaching the user on how to handle a workplace conversation about "{topic}".
        Their boss is a '{boss_style}' leader ({boss_traits}) with strengths like {boss_pros}, and challenges like {boss_cons}.
        Now, incorporate the following Brett element into your response: "{pick['element']}".
        Here's an example of how this element was used in a similar context: "{pick['example']}"
        Provide ONE specific, actionable suggestion that incorporates this element.
        Don't overwhelm the user with multiple suggestions. Begin by briefly acknowledging their situation, then give your advice, and finally ask a gentle follow-up question.
        """
    else:  # emotional_support
        instruction = f"""
        You are providing emotional support to a user dealing with a workplace conversation about "{topic}".
        Be empathetic, validating, and supportive. Use Brett elements to guide your support.
        Now, incorporate the following Brett element into your response: "{pick['element']}".
        Here's an example of how this element was used in a similar context: "{pick['example']}"
        Give one encouraging and supportive message, then ask how they are feeling or what they'd like help with next.
        """

    context.append({"role": "system", "content": scenario_context + "\n" + instruction})

    reply = ask_gpt(client, context, temperature=0.7, max_tokens=300).strip()
    visible_messages.append({"role": "assistant", "content": reply, "brett_element": pick["element"], "brett_example": pick["example"]})
    return reply, turn_count + 1

def summarize_user_feedback(feedback_list, client, info, advisor_traits, turn_count):
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

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def construct_feedback_and_tone_prompt(info: dict, traits: list, turn_count: int) -> str:
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

            Adjust your tone to match or strategically counterbalance this style based on the scenario’s power dynamics.
        """

    if traits:
        current_trait = traits[turn_count % len(traits)].get("element", "neutral_tone")
    else:
        current_trait = "neutral_tone"
    trait_note = f"\nFocus this turn on the trait: **{current_trait}**\n"

    if not (feedback_section or tone_summary):
        return ""
    return feedback_section + tone_summary + trait_note

######################## 12. General Model ####################
def general_next_assistant_turrn(client: AzureOpenAI, visible_messages: list) -> str:
    """
    Simple turn-by-turn conversation with built-in system prompt
    """
    # Define system prompt inside the function
    system_prompt = """You are a helpful workplace conversation assistant. Try to understand what the user wants to gain help about the problem at hand, have they discussed it with the person before, and what is the relationship between them.
    Further from there ask them if they wanna rehearse or just gain advice about it and then get going. 
    Provide clear, professional advice for workplace situations. """
    
    # Create context with conversation history + system prompt
    context = [{"role": "system", "content": system_prompt}]
    
    # Add all previous conversation messages
    for message in visible_messages:
        if message.get("role") in ["user", "assistant"]:
            context.append({
                "role": message["role"], 
                "content": message["content"]
            })
    
    # Get response from the model
    reply = ask_gpt(client, context, temperature=0.7, max_tokens=300).strip()
    
    # Add assistant response to visible messages
    visible_messages.append({"role": "assistant", "content": reply})
    
    return reply
############ 13. Save chat ########################
def save_chat_locally(info, category, language_analysis, messages, advisor_traits, scenario_type, rehearsal_level=None, folder="saved_chats", feedback_log=None):
    os.makedirs(folder, exist_ok=True)
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "scenario_type": scenario_type,
        "rehearsal_level": rehearsal_level,
        "info": info,
        "category": category,
        "language_analysis": language_analysis,
        "advisor_traits": advisor_traits,
        "messages": messages,
        "feedback_log": feedback_log or [],
    }
    filename = os.path.join(folder, f"chat_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json")
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"[Saved] Chat written to: {filename}")
    return filename
