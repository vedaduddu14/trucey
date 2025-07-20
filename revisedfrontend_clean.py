"""
SoCALM Trucey - Frontend Interface
Hello, if you're reading this, it means you're looking at our source code for the frontend built on Streamlit.
This file handles the UI for 4 workplace conversation systems:

1. Trucey Rehearsal: Based on the SoCALM framework, this system provides advice followed by rehearsal on how to handle workplace conversations on various topics. 
2. Control System: A contextualized system with no scaffolding, leading to a direct chat where users can interact with the system without any additional support. 

Structure:
- Session State Management
- Shared UI Components
- System Router
- Main Application Flow
"""

import streamlit as st 
from datetime import datetime
from revised_backend_CLEAN import (
    LanguageStyleAnalyzer,
    get_openai_client,
    save_chat_locally,
    authenticate_participant,
    convert_csv_to_info_format,
    get_power_dynamic_model,
    update_info_with_power_dynamic,
    categorize_situation,
    load_combined_scenario_data,
    summarize_user_feedback,
    get_next_assistant_turn,
    general_next_assistant_turn,
    generate_sentence_starters,
    get_fallback_starters
)

TRUCEY_REHEARSAL = "trucey_rehearsal"
CONTROL_SYSTEM = "control_system"

FEEDBACK_INTERVAL = 2
MAX_TURNS = 10
MIN_INTERACTIONS = 2

#power calibration leadership mcq questions
MCQ_QUESTIONS = [
    {"q": "My boss is naturally creative and enjoys finding innovative solutions to problems", "trait": "openness"},
    {"q": "My boss is assertive and tends to take charge in most situations", "trait": "extraversion"},
    {"q": "My boss genuinely cares about maintaining good relationships with everyone", "trait": "agreeableness_1"},
    {"q": "My boss stays calm and composed even in stressful situations", "trait": "emotional_stability"},
    {"q": "My boss prefers collaborative approaches over competitive ones", "trait": "agreeableness_2"}
]

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Core components
        "client": get_openai_client(),
        "analyzer": LanguageStyleAnalyzer(),
        
        # Login state
        "login_attempted": False,
        "login_error": "",
        "authenticated": False,
        
        # Time tracking
        "session_start_time": None,
        "chat_start_time": None,
        "advice_start_time": None,
        "rehearsal_start_time": None,
        "timestamps": [],
        
        # User data and conversation
        "info": {},
        "messages": [],
        "user_feedback": [],
        "feedback_log": [],
        "participant_data": None,
        "participant_loaded": False,
        
        # UI state and navigation
        "step": "home",
        "system_type": "",
        "awaiting_response": False,
        "awaiting_info_update": False,
        "awaiting_control_response": False,
        
        # Leadership and power dynamics
        "leadership_result": False,
        "leader_input": "",
        "leadership_description": "",
        "mcq_answers": ["", "", "", "", ""],
        "mcq_submitted": False,
        
        # Scenario and conversation management
        "category": None,
        "traits": [],
        "responses": [],
        "rehearsal_level": 3,
        "scenario_loaded": False,
        "turn_count": 0,
        
        # Chat state flags
        "chat_started": False,
        "control_chat_started": False,
        "next_question": "",
        "current_phase": "advice",
        "advice_turn_count": 0,
        "rehearsal_turn_count": 0,
        "advice_traits": [],
        "rehearsal_traits": [],
        "advice_choice_made": False,
        "waiting_for_feedback": False,
        "message_starter": None,
        "current_input": "",
        "starter_clicked": False,
        "input_populated": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============================================================================
# TIME TRACKING HELPER FUNCTIONS
# ============================================================================

def log_event(event_name, additional_data=None):
    """
    Log a timestamped event
    Args:
        event_name (str): Name of the event to log
        additional_data (dict, optional): Additional data to include in the log
    Returns:
        None
    """
    timestamp = datetime.now().isoformat()
    event = {
        "event": event_name,
        "timestamp": timestamp,
        "data": additional_data or {}
    }
    
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    
    st.session_state.timestamps.append(event)
    print(f"LOGGED EVENT: {event_name} at {timestamp}")

def calculate_duration(start_time, end_time=None):
    """
    Calculate duration between two timestamps
    Args:
        start_time (str): Start time in ISO format
        end_time (str, optional): End time in ISO format. Defaults to current time.
    Returns:
        float: Duration in seconds, or None if start_time is not provided
    """
    if not start_time:
        return None
    
    if end_time is None:
        end_time = datetime.now().isoformat()
    
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)
    
    duration = end - start
    return duration.total_seconds()

# ============================================================================
# UNIFIED HELPER FUNCTIONS
# ============================================================================

def render_scenario_info(participant_data, expandable=True, expanded=False):
    """
    Unified function to render scenario information
    Args:
        participant_data (dict): Participant data containing scenario details
        expandable (bool): Whether to use an expander for the scenario content
        expanded (bool): Initial state of the expander
    """
    scenario_content = f"""
    **Your Situation:**
    - **Problem to discuss:** {participant_data['assigned_problem'].replace('_', ' ').title()}
    - **Person you're talking to:** Your {participant_data['person_of_interest']}
    - **Your relationship:** You have had a {participant_data['relationship_quality']} relationship for {participant_data['relationship_length'].replace('_', ' ')}
    - **Previous discussions:** {'You have discussed this before' if participant_data['has_topic_been_discussed'] == 'yes' else 'This is the first time bringing this up'}
    - **Work context:** You are a {participant_data['payment_type'].replace('_', ' ')}
    """
    
    if expandable:
        with st.expander("View Your Assigned Scenario", expanded=expanded):
            st.markdown(scenario_content)
    else:
        st.markdown(scenario_content)

def count_user_messages(messages, phase=None):
    """
    Count user messages, optionally filtered by phase as we do have a different number of user messages allowed in each stage
    Args:
        messages (list): List of message dictionaries
        phase (str, optional): Phase to filter messages by (e.g., "advice", "rehearsal"). Defaults to None.
    Returns:
        int: Count of user messages, optionally filtered by phase
    """
    if phase:
        return len([msg for msg in messages if msg.get("role") == "user" and msg.get("phase") == phase])
    return len([msg for msg in messages if msg.get("role") == "user"])

def render_end_chat_with_minimum_check(messages, system_type, phase=None, min_interactions=MIN_INTERACTIONS):
    """Unified function to render end chat button with minimum interaction check"""
    user_count = count_user_messages(messages, phase)
    
    if user_count >= min_interactions:
        st.markdown("*You can end the conversation when you feel ready, or continue practicing.*")
        render_section_separator()
        
        if render_end_chat_buttons():
            session_duration = calculate_duration(st.session_state.get("session_start_time"))
            chat_duration = calculate_duration(st.session_state.get("chat_start_time"))
            
            log_event("session_ended", {
                "total_session_duration": session_duration,
                "chat_duration": chat_duration,
                "total_messages": len(st.session_state.messages),
                "user_messages": count_user_messages(st.session_state.messages)
            })
            
            filename = save_chat_locally(
                info=st.session_state.get("info", {}),
                category=st.session_state.get("category", {}),
                language_analysis=st.session_state.get("language_analysis", {}),
                messages=st.session_state.get("messages", []),
                advisor_traits=st.session_state.get("advice_traits", []) + st.session_state.get("rehearsal_traits", []),
                scenario_type=system_type,
                rehearsal_level=st.session_state.get("rehearsal_level"),
                feedback_log=st.session_state.get("feedback_log", []),
                time_tracking={
                    "session_start_time": st.session_state.get("session_start_time"),
                    "chat_start_time": st.session_state.get("chat_start_time"),
                    "advice_start_time": st.session_state.get("advice_start_time"),
                    "rehearsal_start_time": st.session_state.get("rehearsal_start_time"),
                    "session_end_time": datetime.now().isoformat(),
                    "total_session_duration_seconds": session_duration,
                    "chat_duration_seconds": chat_duration,
                    "timestamps": st.session_state.get("timestamps", [])
                },
                participant_data=st.session_state.get("participant_data", {})
            )
            render_success_message(f"Chat saved to `{filename}`")
            render_success_message("Chat saved. You can now close this tab or stop the server.")
            return True

def render_sentence_starters():
    """
    Unified function to render sentence starters for prompting users as to how to message
    """
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "assistant" and
        not st.session_state.get("waiting_for_feedback", False) and
        not st.session_state.get("starter_clicked", False)):
        
        starter_key = f"starters_{len(st.session_state.messages)}"
        if starter_key not in st.session_state:
            try:
                last_message = st.session_state.messages[-1]["content"]
                last_msg_obj = st.session_state.messages[-1]
                brett_element = last_msg_obj.get("brett_element", None)
                brett_example = last_msg_obj.get("brett_example", None)
                
                starters = generate_sentence_starters(
                    st.session_state.client,
                    last_message, 
                    st.session_state.current_phase, 
                    st.session_state.participant_data,
                    brett_element,
                    brett_example
                )
                st.session_state[starter_key] = starters
                
            except Exception as e:
                print(f"Error generating starters: {e}")
                st.session_state[starter_key] = get_fallback_starters(st.session_state.current_phase)
        
        starters = st.session_state[starter_key]
        if starters:
            st.markdown("**Try:**")
            cols = st.columns(len(starters))
            for i, starter in enumerate(starters):
                with cols[i]:
                    if st.button(f"{starter}", 
                            key=f"starter_{i}_{len(st.session_state.messages)}", 
                            use_container_width=True):
                        st.session_state.message_starter = starter
                        st.session_state.starter_clicked = True
                        st.rerun()

def handle_phase_transition():
    """
    Handle transition from advice to rehearsal phase
    """
    st.session_state.current_phase = "rehearsal"
    st.session_state.advice_choice_made = True
    
    get_next_assistant_turn(
        st.session_state.client,
        st.session_state.messages,
        st.session_state.info,
        st.session_state.rehearsal_traits,
        "rehearsal",
        st.session_state.category["category"],
        st.session_state.rehearsal_turn_count,
        st.session_state.get("rehearsal_level"),
        st.session_state.participant_data,
        st.session_state.analyzer
    )
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages[-1]["phase"] = "rehearsal"
    
    st.session_state.rehearsal_turn_count += 1

def render_chat_input_unified():
    """
    Unified chat input handling
    """
    conversation_complete = ((st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 5) or
                           (st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7))
    
    if (not conversation_complete and 
        not st.session_state.get("awaiting_response", False) and
        not st.session_state.get("waiting_for_feedback", False)):
        
        if st.session_state.get("message_starter") and not st.session_state.get("input_populated", False):
            st.session_state.current_input = st.session_state.message_starter
            st.session_state.input_populated = True
            del st.session_state.message_starter

        user_text = st.text_area(
            "Your message:", 
            value=st.session_state.current_input,
            key="user_input_direct",
            placeholder="Type your message or click a starter above...",
            height=100,
            label_visibility="visible"
        )

        if user_text != st.session_state.current_input:
            st.session_state.current_input = user_text

        if st.button("Send", key="send_btn") and user_text.strip():
            st.session_state.current_input = ""
            st.session_state.starter_clicked = False
            st.session_state.input_populated = False
            
            user_message = {"role": "user", "content": user_text, "phase": st.session_state.current_phase}
            st.session_state.messages.append(user_message)
            
            if (st.session_state.current_phase == "rehearsal" and 
                st.session_state.rehearsal_turn_count > 0 and 
                st.session_state.rehearsal_turn_count % 2 == 0 and
                st.session_state.rehearsal_turn_count < 7):
                st.session_state.waiting_for_feedback = True
            else:
                st.session_state.awaiting_response = True
            st.rerun()
    
    elif st.session_state.get("awaiting_response", False):
        st.info("Generating response...")

# ============================================================================
# SHARED UI COMPONENTS
# ============================================================================
def render_page_header(title):
    st.title(title)

def render_section_separator():
    st.markdown("---")

def render_current_step_header(step_number, step_name):
    st.subheader(f"Step {step_number}: {step_name}")

def render_success_message(message):
    st.success(message)

def render_info_message(message):
    st.info(message)

def render_continue_button(label="Continue to Next Step"):
    return st.button(label, use_container_width=True)

def render_end_chat_buttons():
    return st.button("End Chat and Save Conversation", use_container_width=True)

def render_chat_messages(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

def render_chat_input(placeholder="Your message…", key_suffix=""):
    return st.chat_input(placeholder, key=f"chat_input_{key_suffix}")

def render_feedback_form(turn_count):
    with st.expander("Optional: Give feedback on the assistant's tone or realism"):
        feedback_input = st.text_area(
            "What could the assistant improve? (e.g., tone, realism, phrasing)",
            key=f"feedback_text_{turn_count}"
        )
        if st.button("Submit Feedback", key=f"feedback_submit_{turn_count}", use_container_width=True):
            return feedback_input.strip() if feedback_input.strip() else ""
    return None

# ============================================================================
# MAIN RENDER FUNCTIONS
# ============================================================================
def render_home_page():
    """
    Login page with time tracking
    As soon as the participant logs in, we start the session timer
    The participant logging in will be convert to a hash program with from their prolific ID instead
    """
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Participant Login")
    st.info("Please enter your assigned participant ID and password to begin the study.")
    
    if st.session_state.login_error:
        st.error(st.session_state.login_error)
    
    with st.form("login_form"):
        participant_id = st.text_input("Participant ID:", placeholder="e.g., trial_user_1")
        login_btn = st.form_submit_button("Login", use_container_width=True)
        
        if login_btn:
            st.session_state.login_attempted = True
            st.session_state.login_error = ""
            
            participant_data = authenticate_participant(participant_id)
            
            if participant_data:
                st.session_state.session_start_time = datetime.now().isoformat()
                log_event("session_started", {
                    "participant_id": participant_id,
                    "assigned_system": participant_data['assigned_system']
                })
                
                st.session_state.authenticated = True
                st.session_state.participant_data = participant_data
                st.session_state.participant_loaded = True
                st.session_state.system_type = participant_data['assigned_system']

                st.session_state.info = participant_data
                log_event("participant_info_loaded", {
                    "problem": participant_data['assigned_problem'],
                    "person_of_interest": participant_data['person_of_interest'],
                    "relationship_quality": participant_data['relationship_quality']
                })

                if participant_data['assigned_system'] == TRUCEY_REHEARSAL:
                    st.session_state.step = "info"
                    log_event("trucey_system_entered")
                elif participant_data['assigned_system'] == CONTROL_SYSTEM:
                    st.session_state.step = "control_info"
                    st.session_state.current_phase = "control" 
                    log_event("control_system_entered")
                    
                st.session_state.messages = []
                st.rerun()
            else:
                st.session_state.login_error = "Invalid participant ID or password. Please try again."
                log_event("login_failed", {"participant_id": participant_id})
                st.rerun()

def render_info_gathering_step():
    """
    Step 1: Show CSV data that was loaded
    This is our stage 1 where we provide the details by which our participant must abide by
    There is no difference between the rehearsal and control system at this stage but we have two different functions currently
    """
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Stage 1: Understanding Your Situation")
    st.info("""
    **What happens in this stage:**
    - We've loaded details of your assigned workplace scenario
    - Review the details and confirm you understand the scenario
    - This information will be used to customize your conversation practice
    
    **Your task:** Review the details and confirm you understand the scenario.
    """)
    
    if not st.session_state.participant_loaded:
        st.error("Participant data not loaded. Please contact the researcher.")
        return
    
    with st.expander("Review Your Scenario", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Conversation Details:**")
            st.write(f"• **Topic:** {st.session_state.info['topic']}")
            st.write(f"• **Person:** {st.session_state.info['person_of_interest']}")
            st.write(f" • **Payment Type:** {st.session_state.info['payment_type']}")
        
        with col2:
            st.markdown("**Relationship Context:**")
            st.write(f"• **Relationship:** {st.session_state.info['relationship_quality']}")
            st.write(f"• **Previous Discussion:** {st.session_state.info['has_topic_been_discussed']}")
    
    log_event("participant_reviewed_loaded_info")
    
    st.markdown("---")
    st.markdown("**All set!** This information will help us provide personalized guidance for your conversation.")
    
    if render_continue_button("I've reviewed my information - Continue"):
        log_event("stage1_completed")
        st.session_state.step = "power"
        st.rerun()

def render_control_info_step():
    """Control system scenario display
    Step 1: Show CSV data that was loaded
    This is our stage 1 where we provide the details by which our participant must abide by
    There is no difference between the rehearsal and control system at this stage but we have two different functions currently
    """
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Stage 1: Understanding Your Situation")
    st.info("""
    **What happens next:**
    - We'll show you the workplace scenario you'll be working through
    - This is based on your interest form responses  
    - You'll then chat directly with an AI assistant for guidance
    - The assistant will help you explore your situation and develop an approach
    """)
    
    if not st.session_state.participant_loaded:
        st.error("Participant data not loaded. Please contact the researcher.")
        return
    
    st.markdown("### Your Assigned Scenario")
    st.info("Based on your interest form responses, here is the workplace situation we'd like you to work through:")
    
    render_scenario_info(st.session_state.participant_data, expandable=False)
    
    st.markdown("---")
    st.markdown("**Take a moment to imagine yourself in this situation.** The AI assistant will help you explore your options and develop an approach for this conversation.")
    
    if render_continue_button("I understand my scenario - Start Chat"):
        st.session_state.step = "control_chat"
        st.rerun()

def render_power_dynamic_step():
    """
    Step 2: Leadership assessment with time tracking
    Through a mcq form, we'll do our best in assessing the leadership style their boss could have based on personality traits
    """
    render_page_header("SoCALM's Trucey")
    render_section_separator()
    
    st.markdown("### Stage 2: Analyzing Leadership Style")
    st.info("""
    **What happens in this stage:**
    - You'll answer 5 questions about your boss's personality and leadership approach
    - The system will match their style to provide tailored advice
    - This helps us customize the conversation guidance for your specific situation
    
    **Your task:** Rate each statement based on how well it describes your boss, then optionally add any additional details.
    """)
    
    if not st.session_state.leadership_result:
        if "leadership_assessment_started" not in [event["event"] for event in st.session_state.get("timestamps", [])]:
            log_event("leadership_assessment_started")
        
        st.subheader("Step 2: Describe your boss's nature and leadership style")
        st.markdown("**Rate how well each statement describes your boss's general approach and personality:**")
        
        with st.form("leadership_form", clear_on_submit=False):
            mcq_answers = []
            
            for i, qdict in enumerate(MCQ_QUESTIONS):
                st.markdown(f"<h4 style='text-align: center; margin-bottom: 30px;'>Q{i+1}. {qdict['q']}</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    rating = st.radio(
                        "Rating:",
                        options=[1, 2, 3, 4, 5],
                        index=2,
                        format_func=lambda x: str(x),
                        key=f"mcq_{i}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                
                st.markdown("""
                <div style="display: flex; justify-content: center; margin-top: 5px; margin-bottom: 25px;">
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; width: 420px; text-align: center;">
                        <div style="font-size: 0.75em; color: #666; line-height: 1.2;">Strongly<br>Disagree</div>
                        <div style="font-size: 0.75em; color: #666;">Disagree</div>
                        <div style="font-size: 0.75em; color: #666;">Neutral</div>
                        <div style="font-size: 0.75em; color: #666;">Agree</div>
                        <div style="font-size: 0.75em; color: #666; line-height: 1.2;">Strongly<br>Agree</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                mcq_answers.append(rating)
                
                if i < len(MCQ_QUESTIONS) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            leader_input = st.text_area(
                "**Additional Description (Optional):** In 2-3 sentences, add anything else you would like to use to describe this person's leadership style or behavior:",
                value=st.session_state.leader_input
            )
            
            submit_leadership = st.form_submit_button("Analyze Leadership Style")
            
            if submit_leadership:
                log_event("leadership_assessment_completed", {
                    "mcq_answers": mcq_answers,
                    "has_text_input": bool(leader_input.strip())
                })
                
                st.session_state.mcq_answers = mcq_answers
                st.session_state.leader_input = leader_input
                
                leadership = get_power_dynamic_model(user_description=leader_input, mcq_answers=mcq_answers)
                
                if leadership:
                    log_event("leadership_style_detected", {
                        "style_name": leadership.get("name"),
                        "confidence": leadership.get("confidence")
                    })
                
                st.session_state.info = update_info_with_power_dynamic(st.session_state.info, leadership)
                
                if leadership and leadership.get("obtained", True):
                    lines = [
                        f"**Based on your answers, the best match is:** {leadership['name']}",
                        "",
                        f"--- **{leadership['name']}** Leadership Style ---",
                        "",
                        "**Strengths:**"
                    ]
                    for pro in leadership["pros"]:
                        lines.append(f"- {pro}")
                    lines.append("")
                    lines.append("**Challenges:**")
                    for con in leadership["cons"]:
                        lines.append(f"- {con}")
                    st.session_state.leadership_description = "\n\n".join(lines)
                else:
                    st.session_state.leadership_description = "*Leadership analysis not available.*"
                
                st.session_state.leadership_result = True
                st.rerun()
    else:
        st.markdown(f"**You wrote:** {st.session_state.leader_input}")
        st.markdown(st.session_state.leadership_description)
        st.markdown(f"**Quiz answers:** {', '.join(map(str, st.session_state.mcq_answers))}")
        
        if render_continue_button("Continue to Next Step"):
            log_event("transitioning_to_chat")
            with st.spinner("Preparing your conversation..."):
                auto_categorize_and_load()
            st.rerun()

def auto_categorize_and_load():
    """
    Background categorization and scenario loading
    Based on the topic at hand, we need to load data so ChatGPT essentially takes a stab at categorizing the situation here
    """
    if st.session_state.category is None:
        st.session_state.category = categorize_situation(
            st.session_state.client,
            st.session_state.info
        )

    if st.session_state.category and "category" in st.session_state.category:
        original_category = st.session_state.category["category"]
        cleaned_category = original_category.split(' (')[0].strip()
        st.session_state.category["category"] = cleaned_category
    
    st.session_state.rehearsal_level = 1
    
    if not st.session_state.get("scenario_loaded", False):
        combined_data = load_combined_scenario_data(
            st.session_state.category["category"],
            st.session_state.get("rehearsal_level")
        )
        
        st.session_state.advice_traits = combined_data["advice_traits"]
        st.session_state.advice_responses = combined_data["advice_responses"]
        st.session_state.rehearsal_traits = combined_data["rehearsal_traits"]
        st.session_state.rehearsal_responses = combined_data["rehearsal_responses"]
        st.session_state.traits = combined_data["advice_traits"]
        st.session_state.responses = combined_data["advice_responses"]
        st.session_state.scenario_loaded = True
    
    st.session_state.step = "chat"

def render_trucey_chat_step():
    """
    Main Trucey conversation interface with time tracking
    For the treatmenet system, we have two phases:
    - Phase 1: Advice (5 responses) (user gets to send in 5 messages to the assistant to get advice on how to attempt the situation at hand)
    - Phase 2: Rehearsal (7 responses) (user gets to practice the conversation with the assistant role-playing as their boss based on the leadership style calibrated)
    """
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Stage 3: Conversation Practice")
    st.info("""
    **What happens in this stage:**
    - **Phase 1 - Advice (5 responses):** Get strategic guidance for your situation
    - **Phase 2 - Rehearsal (7 responses):** Practice the conversation with me role-playing as your boss
    
    **Your task:**
    - Ask questions, share concerns, and get advice in Phase 1
    - Practice your approach and responses in Phase 2
    - Use the suggested sentence starters or type your own messages
    """)
    st.markdown("---")
    
    # Phase headers
    if st.session_state.current_phase == "advice":
        render_current_step_header(3, f"Phase 1: Getting Advice ({st.session_state.advice_turn_count}/5 responses)")
        if st.session_state.advice_turn_count == 0:
            st.markdown("*The system will provide strategic advice based on your situation and boss's leadership style.*")
    else:
        render_current_step_header(3, f"Conversation - Advice Phase (Complete)")
        st.markdown("*The system will provide strategic advice based on your situation and boss's leadership style.*")
        
        if st.session_state.rehearsal_turn_count == 0:
            st.markdown("### Now Starting: Rehearsal Phase")
            st.warning("""
            **Instructions for Rehearsal:**
            - I will now role-play as your boss
            - Practice the conversation using the advice you just received
            - This is your chance to rehearse what you'll actually say
            - I'll respond based on your boss's leadership style and personality
            """)
            st.markdown("---")
    
    if not st.session_state.chat_started:
        st.session_state.chat_start_time = datetime.now().isoformat()
        st.session_state.advice_start_time = datetime.now().isoformat()
        log_event("chat_started", {"phase": "advice"})
        log_event("advice_phase_started")
        
        st.session_state.messages = []
        
        get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            st.session_state.advice_traits,
            "advice",
            st.session_state.category["category"],
            st.session_state.advice_turn_count,
            st.session_state.get("rehearsal_level"),
            st.session_state.participant_data,
            st.session_state.analyzer
        )
        
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages[-1]["phase"] = "advice"
        
        st.session_state.chat_started = True
        st.session_state.advice_turn_count += 1

    if st.session_state.current_phase == "advice":
        render_chat_messages(st.session_state.messages)
    else:
        advice_messages = [msg for msg in st.session_state.messages if msg.get("phase", "advice") == "advice"]
        render_chat_messages(advice_messages)
        
        st.markdown("---")
        if st.session_state.rehearsal_turn_count == 1:
            st.markdown("*I'm now role-playing as your boss. Practice the conversation using the advice you received.*")
        render_current_step_header(3, f"Conversation - Rehearsal Phase ({st.session_state.rehearsal_turn_count}/7 responses)")
        
        rehearsal_messages = [msg for msg in st.session_state.messages if msg.get("phase") == "rehearsal"]
        render_chat_messages(rehearsal_messages)
    
    if (st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 2):
        if st.session_state.advice_turn_count >= 5:
            st.info("Great! Now let's practice this conversation. I'll role-play as your boss so you can rehearse what we just discussed.")
            if st.button("Start Rehearsal Practice", use_container_width=True):
                st.session_state.rehearsal_start_time = datetime.now().isoformat()
                log_event("rehearsal_phase_started", {"transition_type": "automatic"})
                
                advice_duration = calculate_duration(st.session_state.advice_start_time)
                log_event("advice_phase_completed", {"duration_seconds": advice_duration})
                
                handle_phase_transition()
                st.rerun()
            return
        else:
            if not st.session_state.get("advice_choice_made", False):
                st.info(f"You've received {st.session_state.advice_turn_count} pieces of advice. What would you like to do next?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Start Rehearsal Practice Now", use_container_width=True):
                        st.session_state.rehearsal_start_time = datetime.now().isoformat()
                        log_event("rehearsal_phase_started", {"transition_type": "early"})
                        advice_duration = calculate_duration(st.session_state.advice_start_time)
                        log_event("advice_phase_completed", {"duration_seconds": advice_duration})
                        handle_phase_transition()
                        st.rerun()
                with col2:
                    if st.button("Continue Getting Advice", use_container_width=True):
                        log_event("user_chose_continue_advice")
                        st.session_state.advice_choice_made = True
                        st.rerun()
                return

    render_sentence_starters()
    
    if st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7:
        st.success("Conversation practice complete! Thank you for participating.")
        if render_end_chat_buttons():
            filename = save_chat_locally(
                info=st.session_state.get("info", {}),
                category=st.session_state.get("category", {}),
                language_analysis=st.session_state.get("language_analysis", {}),
                messages=st.session_state.get("messages", []),
                advisor_traits=st.session_state.get("advice_traits", []) + st.session_state.get("rehearsal_traits", []),
                scenario_type="trucey_rehearsal",
                rehearsal_level=st.session_state.get("rehearsal_level"),
                feedback_log=st.session_state.get("feedback_log", []),
                participant_data=st.session_state.get("participant_data", {})
            )
            render_success_message(f"Chat saved to `{filename}`")
            render_success_message("Chat saved. You can now close this tab or stop the server.")
        return

    if st.session_state.current_phase == "rehearsal" and not st.session_state.get("waiting_for_feedback", False):
        if render_end_chat_with_minimum_check(st.session_state.messages, "trucey_rehearsal", "rehearsal"):
            return

    if st.session_state.current_phase == "rehearsal" and st.session_state.get("waiting_for_feedback", False):
        feedback = render_feedback_form(st.session_state.rehearsal_turn_count)
        if feedback is not None:
            if feedback:
                st.session_state.user_feedback.append(feedback)
                st.session_state.feedback_log.append({
                    "content": feedback,
                    "turn_count": st.session_state.rehearsal_turn_count,
                    "timestamp": datetime.now().isoformat(),
                })
                render_success_message("Feedback received — will be used in the next assistant response.")
            else:
                render_success_message("Thank you! Continuing without feedback.")

            st.session_state.info["user_feedback"] = st.session_state.user_feedback
            if st.session_state.user_feedback:
                summary = summarize_user_feedback(
                    st.session_state.user_feedback,
                    st.session_state.client,
                    st.session_state.info,
                    st.session_state.rehearsal_traits,
                    st.session_state.rehearsal_turn_count
                )
                st.session_state.info["latest_feedback_summary"] = summary

            st.session_state.waiting_for_feedback = False
            st.session_state.awaiting_response = True
            st.rerun()
        return

    render_chat_input_unified()
    
    if st.session_state.get("awaiting_response", False):
        last_user_msg = st.session_state.messages[-1]["content"]
        
        style_scores = st.session_state.analyzer.analyze_text(last_user_msg)
        if "language_style" not in st.session_state.info:
            st.session_state.info["language_style"] = {}
        st.session_state.info["language_style"].update(style_scores)

        if st.session_state.current_phase == "advice":
            current_traits = st.session_state.advice_traits
            scenario_type = "advice"
            turn_count = st.session_state.advice_turn_count
        else:
            current_traits = st.session_state.rehearsal_traits
            scenario_type = "rehearsal"
            turn_count = st.session_state.rehearsal_turn_count

        get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            current_traits,
            scenario_type,
            st.session_state.category["category"],
            turn_count,
            st.session_state.get("rehearsal_level"),
            st.session_state.participant_data,
            st.session_state.analyzer
        )
        
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages[-1]["phase"] = st.session_state.current_phase
        
        if st.session_state.current_phase == "advice":
            st.session_state.advice_turn_count += 1
            if st.session_state.advice_turn_count >= 2:
                st.session_state.advice_choice_made = False
        else:
            st.session_state.rehearsal_turn_count += 1
            
        st.session_state.awaiting_response = False
        st.rerun()

def render_control_chat_step():
    """Control system chat with time tracking"""
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Your Workplace Scenario")
    
    if not st.session_state.participant_loaded:
        st.error("Participant data not loaded. Please contact the researcher.")
        return
    
    render_scenario_info(st.session_state.participant_data, expandable=True, expanded=False)
    render_section_separator()
    
    # Chat interface
    render_current_step_header(2, "Stage 2: Chat with AI Assistant")
    render_info_message("I'll assist you in exploring your workplace situation and guide you toward the best approach you may take for this situation. You can ask me for advice or request me to roleplay as your boss too.")
    
    if not st.session_state.control_chat_started:
        st.session_state.chat_start_time = datetime.now().isoformat()
        log_event("control_chat_started")
        
        general_next_assistant_turn(st.session_state.client, st.session_state.messages)
        st.session_state.control_chat_started = True

    render_chat_messages(st.session_state.messages)
    render_sentence_starters()

    user_text = render_chat_input("Your message…", "control_chat")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.awaiting_control_response = True
        st.rerun()

    if st.session_state.get("awaiting_control_response", False):
        general_next_assistant_turn(st.session_state.client, st.session_state.messages)
        st.session_state.awaiting_control_response = False
        st.rerun()

    render_end_chat_with_minimum_check(st.session_state.messages, "control")

# ============================================================================
# MAIN ROUTING
# ============================================================================

def route_to_system():
    """Main router"""
    if not st.session_state.authenticated:
        render_home_page()
        return
    
    system_type = st.session_state.get("system_type", "")
    step = st.session_state.get("step", "home")
    
    if system_type == TRUCEY_REHEARSAL:
        if step == "info":
            render_info_gathering_step()
        elif step == "power":
            render_power_dynamic_step()
        elif step == "chat":
            render_trucey_chat_step()
        else:
            render_home_page()
    elif system_type == CONTROL_SYSTEM:
        if step == "control_info":
            render_control_info_step()
        elif step == "control_chat":
            render_control_chat_step()
        else:
            render_control_chat_step()
    else:
        render_home_page()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

init_session_state()
route_to_system()