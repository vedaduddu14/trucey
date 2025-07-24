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
    convert_sqlite_to_info_format,
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
    """Initialize all session state variables - improved version"""
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
        "current_input": "",
        
        # FIXED: Better starter and input state management
        "starter_clicked": False,
        "input_populated": False,
        "message_starter": None,
        "message_sent": False,  # NEW: Track if message was sent
    }
    
    # Initialize all at once to reduce multiple re-renders
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
def render_scenario_info(info, expandable=True, expanded=False, use_sidebar=False):
    """
    Unified function to render scenario information from structured info
    Args:
        info: Structured info dictionary
        expandable: Whether to use an expander for the scenario content
        expanded: Initial state of the expander
        use_sidebar: Whether to render in sidebar (uses st.sidebar functions)
    """
    scenario_content = f"""
    **Your Situation:**
    - **Problem to discuss:** {info['topic']['description']}
    - **Person you're talking to:** {info['individual']['description']}
    - **Your relationship:** {info['relationship']['description']}
    - **Previous discussions:** {info['previous_interaction']['description']}
    - **Work context:** {info['work_context']['description']}
    """
    
    # Choose the right Streamlit functions based on location
    if use_sidebar:
        expander_func = st.sidebar.expander
        markdown_func = st.sidebar.markdown
    else:
        expander_func = st.expander
        markdown_func = st.markdown
    
    if expandable:
        expander_title = "Review Your Scenario" if use_sidebar else "View Your Assigned Scenario"
        with expander_func(expander_title, expanded=expanded):
            markdown_func(scenario_content)
    else:
        markdown_func(scenario_content)

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
            
            # ADD SURVEY LINK HERE TOO:
            st.markdown("### Thank you for participating!")
            st.markdown("**Please click the link below to complete the study:**")
            st.link_button("Complete Study Survey", "https://illinois.qualtrics.com/jfe/form/SV_20mTWTy5iUWn6qW")
            return True

def render_sentence_starters():
    """
    Fixed version - starters disappear when clicked and populate text box
    """
    # Check if we should show starters
    should_show_starters = (
        st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "assistant" and
        not st.session_state.get("waiting_for_feedback", False) and
        not st.session_state.get("starter_clicked", False) and  # Hide if starter was clicked
        not st.session_state.get("awaiting_response", False) and
        not st.session_state.get("message_sent", False)  # Hide if message was sent
    )
    
    if should_show_starters:
        starter_key = f"starters_{len(st.session_state.messages)}"
        
        # Generate starters if not cached
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
                    st.session_state.info,
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
                        # Set the starter text and mark as clicked
                        st.session_state.message_starter = starter
                        st.session_state.starter_clicked = True
                        st.session_state.input_populated = False
                        # Force immediate rerun to hide starters
                        st.rerun()

def handle_phase_transition():
    """
    Handle transition from advice to rehearsal phase - reset UI states
    """
    st.session_state.update({
        'current_phase': "rehearsal",
        'advice_choice_made': True,
        'starter_clicked': False,  # Reset starters for new phase
        'input_populated': False,
        'message_sent': False  # Reset message sent state
    })
    
    get_next_assistant_turn(
        st.session_state.client,
        st.session_state.messages,
        st.session_state.info,
        st.session_state.rehearsal_traits,
        "rehearsal",
        st.session_state.category["category"],
        st.session_state.rehearsal_turn_count,
        st.session_state.get("rehearsal_level"),
        st.session_state.analyzer
    )
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages[-1]["phase"] = "rehearsal"
    
    st.session_state.rehearsal_turn_count += 1


def render_chat_input_unified():
    """
    Fixed chat input handling with proper state management
    """
    conversation_complete = ((st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 5) or
                           (st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7))
    
    # Create placeholders for dynamic content
    input_placeholder = st.empty()
    status_placeholder = st.empty()
    end_chat_placeholder = st.empty()

    # Show input section only if conversation not complete and not awaiting response
    if (not conversation_complete and 
        not st.session_state.get("awaiting_response", False) and
        not st.session_state.get("waiting_for_feedback", False) and
        not st.session_state.get("message_sent", False)):  # Hide input after sending

        with input_placeholder.container():
            # Handle starter text population
            current_input_value = ""
            
            if (st.session_state.get("message_starter") and 
                st.session_state.get("starter_clicked", False) and 
                not st.session_state.get("input_populated", False)):
                current_input_value = st.session_state.message_starter
                st.session_state.current_input = current_input_value
                st.session_state.input_populated = True
                # Clear the starter from session state
                if "message_starter" in st.session_state:
                    del st.session_state.message_starter
            else:
                current_input_value = st.session_state.get("current_input", "")

            text_area_key = f"user_input_{len(st.session_state.messages)}_{st.session_state.get('message_sent', False)}"
            user_text = st.text_area(
                "Your message:", 
                value= text_area_key, 
                key="user_input_direct",
                placeholder="Type your message or click a starter above...",
                height=100,
                label_visibility="visible"
            )

            # Update current input state
            if user_text != st.session_state.get("current_input", ""):
                st.session_state.current_input = user_text

            if st.button("Send", key="send_btn", use_container_width=True) and user_text.strip():
                # Clear input and set states
                st.session_state.update({
                    'current_input': "",
                    'starter_clicked': False,
                    'input_populated': False,
                    'message_sent': True,  # Mark message as sent to hide input
                    'awaiting_response': True
                })
                user_message = {"role": "user", "content": user_text, "phase": st.session_state.current_phase}
                st.session_state.messages.append(user_message)
                
                if (st.session_state.current_phase == "rehearsal" and 
                    st.session_state.rehearsal_turn_count > 0 and 
                    st.session_state.rehearsal_turn_count % 2 == 0 and
                    st.session_state.rehearsal_turn_count < 7):
                    st.session_state.waiting_for_feedback = True
                    st.session_state.awaiting_response = False
                
                st.rerun()

    # Show loading indicator when awaiting response
    elif st.session_state.get("awaiting_response", False):
        with status_placeholder.container():
            st.info("*Generating response...*")
    
    # Show end chat button AFTER the input section (when minimum interactions met)
    if (st.session_state.current_phase == "rehearsal" and 
        not st.session_state.get("waiting_for_feedback", False)):
        
        user_count = count_user_messages(st.session_state.messages, "rehearsal") 
        if user_count >= MIN_INTERACTIONS:
            with end_chat_placeholder.container():
                st.markdown("---")
                st.markdown("*You can end the conversation when you feel ready, or continue practicing.*")
                if st.button("End Chat and Save Conversation", key="end_chat_after_input", use_container_width=True):
                    # Handle end chat logic
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
                        scenario_type="trucey_rehearsal",
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
                    
                    st.success(f"Chat saved to `{filename}`")
                    st.success("Chat saved. You can now close this tab or stop the server.")
                    
                    st.markdown("### Thank you for participating!")
                    st.markdown("**Please click the link below to complete the study:**")
                    st.link_button("Complete Study Survey", "https://illinois.qualtrics.com/jfe/form/SV_20mTWTy5iUWn6qW")


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
    st.markdown("**How was that response?** Any suggestions for better tone, realism, or phrasing? (Optional - leave blank if none, then Submit)")
    feedback_input = st.text_area(
        "Your feedback (optional):",
        key=f"feedback_text_{turn_count}"
    )
    if st.button("Submit Feedback", key=f"feedback_submit_{turn_count}", use_container_width=True):
        return feedback_input.strip() if feedback_input.strip() else ""
    return None

def handle_end_chat_save():
    """Handle the end chat and save functionality"""
    session_duration = calculate_duration(st.session_state.get("session_start_time"))
    chat_duration = calculate_duration(st.session_state.get("chat_start_time"))
    
    log_event("session_ended", {
        "total_session_duration": session_duration,
        "chat_duration": chat_duration,
        "total_messages": len(st.session_state.messages),
        "user_messages": count_user_messages(st.session_state.messages),
        "phase_ended": st.session_state.current_phase
    })
    
    filename = save_chat_locally(
        info=st.session_state.get("info", {}),
        category=st.session_state.get("category", {}),
        language_analysis=st.session_state.get("language_analysis", {}),
        messages=st.session_state.get("messages", []),
        advisor_traits=st.session_state.get("advice_traits", []) + st.session_state.get("rehearsal_traits", []),
        scenario_type="trucey_rehearsal",
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
    
    st.success(f"Chat saved to `{filename}`")
    st.success("Chat saved. You can now close this tab or stop the server.")
    
    st.markdown("### Thank you for participating!")
    st.markdown("**Please click the link below to complete the study:**")
    st.link_button("Complete Study Survey", "https://illinois.qualtrics.com/jfe/form/SV_20mTWTy5iUWn6qW")

# ============================================================================
# MAIN RENDER FUNCTIONS
# ============================================================================
def render_home_page():
    """
    Fixed login page - reduces re-rendering
    """
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Participant Login")
    st.info("Please enter your Prolific ID to begin the study. We thank you for your participation!")
    
    if st.session_state.login_error:
        st.error(st.session_state.login_error)
    
    with st.form("login_form"):
        prolific_id = st.text_input("Prolific ID:", placeholder="e.g., trial_user_1")
        login_btn = st.form_submit_button("Login", use_container_width=True)
        
        if login_btn:
            st.session_state.login_attempted = True
            st.session_state.login_error = ""
            
            participant_data = authenticate_participant(prolific_id)
            
            if participant_data:
                # Set all session state at once to reduce re-renders
                st.session_state.update({
                    'session_start_time': datetime.now().isoformat(),
                    'info': convert_sqlite_to_info_format(participant_data),
                    'authenticated': True,
                    'participant_loaded': True,
                    'system_type': participant_data['assigned_system'],
                    'messages': []
                })
                
                log_event("session_started", {
                    "prolific_id": prolific_id,
                    "assigned_system": participant_data['assigned_system']
                })
                
                log_event("participant_info_loaded", {
                    "problem": participant_data['assigned_problem'],
                    "person_of_interest": participant_data['person_of_interest'],
                    "relationship_quality": participant_data['relationship_quality']
                })

                # Set the appropriate step based on system type
                if participant_data['assigned_system'] == TRUCEY_REHEARSAL:
                    st.session_state.step = "info"
                    log_event("trucey_system_entered")
                elif participant_data['assigned_system'] == CONTROL_SYSTEM:
                    st.session_state.step = "control_info"
                    st.session_state.current_phase = "control" 
                    log_event("control_system_entered")
                
                # FIXED: Only call rerun once after all state changes
                st.rerun()
            else:
                st.session_state.login_error = "Invalid participant ID or password. Please try again."
                log_event("login_failed", {"participant_id": prolific_id})
                # Don't call st.rerun() for errors - form will handle it

# ============================================================================
# FIXED CHAT INPUT FUNCTION
# ============================================================================

def render_chat_input_unified():
    """
    Fixed chat input handling with proper end chat button placement
    """
    conversation_complete = ((st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 5) or
                           (st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7))
    
    # Create placeholders for dynamic content
    input_placeholder = st.empty()
    status_placeholder = st.empty()
    end_chat_placeholder = st.empty()

    # Show input section only if conversation not complete and not awaiting response
    if (not conversation_complete and 
        not st.session_state.get("awaiting_response", False) and
        not st.session_state.get("waiting_for_feedback", False) and
        not st.session_state.get("message_sent", False)):

        with input_placeholder.container():
            # Handle starter text population
            current_input_value = ""
            
            if (st.session_state.get("message_starter") and 
                st.session_state.get("starter_clicked", False) and 
                not st.session_state.get("input_populated", False)):
                current_input_value = st.session_state.message_starter
                st.session_state.current_input = current_input_value
                st.session_state.input_populated = True
                if "message_starter" in st.session_state:
                    del st.session_state.message_starter
            else:
                current_input_value = st.session_state.get("current_input", "")

            user_text = st.text_area(
                "Your message:", 
                value=current_input_value,
                key="user_input_direct",
                placeholder="Type your message or click a starter above...",
                height=100,
                label_visibility="visible"
            )

            if user_text != st.session_state.get("current_input", ""):
                st.session_state.current_input = user_text

            if st.button("Send", key="send_btn", use_container_width=True) and user_text.strip():
                st.session_state.update({
                    'current_input': "",
                    'starter_clicked': False,
                    'input_populated': False,
                    'message_sent': True,
                    'awaiting_response': True
                })
                
                user_message = {"role": "user", "content": user_text, "phase": st.session_state.current_phase}
                st.session_state.messages.append(user_message)
                
                if (st.session_state.current_phase == "rehearsal" and 
                    st.session_state.rehearsal_turn_count > 0 and 
                    st.session_state.rehearsal_turn_count % 2 == 0 and
                    st.session_state.rehearsal_turn_count < 7):
                    st.session_state.waiting_for_feedback = True
                    st.session_state.awaiting_response = False
                
                st.rerun()

    # Show loading indicator when awaiting response
    elif st.session_state.get("awaiting_response", False):
        with status_placeholder.container():
            st.info("*Generating response...*")
    
    # FIXED: Show end chat button after minimum interactions for BOTH advice and rehearsal
    current_phase = st.session_state.current_phase
    
    # Check if we should show end chat button
    should_show_end_chat = False
    user_count = 0
    
    if current_phase == "advice":
        user_count = count_user_messages(st.session_state.messages, "advice")
        should_show_end_chat = (user_count >= MIN_INTERACTIONS and 
                               not st.session_state.get("waiting_for_feedback", False) and
                               not conversation_complete)
    elif current_phase == "rehearsal":
        user_count = count_user_messages(st.session_state.messages, "rehearsal") 
        should_show_end_chat = (user_count >= MIN_INTERACTIONS and 
                               not st.session_state.get("waiting_for_feedback", False) and
                               not conversation_complete)
    
    if should_show_end_chat:
        with end_chat_placeholder.container():
            st.markdown("---")
            st.markdown(f"*You've sent {user_count} messages in {current_phase} phase. You can end the conversation when you feel ready, or continue practicing.*")
            if st.button("End Chat and Save Conversation", key=f"end_chat_{current_phase}", use_container_width=True):
                handle_end_chat_save()

def render_info_gathering_step():
    """
    UPDATED: Step 1 using structured info with unified scenario display
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
    
    render_scenario_info(st.session_state.info, expandable=False, use_sidebar=False)
    
    log_event("participant_reviewed_loaded_info")
    
    st.markdown("---")
    st.markdown("**All set!** This information will help us provide personalized guidance for your conversation.")
    
    if render_continue_button("I've reviewed my information - Continue"):
        log_event("stage1_completed")
        st.session_state.step = "power"
        st.rerun()

def render_control_info_step():
    """UPDATED: Control system scenario display using structured info"""
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Stage 1: Understanding Your Situation")
    st.info("""
    **What happens next:**
    - We'll show you the workplace scenario you'll be working through 
    - You'll then chat directly with an AI assistant for guidance
    - The assistant will help you explore your situation however you would like to 
    """)
    
    if not st.session_state.participant_loaded:
        st.error("Participant data not loaded. Please contact the researcher.")
        return
    
    st.markdown("### Your Assigned Scenario")
    
    render_scenario_info(st.session_state.info, expandable=False, use_sidebar=False)
    
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
        st.markdown(""" **Rating Scale:** **1** = Strongly Disagree **2** = Disagree **3** = Neutral **4** = Agree **5** = Strongly Agree""")

        with st.form("leadership_form", clear_on_submit=False):
            mcq_answers = []
            
            for i, qdict in enumerate(MCQ_QUESTIONS):
                # LEFT-ALIGNED question (remove center styling)
                st.markdown(f"**Q{i+1}. {qdict['q']}**")
                
                # CENTERED radio buttons using columns
                col1, col2, col3 = st.columns([1, 2, 1])
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
                
                st.markdown("<br>", unsafe_allow_html=True)
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
    Main Trucey conversation interface - CLEANED UP end chat handling
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
    
    # Initialize chat if not started
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
            st.session_state.analyzer
        )
        
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages[-1]["phase"] = "advice"
        
        st.session_state.chat_started = True
        st.session_state.advice_turn_count += 1

    # Display messages based on current phase
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
    
    # Handle advice phase transitions
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

    # Check if rehearsal is complete (7 responses)
    if st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7:
        st.success("Conversation practice complete! Thank you for participating.")
        handle_end_chat_save()
        return

    # Handle feedback collection
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

    # Render sentence starters BEFORE input
    render_sentence_starters()
    
    # Render the chat input (now includes proper end chat button logic)
    render_chat_input_unified()
    
    # Handle AI response generation
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
        
        # Reset states for next interaction
        st.session_state.update({
            'awaiting_response': False,
            'message_sent': False,
            'starter_clicked': False
        })
        st.rerun()

def render_control_chat_step():
    """Control system chat with time tracking"""
    render_page_header("SoCALM User Study")
    render_section_separator()
    
    st.markdown("### Your Workplace Scenario")
    
    if not st.session_state.participant_loaded:
        st.error("Participant data not loaded. Please contact the researcher.")
        return
    
    # Chat interface
    render_current_step_header(2, "Stage 2: Chat with AI Assistant")
    render_info_message("I'll assist you in exploring your workplace situation and guide you toward the best approach you may take for this situation. You can ask me for advice or request me to roleplay as your boss too.")
    
    if not st.session_state.control_chat_started:
        st.session_state.chat_start_time = datetime.now().isoformat()
        log_event("control_chat_started")
        
        general_next_assistant_turn(st.session_state.client, st.session_state.messages, st.session_state.info)
        st.session_state.control_chat_started = True

    render_chat_messages(st.session_state.messages)
    render_sentence_starters()

    user_text = render_chat_input("Your message…", "control_chat")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.awaiting_control_response = True
        st.rerun()

    if st.session_state.get("awaiting_control_response", False):
        general_next_assistant_turn(st.session_state.client, st.session_state.messages, st.session_state.info)
        st.session_state.awaiting_control_response = False
        st.rerun()

    render_end_chat_with_minimum_check(st.session_state.messages, "control")

def render_sidebar_scenario():
    """Simple sidebar that shows info based on completed stages"""
    
    current_step = st.session_state.get("step", "home")
    
    # If stage 1 passed: show participant details (for BOTH treatment and control)
    if current_step in ["power", "chat", "control_chat"]:
        st.sidebar.markdown("---")
        
        if not st.session_state.participant_loaded:
            st.sidebar.error("Participant data not loaded. Please contact the researcher.")
        else:
           render_scenario_info(st.session_state.info, expandable=False, use_sidebar=True)

        st.sidebar.markdown("---")
        # If stage 2 passed: showcase leadership pros and cons (ONLY for treatment)
        if (st.session_state.system_type == TRUCEY_REHEARSAL and
            current_step in ["chat"] and 
            st.session_state.get("leadership_result") and 
            st.session_state.get("leadership_description")):
            
            with st.sidebar.expander("Boss's Leadership Style", expanded=False):
                if (st.session_state.system_type == TRUCEY_REHEARSAL and
                current_step in ["chat"] and 
                st.session_state.get("leadership_result") and 
                st.session_state.get("leadership_description")):
                
                    with st.sidebar.expander("Boss's Leadership Style", expanded=False):
                        leadership_desc = st.session_state.leadership_description
                        
                        lines = [line.strip() for line in leadership_desc.split('\n') if line.strip()]
                        style_name = ""
                        pros = []
                        cons = []
                        current_section = None
                        
                        for line in lines:
                            if line.startswith("**Based on your answers, the best match is:**"):
                                style_name = line.split(":")[-1].strip()
                            elif "Leadership Style" in line and "---" in line:
                                if "**" in line:
                                    parts = line.split("**")
                                    if len(parts) >= 3: 
                                        style_name = parts[1].strip()
                            elif line == "**Strengths:**":
                                current_section = "strengths"
                            elif line == "**Challenges:**":
                                current_section = "challenges"
                            elif line.startswith("- ") and current_section:
                                item = line[2:].strip()  
                                if current_section == "strengths":
                                    pros.append(item)
                                elif current_section == "challenges":
                                    cons.append(item)
                        
                        # Display the cleaned info
                        if style_name:
                            st.sidebar.markdown(f"**Assigned Leadership Style:** {style_name}")
                        
                        if pros:
                            st.sidebar.markdown("**Key Strengths:**")
                            for pro in pros:
                                st.sidebar.markdown(f"• {pro}")
                        
                        if cons:
                            st.sidebar.markdown("**Key Challenges:**")
                            for con in cons:
                                st.sidebar.markdown(f"• {con}")
        
        st.sidebar.markdown("---")
# ============================================================================
# MAIN ROUTING
# ============================================================================

def route_to_system():
    """Main router"""
    if st.session_state.authenticated:
        st.info("""
        **Study Objective:** Practice having a workplace conversation with your boss using AI guidance. Any information collected during previous stages will be available in the sidebar.
        
        **Please Contact the Study Organizers in case of any difficulty**
        
        **Expected Time:** 5-10 minutes | **Your responses are confidential**
        """)
        st.markdown("---")
        render_sidebar_scenario()
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