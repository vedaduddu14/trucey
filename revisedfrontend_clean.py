"""
SoCALM Trucey - Frontend Interface (CLEANED VERSION)
Simplified version with consolidated functions and removed duplications
"""

import streamlit as st 
from datetime import datetime
from revised_backend_CLEAN import (
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
    get_fallback_starters,
    knowledge_test_assistant_turn
)

TRUCEY_REHEARSAL = "trucey_rehearsal"
CONTROL_SYSTEM = "control_system"
KNOWLEDGE_TEST = "knowledge_test"
MIN_INTERACTIONS = 2

# MCQ Questions for leadership assessment
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
        
        # Authentication
        "authenticated": False,
        "login_error": "",
        
        # Time tracking
        "session_start_time": None,
        "chat_start_time": None,
        "advice_start_time": None,
        "rehearsal_start_time": None,
        "timestamps": [],
        
        # User data
        "info": {},
        "messages": [],
        "user_feedback": [],
        "feedback_log": [],
        "participant_data": None,
        "participant_loaded": False,
        
        # Navigation
        "step": "home",
        "system_type": "",
        
        # Leadership assessment
        "leadership_result": False,
        "leader_input": "",
        "leadership_description": "",
        "mcq_answers": ["", "", "", "", ""],
        
        # Conversation state
        "category": None,
        "rehearsal_level": 3,
        "scenario_loaded": False,
        "chat_started": False,
        "control_chat_started": False,
        "current_phase": "advice",
        "advice_turn_count": 0,
        "rehearsal_turn_count": 0,
        "advice_traits": [],
        "rehearsal_traits": [],
        "advice_choice_made": False,
        "waiting_for_feedback": False,
        "chat_ended": False,
        "needs_ai_response": False, 
        "knowledge_test_started": False,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_event(event_name, additional_data=None):
    """Log a timestamped event"""
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
    """Calculate duration between two timestamps"""
    if not start_time:
        return None
    
    if end_time is None:
        end_time = datetime.now().isoformat()
    
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)
    
    duration = end - start
    return duration.total_seconds()

def count_user_messages(messages, phase=None):
    """Count user messages, optionally filtered by phase"""
    if phase:
        return len([msg for msg in messages if msg.get("role") == "user" and msg.get("phase") == phase])
    return len([msg for msg in messages if msg.get("role") == "user"])

def render_scenario_info(info, expandable=True, use_sidebar=False):
    """Render scenario information"""
    scenario_content = f"""
    **Your Situation:**
    - **Problem to discuss:** {info['topic']['description']}
    - **Person you're talking to:** {info['individual']['description']}
    - **Your relationship:** {info['relationship']['description']}
    - **Previous discussions:** {info['previous_interaction']['description']}
    - **Work context:** {info['work_context']['description']}
    """
    
    if use_sidebar:
        if expandable:
            with st.sidebar.expander("Review Your Scenario", expanded=False):
                st.sidebar.markdown(scenario_content)
        else:
            st.sidebar.markdown(scenario_content)
    else:
        if expandable:
            with st.expander("View Your Assigned Scenario", expanded=False):
                st.markdown(scenario_content)
        else:
            st.markdown(scenario_content)

# ============================================================================
# CONSOLIDATED CHAT FUNCTIONS
# ============================================================================

def render_sentence_starters():
    """Show sentence starters if appropriate"""
    should_show = (
        st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "assistant" and
        not st.session_state.get("waiting_for_feedback", False) and
        not st.session_state.get("chat_ended", False) and
        not st.session_state.get("selected_starter")  # FIXED: Hide starters once one is clicked
    )
    
    if should_show:
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
            st.markdown("**Try asking about:**")
            cols = st.columns(len(starters))
            for i, starter in enumerate(starters):
                with cols[i]:
                    display_text = starter + " ..."
                    
                    st.button(display_text, 
                            key=f"starter_{i}_{len(st.session_state.messages)}", 
                            use_container_width=True,
                            help=starter)


def render_chat_input():
    """Unified chat input function"""
    # Don't show input if chat is complete or ended
    if st.session_state.get("chat_ended", False):
        return
        
    # Don't show input if waiting for feedback
    if st.session_state.get("waiting_for_feedback", False):
        return
    
    # FIXED: Don't show input if generating AI response
    if st.session_state.get("needs_ai_response", False):
        st.info("*Generating response...*")
        return
    
    # Check conversation limits (only for Trucey system)
    if st.session_state.get("system_type") == TRUCEY_REHEARSAL:
        if st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 5:
            return
        if st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7:
            return
    
    # Show any stored error messages
    if st.session_state.get("message_error"):
        st.error(st.session_state.message_error)
    
    # Get starter text if available
    starter_text = st.session_state.get("selected_starter", "")
    
    # Show input form
    with st.form(key=f"message_form_{len(st.session_state.messages)}", clear_on_submit=True):
        user_text = st.text_area(
            "Your message:",
            value=starter_text,  # Pre-fill with starter text
            placeholder="Type your message or click a starter above...",
            height=100,
            key=f"user_input_{len(st.session_state.messages)}"
        )
        
        send_clicked = st.form_submit_button("Send", use_container_width=True)
        
        if send_clicked and user_text.strip():
            # Clear the selected starter after sending
            if "selected_starter" in st.session_state:
                del st.session_state.selected_starter
            send_message(user_text)

def send_message(user_text):
    """Handle sending a user message"""
    
    if len(user_text.strip()) < 120:
        # Store error message in session state so it persists after rerun
        st.session_state.message_error = f"Your message is too short. Please write a longer message for a more quality response"
        st.rerun()
        return

    # Clear any error message
    if "message_error" in st.session_state:
        del st.session_state.message_error

    # Clear any selected starter
    if "selected_starter" in st.session_state:
        del st.session_state.selected_starter
        
    # Add message to conversation
    user_message = {"role": "user", "content": user_text, "phase": st.session_state.current_phase}
    st.session_state.messages.append(user_message)
    
    # Set flag to indicate we need AI response
    st.session_state.needs_ai_response = True
    
    # Check if we need feedback (rehearsal only)
    if (st.session_state.current_phase == "rehearsal" and 
        st.session_state.rehearsal_turn_count > 0 and 
        st.session_state.rehearsal_turn_count % 2 == 0 and
        st.session_state.rehearsal_turn_count < 7):
        st.session_state.waiting_for_feedback = True
        st.session_state.needs_ai_response = False  # Don't generate response if waiting for feedback
    
    st.rerun()

def render_end_chat_button():
    """Show end chat button when minimum interactions are met"""
    if st.session_state.get("chat_ended", False):
        return
        
    current_phase = st.session_state.current_phase
    
    # Count user messages based on system type and phase
    if current_phase == "advice":
        user_count = count_user_messages(st.session_state.messages, "advice")
    elif current_phase == "rehearsal":
        user_count = count_user_messages(st.session_state.messages, "rehearsal")
    else:  # Control system or other
        user_count = count_user_messages(st.session_state.messages)  # Count all user messages
    
    if user_count >= MIN_INTERACTIONS:
        st.markdown("---")
        if current_phase in ["advice", "rehearsal"]:
            st.markdown(f"*You've sent {user_count} messages in {current_phase} phase. You can end the conversation when you feel ready, or continue practicing.*")
        else:
            st.markdown(f"*You've sent {user_count} messages. You can end the conversation when you feel ready, or continue practicing.*")
        
        if st.button("End Chat and Save Conversation", key=f"end_chat_{current_phase}", use_container_width=True):
            end_chat_and_save()

def end_chat_and_save():
    """Handle ending chat and saving - SINGLE FUNCTION"""
    st.session_state.chat_ended = True
    
    # Calculate durations
    session_duration = calculate_duration(st.session_state.get("session_start_time"))
    chat_duration = calculate_duration(st.session_state.get("chat_start_time"))
    
    # Log event
    log_event("session_ended", {
        "total_session_duration": session_duration,
        "chat_duration": chat_duration,
        "total_messages": len(st.session_state.messages),
        "user_messages": count_user_messages(st.session_state.messages),
        "phase_ended": st.session_state.current_phase
    })
    
    # Get prolific ID for filename
    prolific_id = "unknown_user"
    if st.session_state.get("info") and st.session_state.info.get("_source"):
        prolific_id = st.session_state.info["_source"].get("prolific_id", "unknown_user")
    
    # Save conversation
    filename = save_chat_locally(
        info=st.session_state.get("info", {}),
        category=st.session_state.get("category", {}),
        language_analysis=st.session_state.get("language_analysis", {}),
        messages=st.session_state.get("messages", []),
        advisor_traits=st.session_state.get("advice_traits", []) + st.session_state.get("rehearsal_traits", []),
        scenario_type=st.session_state.get("system_type", "unknown"),
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
        participant_data=st.session_state.get("participant_data", {}),
        prolific_id=prolific_id
    )
    
    st.rerun()

def show_completion_screen():
    """Show the final completion screen"""
    st.success("🎉 Conversation practice complete! Thank you for participating.")
    st.markdown("### Thank you for participating!")
    st.markdown("**Please click the link below to complete the study:**")
    st.link_button("Complete Study Survey", "https://illinois.qualtrics.com/jfe/form/SV_20mTWTy5iUWn6qW")

def render_chat_messages(messages):
    """Render chat messages"""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

def render_feedback_form(turn_count):
    """Render feedback form for rehearsal"""
    st.markdown("**How was that response?** Any suggestions for better tone, realism, or phrasing? (Optional - leave blank if none, then Submit)")
    feedback_input = st.text_area(
        "Your feedback (optional):",
        key=f"feedback_text_{turn_count}"
    )
    if st.button("Submit Feedback", key=f"feedback_submit_{turn_count}", use_container_width=True):
        return feedback_input.strip() if feedback_input.strip() else ""
    return None

# ============================================================================
# MAIN RENDER FUNCTIONS
# ============================================================================

def render_home_page():
    """Login page"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    st.markdown("### Participant Login")
    st.info("Please enter your Prolific ID to begin the study. We thank you for your participation!")
    
    if st.session_state.get("login_error"):
        st.error(st.session_state.login_error)
    
    with st.form("login_form"):
        prolific_id = st.text_input("Prolific ID:", placeholder="e.g., trial_user_1")
        login_btn = st.form_submit_button("Login", use_container_width=True)
        
        if login_btn:
            participant_data = authenticate_participant(prolific_id)
            
            if participant_data:
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

                # Set appropriate step
                if participant_data['assigned_system'] == TRUCEY_REHEARSAL:
                    st.session_state.step = "info"
                elif participant_data['assigned_system'] == CONTROL_SYSTEM:
                    st.session_state.step = "control_info"
                    st.session_state.current_phase = "control"
                elif participant_data['assigned_system'] == KNOWLEDGE_TEST: 
                    st.session_state.step = "knowledge_test"
                    st.session_state.current_phase = "knowledge"
                
                st.rerun()
            else:
                st.session_state.login_error = "Unable to assign participant ID . Please try again in a minute or contact support."

def render_info_step():
    """Step 1: Show scenario info"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    st.markdown("### Stage 1: Understanding Your Situation")
    st.info("""
    **What happens in this stage:**
    - We've loaded details of your assigned workplace scenario
    - Review the details and confirm you understand the scenario
    - This information will be used to customize your conversation practice
    """)
    
    render_scenario_info(st.session_state.info, expandable=False)
    
    st.markdown("---")
    st.markdown("**All set!** This information will help us provide personalized guidance for your conversation.")
    
    if st.button("I've reviewed my information - Continue", use_container_width=True):
        st.session_state.step = "power"
        st.rerun()

def render_power_step():
    """Step 2: Leadership assessment"""
    st.title("SoCALM's Trucey")
    st.markdown("---")
    
    st.markdown("### Stage 2: Analyzing Leadership Style")
    st.info("""
    **What happens in this stage:**
    - You'll answer 5 questions about your boss's personality and leadership approach
    - The system will match their style to provide tailored advice
    - This helps us customize the conversation guidance for your specific situation
    """)
    
    if not st.session_state.leadership_result:
        st.subheader("Describe your boss's nature and leadership style")
        st.markdown("**Rate how well each statement describes your boss:**")
        st.markdown("**Rating Scale:** 1 = Strongly Disagree | 2 = Disagree | 3 = Neutral | 4 = Agree | 5 = Strongly Agree")

        with st.form("leadership_form"):
            mcq_answers = []
            
            for i, qdict in enumerate(MCQ_QUESTIONS):
                st.markdown(f"**Q{i+1}. {qdict['q']}**")
                
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
                
                mcq_answers.append(rating)
                if i < len(MCQ_QUESTIONS) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            leader_input = st.text_area(
                "**Additional Description (Optional):** In 2-3 sentences, describe this person's leadership style:",
                value=st.session_state.leader_input
            )
            
            submit_leadership = st.form_submit_button("Analyze Leadership Style")
            
            if submit_leadership:
                st.session_state.mcq_answers = mcq_answers
                st.session_state.leader_input = leader_input
                
                leadership = get_power_dynamic_model(user_description=leader_input, mcq_answers=mcq_answers)
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
        st.markdown(f"**Your description:** {st.session_state.leader_input}")
        st.markdown(st.session_state.leadership_description)
        
        if st.button("Continue to Next Step", use_container_width=True):
            with st.spinner("Preparing your conversation..."):
                load_conversation_data()
            st.session_state.step = "chat"
            st.rerun()

def load_conversation_data():
    """Load conversation data in background"""
    if st.session_state.category is None:
        st.session_state.category = categorize_situation(st.session_state.info)

    if st.session_state.category and "category" in st.session_state.category:
        original_category = st.session_state.category["category"]
        cleaned_category = original_category.split(' (')[0].strip()
        st.session_state.category["category"] = cleaned_category
    
    if not st.session_state.get("scenario_loaded", False):
        combined_data = load_combined_scenario_data(
            st.session_state.category["category"],
            st.session_state.get("rehearsal_level")
        )
        
        st.session_state.advice_traits = combined_data["advice_traits"]
        st.session_state.rehearsal_traits = combined_data["rehearsal_traits"]
        st.session_state.scenario_loaded = True

def render_trucey_chat():
    """Main Trucey conversation interface"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    # Show completion screen if chat ended
    if st.session_state.get("chat_ended", False):
        show_completion_screen()
        return
    
    st.markdown("### Stage 3: Conversation Practice")
    st.info("""
    **What happens in this stage:**
    - **Phase 1 - Advice (5 responses):** Get strategic guidance for your situation
    - **Phase 2 - Rehearsal (7 responses):** Practice the conversation with me role-playing as your boss
    """)
    st.markdown("---")
    
    # Initialize chat if not started
    if not st.session_state.chat_started:
        st.session_state.chat_start_time = datetime.now().isoformat()
        st.session_state.advice_start_time = datetime.now().isoformat()
        
        get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            st.session_state.advice_traits,
            "advice",
            st.session_state.category["category"],
            st.session_state.advice_turn_count,
            st.session_state.get("rehearsal_level")
        )
        
        if st.session_state.messages:
            st.session_state.messages[-1]["phase"] = "advice"
        
        st.session_state.chat_started = True
        st.session_state.advice_turn_count += 1
    
    # Phase headers and message display
    if st.session_state.current_phase == "advice":
        st.subheader(f"Phase 1: Getting Advice ({st.session_state.advice_turn_count}/5 responses)")
        render_chat_messages(st.session_state.messages)
    else:
        st.subheader("Phase 1: Advice (Complete)")
        advice_messages = [msg for msg in st.session_state.messages if msg.get("phase", "advice") == "advice"]
        render_chat_messages(advice_messages)
        
        st.markdown("---")
        st.subheader(f"Phase 2: Rehearsal ({st.session_state.rehearsal_turn_count}/7 responses)")
        st.warning("**Rehearsal Mode:** I'm now role-playing as your boss. Practice the conversation!")
        
        rehearsal_messages = [msg for msg in st.session_state.messages if msg.get("phase") == "rehearsal"]
        render_chat_messages(rehearsal_messages)
    
    # Handle advice phase completion
    if st.session_state.current_phase == "advice" and st.session_state.advice_turn_count >= 2:
        if st.session_state.advice_turn_count >= 5:
            st.info("Great! Now let's practice this conversation.")
            if st.button("Start Rehearsal Practice", use_container_width=True):
                transition_to_rehearsal()
                st.rerun()
            return
        else:
            if not st.session_state.get("advice_choice_made", False):
                st.info(f"You've received {st.session_state.advice_turn_count} pieces of advice. What would you like to do next?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Start Rehearsal Practice Now", use_container_width=True):
                        transition_to_rehearsal()
                        st.rerun()
                with col2:
                    if st.button("Continue Getting Advice", use_container_width=True):
                        st.session_state.advice_choice_made = True
                        st.rerun()
                return
    
    # Handle rehearsal completion
    if st.session_state.current_phase == "rehearsal" and st.session_state.rehearsal_turn_count >= 7:
        st.success("Conversation practice complete!")
        end_chat_and_save()
        return
    
    # Handle feedback collection
    if st.session_state.get("waiting_for_feedback", False):
        feedback = render_feedback_form(st.session_state.rehearsal_turn_count)
        if feedback is not None:
            if feedback:
                st.session_state.user_feedback.append(feedback)
                st.session_state.feedback_log.append({
                    "content": feedback,
                    "turn_count": st.session_state.rehearsal_turn_count,
                    "timestamp": datetime.now().isoformat(),
                })
                st.success("Feedback received!")
            
            st.session_state.waiting_for_feedback = False
            st.session_state.needs_ai_response = True  # FIXED: Set flag to generate AI response
            st.rerun()
        return
    
    # Show starters and input
    render_sentence_starters()
    render_chat_input()
    render_end_chat_button()
    
    # Handle AI response generation - FIXED: Use flag to prevent double rendering
    if st.session_state.get("needs_ai_response", False):
        # Clear the flag first to prevent re-triggering
        st.session_state.needs_ai_response = False
        
        # Determine current traits and phase
        if st.session_state.current_phase == "advice":
            current_traits = st.session_state.advice_traits
            scenario_type = "advice"
            turn_count = st.session_state.advice_turn_count
        else:
            current_traits = st.session_state.rehearsal_traits
            scenario_type = "rehearsal"
            turn_count = st.session_state.rehearsal_turn_count
        
        # Generate AI response
        get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            current_traits,
            scenario_type,
            st.session_state.category["category"],
            turn_count,
            st.session_state.get("rehearsal_level")
        )
        
        # Set phase and update counters
        if st.session_state.messages:
            st.session_state.messages[-1]["phase"] = st.session_state.current_phase
        
        if st.session_state.current_phase == "advice":
            st.session_state.advice_turn_count += 1
            if st.session_state.advice_turn_count >= 2:
                st.session_state.advice_choice_made = False
        else:
            st.session_state.rehearsal_turn_count += 1
        
        st.rerun()

def transition_to_rehearsal():
    """Handle transition from advice to rehearsal phase"""
    st.session_state.rehearsal_start_time = datetime.now().isoformat()
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
        st.session_state.get("rehearsal_level")
    )
    
    if st.session_state.messages:
        st.session_state.messages[-1]["phase"] = "rehearsal"
    
    st.session_state.rehearsal_turn_count += 1
    # FIXED: Don't set needs_ai_response since we just generated the response

def render_control_info():
    """Control system info page"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    st.markdown("### Stage 1: Understanding Your Situation")
    st.info("""
    **What happens next:**
    - We'll show you the workplace scenario you'll be working through 
    - You'll then chat directly with an AI assistant for guidance
    """)
    
    st.markdown("### Your Assigned Scenario")
    render_scenario_info(st.session_state.info, expandable=False)
    
    st.markdown("---")
    if st.button("I understand my scenario - Start Chat", use_container_width=True):
        st.session_state.step = "control_chat"
        st.rerun()

def render_control_chat():
    """Control system chat"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    # Show completion screen if chat ended
    if st.session_state.get("chat_ended", False):
        show_completion_screen()
        return
    
    st.markdown("### Stage 2: Chat with AI Assistant")
    st.info("I'll assist you in exploring your workplace situation and guide you toward the best approach.")
    
    if not st.session_state.control_chat_started:
        st.session_state.chat_start_time = datetime.now().isoformat()
        log_event("control_chat_started")
        general_next_assistant_turn(st.session_state.client, st.session_state.messages, st.session_state.info)
        st.session_state.control_chat_started = True

    render_chat_messages(st.session_state.messages)
    render_sentence_starters()
    render_chat_input()
    render_end_chat_button()
    
    # Handle AI response generation for control - FIXED: Use flag to prevent double rendering
    if st.session_state.get("needs_ai_response", False):
        st.session_state.needs_ai_response = False
        general_next_assistant_turn(st.session_state.client, st.session_state.messages, st.session_state.info)
        st.rerun()

def render_knowledge_test():
    """Knowledge test system - direct ChatGPT-like interface"""
    st.title("SoCALM User Study")
    st.markdown("---")
    
    # Show completion screen if chat ended
    if st.session_state.get("chat_ended", False):
        show_completion_screen()
        return
    
    st.markdown("### AI Chat Assistant")
    st.info("I'm here to help you with any workplace communication questions or situations you'd like to discuss.")
    
    # Initialize chat if not started
    if not st.session_state.get("knowledge_test_started", False):
        st.session_state.chat_start_time = datetime.now().isoformat()
        log_event("knowledge_test_started")
        
        welcome_message = {
            "role": "assistant", 
            "content": "Hello! I'm an AI assistant who is pretending to be your boss. Try negotiating with me about a workplace situation.",
            "phase": "knowledge"
        }
        st.session_state.messages.append(welcome_message)
        st.session_state.knowledge_test_started = True

    render_chat_messages(st.session_state.messages)
    render_chat_input()
    render_end_chat_button()
    
    if st.session_state.get("needs_ai_response", False):
        st.session_state.needs_ai_response = False
        knowledge_test_assistant_turn(st.session_state.client, st.session_state.messages)
        st.rerun()


def render_sidebar():
    """Render sidebar with scenario info"""
    current_step = st.session_state.get("step", "home")
    
    if current_step in ["power", "chat", "control_chat"] and st.session_state.participant_loaded:
        st.sidebar.markdown("---")
        render_scenario_info(st.session_state.info, expandable=True, use_sidebar=True)
        st.sidebar.markdown("---")
        
        # Show leadership style for Trucey system in chat phase
        if (st.session_state.system_type == TRUCEY_REHEARSAL and
            current_step == "chat" and 
            st.session_state.get("leadership_result") and 
            st.session_state.get("leadership_description")):
            
            with st.sidebar.expander("Boss's Leadership Style", expanded=False):
                leadership_desc = st.session_state.leadership_description
                
                # Parse the leadership description to extract key information
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
                    st.sidebar.markdown(f"**Leadership Style:** {style_name}")
                
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

def main():
    """Main application"""
    # Show study info if authenticated
    if st.session_state.get("authenticated"):
        st.info("""
        **Study Objective:** Practice having a workplace conversation with your boss using AI guidance.
        **Expected Time:** 5-10 minutes | **Your responses are confidential**
                
        **If there is any problem during the study, please use the contact present on the prolific post**
        """)
        st.markdown("---")
        render_sidebar()
    
    # Route to appropriate page
    if not st.session_state.get("authenticated"):
        render_home_page()
    else:
        system_type = st.session_state.get("system_type", "")
        step = st.session_state.get("step", "home")
        
        if system_type == TRUCEY_REHEARSAL:
            if step == "info":
                render_info_step()
            elif step == "power":
                render_power_step()
            elif step == "chat":
                render_trucey_chat()
        elif system_type == CONTROL_SYSTEM:
            if step == "control_info":
                render_control_info()
            elif step == "control_chat":
                render_control_chat()
        elif system_type == KNOWLEDGE_TEST: 
            render_knowledge_test()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    init_session_state()
    main()