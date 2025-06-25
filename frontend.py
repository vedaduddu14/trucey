import streamlit as st
from datetime import datetime
from backend import (
    required_information,
    LanguageStyleAnalyzer,
    get_openai_client,
    analyze_and_update_info,
    get_power_dynamic_model, 
    update_info_with_power_dynamic,
    categorize_situation,
    load_scenario_data,
    extract_advisor_traits,
    get_next_assistant_turn,
    summarize_user_feedback,
    save_chat_locally
)

MCQ_QUESTIONS = [
    {
        "q": "How do they prefer to approach new ideas or projects?",
        "options": {
            "A": "They generate tons of creative possibilities and imagine how the future could look",
            "B": "They enjoy thinking outside the box but prefer some structure",
            "C": "They prefer proven approaches that work well"
        }
    },
    {
        "q": "In group settings, how do they typically behave?",
        "options": {
            "A": "They take the lead, energize others, and enjoy being in the spotlight",
            "B": "They are social and approachable but don’t always seek attention",
            "C": "They tend to keep to myself and only speak up when needed"
        }
    },
    {
        "q": "How important is harmony and maintaining good relationships to them at work?",
        "options": {
            "A": "Extremely important— they often go out of their way to support others",
            "B": "They try to be collaborative but can assert their views when needed",
            "C": "They prioritize results over relationships if needed"
        }
    },
    {
        "q": "How do they respond to uncertainty or last-minute changes?",
        "options": {
            "A": "They stay calm and adapt easily",
            "B": "They manage but prefer predictability",
            "C": "It stresses them out and they prefer sticking to the plan"
        }
    },
    {
        "q": "How do they usually handle feedback or critique?",
        "options": {
            "A": "They take it constructively and reflect deeply",
            "B": "They can take it but feel emotionally impacted",
            "C": "They get defensive or shut down"
        }
    },
]

st.set_page_config(page_title="Workplace Conversation (Steps 1–4)", layout="wide")

# ─── 1. Initialize session state ───────────────────────────────
if "info" not in st.session_state:
    st.session_state.info = required_information()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Can you tell me a little about the trouble you're facing at work and require some support for?"}
    ]
if "step" not in st.session_state:
    st.session_state.step = "info"
if "analyzer" not in st.session_state:
    st.session_state.analyzer = LanguageStyleAnalyzer()
if "client" not in st.session_state:
    st.session_state.client = get_openai_client()
if "next_question" not in st.session_state:
    st.session_state.next_question = ""
if "awaiting_info_update" not in st.session_state:
    st.session_state.awaiting_info_update = False
if "leadership_result" not in st.session_state:
    st.session_state.leadership_result = False
if "leader_input" not in st.session_state:
    st.session_state.leader_input = ""
if "leadership_description" not in st.session_state:
    st.session_state.leadership_description = ""
if "category" not in st.session_state:
    st.session_state.category = None
if "scenario_type" not in st.session_state:
    st.session_state.scenario_type = ""
if "traits" not in st.session_state:
    st.session_state.traits = []
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "responses" not in st.session_state:
    st.session_state.responses = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "mcq_answers" not in st.session_state:
    st.session_state.mcq_answers = ["", "", "", "", ""]
if "mcq_submitted" not in st.session_state:
    st.session_state.mcq_submitted = False

# ─── 2. Render full chat history ───────────────────────────────
st.title("SoCALM's Trucey")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── 3. STEP 1: Gather Required Information ───────────────────
if st.session_state.step == "info":
    user_text = st.chat_input("Your response…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.awaiting_info_update = True
        st.rerun()

    if st.session_state.awaiting_info_update:
        last_user_msg = st.session_state.messages[-1]["content"]
        style_scores = st.session_state.analyzer.analyze_text(last_user_msg)
        st.session_state.info["language_style"].update(style_scores)

        updated_info, next_q = analyze_and_update_info(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info
        )
        st.session_state.info = updated_info
        st.session_state.next_question = next_q

        for field, meta in st.session_state.info.items():
            if field != "language_style" and meta.get("explanation"):
                st.markdown(f"**{field.replace('_',' ').title()}:** {meta['explanation']}")

        if next_q:
            st.session_state.messages.append({"role": "assistant", "content": next_q})
        else:
            st.session_state.step = "power"
        st.session_state.awaiting_info_update = False
        st.rerun()

# ─── 4. STEP 2: Power Dynamic Calibration ─────────────────────
if st.session_state.step in ["power", "category", "load", "chat"]:
    st.markdown("---")
    st.subheader("Step 2: Describe the Other Person’s Behavior")

    with st.form("leadership_form", clear_on_submit=False):
        # 1. MCQ questions
        st.markdown("**Quick Quiz:** Select the option that best matches their typical behavior…")
        mcq_answers = []
        for i, qdict in enumerate(MCQ_QUESTIONS):
            ans = st.radio(
                qdict["q"],
                options=list(qdict["options"].keys()),
                format_func=lambda x, qdict=qdict: qdict["options"][x],
                key=f"mcq_{i}",
                index=["A", "B", "C"].index(st.session_state.mcq_answers[i]) if st.session_state.mcq_answers[i] else 0
            )
            mcq_answers.append(ans)
        # 2. Paragraph
        leader_input = st.text_area(
            "In a few sentences, describe this person’s leadership style or behavior (optional):",
            value=st.session_state.leader_input
        )
        submit_leadership = st.form_submit_button("Analyze Leadership Style")
    
    # When form is submitted
    if submit_leadership:
        st.session_state.mcq_answers = mcq_answers
        st.session_state.leader_input = leader_input
        # CALL NEW BACKEND FUNCTION
        leadership = get_power_dynamic_model(
            user_description=leader_input,
            mcq_answers=mcq_answers
        )
        st.session_state.info = update_info_with_power_dynamic(st.session_state.info, leadership)
        # Build summary
        if leadership and leadership.get("obtained", True):
            lines = [
                f"**Based on your answers, the best match is:** {leadership['name']}",
                "",
                f"--- **{leadership['name']}** Leadership Style ---",
                f"**Traits:** {leadership['traits']}",
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
            st.session_state.leadership_description = "*User skipped leadership analysis or not enough info provided.*"
        st.session_state.leadership_result = True
        st.rerun()

    if st.session_state.leadership_result:
        st.markdown(f"**You wrote:** {st.session_state.leader_input}")
        st.markdown(st.session_state.leadership_description)
        st.markdown(f"**Quiz answers:** {', '.join(st.session_state.mcq_answers)}")
        if st.button("That doesn’t look right—let me try again"):
            st.session_state.leadership_result = False
            st.session_state.leader_input = ""
            st.session_state.leadership_description = ""
            st.session_state.info["leadership_style"] = {
                "obtained": False,
                "explanation": None}
            st.rerun()
        if st.session_state.step == "power":
            if st.button("Continue to Step 3: Categorize"):
                st.session_state.step = "category"
                st.rerun()

# ─── 5. STEP 3: Categorize & Choose Scenario Type ─────────────
if st.session_state.step in ["load", "chat", "category"]:
    st.markdown("---")
    st.subheader("Step 3: What’s the Nature of Your Situation?")
    if st.session_state.category is None:
        with st.spinner("Categorizing situation…"):
            st.session_state.category = categorize_situation(
                st.session_state.client,
                st.session_state.info
            )

    detected = st.session_state.category["category"]
    st.success(f"Detected category: **{detected}**")
    st.write(st.session_state.category.get("explanation", ""))
    choices = ["Promotion", "Sign-on (new job or role)", "Work-related problems"]
    if "override_category" not in st.session_state:
        st.session_state.override_category = detected

    _ = st.selectbox(
        "Choose a different category if you’d like:",
        choices,
        index=choices.index(detected),
        key="override_category"
    )
    st.subheader("What kind of help would you like?")
    scenario_choices = ["advice", "rehearsal", "emotional_support"]
    scenario_type_val = st.session_state.get("scenario_type")
    default_index = scenario_choices.index(scenario_type_val) if scenario_type_val in scenario_choices else 0

    scenario_type = st.radio(
        "Choose a mode:",
        scenario_choices,
        index=default_index,
        key="scenario_type"
    )
    if scenario_type == "rehearsal":
        rehearsal_level = st.slider(
            "Select rehearsal realism level (1 = most unreasonable, 5 = most reasonable):",
            min_value=1,
            max_value=5,
            value=3,
            key="rehearsal_level"
        )
    else:
        rehearsal_level = None
    if st.button("Save Selections"):
        st.session_state.category = {
            "category": st.session_state.override_category,
            "explanation": ("Overridden by user" if st.session_state.override_category != detected else
                            st.session_state.category.get("explanation", ""))
        }
        st.success("Selections saved. When ready, click 'Next Step' below.")
    if st.button("Next Step"):
        st.session_state.step = "load"
        st.rerun()

# ─── 6. STEP 4: Load Scenario Data ─────────────────────────────
if st.session_state.step in ["load", "chat"]:
    st.markdown("---")
    st.subheader("Step 4: Loading Example Conversations")
    if not st.session_state.get("scenario_loaded", False):
        with st.spinner("Loading example conversations…"):
            ideal_dialogue = load_scenario_data(
                st.session_state.scenario_type,
                st.session_state.category["category"],
                st.session_state.get("rehearsal_level")
            )
            advisor_traits, advisor_responses = extract_advisor_traits(ideal_dialogue)
            st.session_state.traits = advisor_traits
            st.session_state.responses = advisor_responses

            st.session_state.scenario_loaded = True
            st.session_state.step = "chat"
            st.rerun()
    else:
        st.markdown("Scenario data loaded successfully.")
        st.markdown(f" - {len(st.session_state.traits)} Brett elements extracted.")
        st.markdown(f" - {len(st.session_state.responses)} example responses ready.")

# ─── 7. STEP 5: Load Chat ─────────────────────────────────────
if st.session_state.step == "chat":
    FEEDBACK_INTERVAL = 2

    # Initialize chat with first assistant message if not started
    if not st.session_state.chat_started:
        # Clean up messages to only include user/assistant roles
        st.session_state.messages = [
            m for m in st.session_state.messages
            if m.get("role") in {"user", "assistant"}
        ]
        
        # Generate first assistant message
        _ = get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            st.session_state.traits,
            st.session_state.scenario_type,
            st.session_state.category["category"],
            st.session_state.turn_count,
            st.session_state.get("rehearsal_level")
        )
        st.session_state.chat_started = True
        st.session_state.turn_count += 1

    # --- Render all messages up to now ---
    for msg in st.session_state.messages[st.session_state.get("chat_start_idx", 0):]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Check if we need to show feedback form ---
    if st.session_state.get("waiting_for_feedback", False):
        with st.expander("Optional: Give feedback on the assistant's tone or realism"):
            feedback_input = st.text_area("What could the assistant improve? (e.g., tone, realism, phrasing)")
            if st.button("Submit Feedback"):
                if feedback_input.strip():
                    st.session_state.user_feedback.append(feedback_input.strip())
                    st.session_state.feedback_log.append({
                        "content": feedback_input.strip(),
                        "turn_count": st.session_state.turn_count,
                        "timestamp": datetime.now().isoformat(),
                    })
                    st.success("✅ Feedback received — will be used in the next assistant response.")

                st.session_state.info["user_feedback"] = st.session_state.user_feedback
                if st.session_state.user_feedback:
                    summary = summarize_user_feedback(
                        st.session_state.user_feedback,
                        st.session_state.client,
                        st.session_state.info,
                        st.session_state.traits,
                        st.session_state.turn_count
                    )
                    st.session_state.info["latest_feedback_summary"] = summary

                st.session_state.waiting_for_feedback = False

                # Process the last user message and get assistant response
                last_user_message = next(
                    (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
                style_scores = st.session_state.analyzer.analyze_text(last_user_message)
                st.session_state.info["language_style"].update(style_scores)

                st.session_state.messages = [
                    m for m in st.session_state.messages
                    if m.get("role") in {"user", "assistant"}
                ]
                _ = get_next_assistant_turn(
                    st.session_state.client,
                    st.session_state.messages,
                    st.session_state.info,
                    st.session_state.traits,
                    st.session_state.scenario_type,
                    st.session_state.category["category"],
                    st.session_state.turn_count,
                    st.session_state.get("rehearsal_level")
                )
                st.session_state.turn_count += 1
                st.rerun()
    else:
        # --- Single chat input with unique key ---
        user_text = st.chat_input("Your message…", key="main_chat_input")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            # Check if we need feedback after this turn
            if st.session_state.turn_count > 0 and st.session_state.turn_count % FEEDBACK_INTERVAL == 0:
                st.session_state.waiting_for_feedback = True
            else:
                st.session_state.awaiting_response = True
            st.rerun()

    # --- If awaiting assistant response: call model, append reply, clear flag, rerun ---
    if st.session_state.get("awaiting_response", False):
        # Update language style
        last_user_msg = st.session_state.messages[-1]["content"]
        style_scores = st.session_state.analyzer.analyze_text(last_user_msg)
        st.session_state.info["language_style"].update(style_scores)

        # Update user feedback info
        st.session_state.info["user_feedback"] = st.session_state.user_feedback
        if st.session_state.user_feedback:
            summary = summarize_user_feedback(
                st.session_state.user_feedback,
                st.session_state.client,
                st.session_state.info,
                st.session_state.traits,
                st.session_state.turn_count
            )
            st.session_state.info["latest_feedback_summary"] = summary

        # Clean messages and call model
        st.session_state.messages = [
            m for m in st.session_state.messages
            if m.get("role") in {"user", "assistant"}
        ]
        _ = get_next_assistant_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.info,
            st.session_state.traits,
            st.session_state.scenario_type,
            st.session_state.category["category"],
            st.session_state.turn_count,
            st.session_state.get("rehearsal_level")
        )
        st.session_state.turn_count += 1
        st.session_state.awaiting_response = False
        st.rerun()

    st.markdown("---")
    if st.button("End Chat and Save"):
        filename = save_chat_locally(
            info=st.session_state.get("info", {}),
            category=st.session_state.get("category", {}),
            language_analysis=st.session_state.get("language_analysis", {}),
            messages=st.session_state.get("messages", []),
            advisor_traits=st.session_state.get("traits", []),
            scenario_type=st.session_state.get("scenario_type", "unknown"),
            rehearsal_level=st.session_state.get("rehearsal_level"),
            feedback_log=st.session_state.get("feedback_log", [])
        )
        st.success(f"✅ Chat saved to `{filename}`")
        st.success("Chat saved. You can now close this tab or stop the server.")
    st.stop()

