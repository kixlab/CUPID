def format_interaction_log(prior_interaction_log):
    interaction_log_str = ""
    for i, interaction in enumerate(prior_interaction_log):
        interaction_log_str += f"### Session {i+1}\n\n"
        for message in interaction['dialogue']:
            role_str = "User" if message['role'] == "user" else "AI Assistant"
            content = message['content']
            interaction_log_str += f"#### {role_str}\n\n{content}\n\n"       
        interaction_log_str += "---\n\n"
    return interaction_log_str.strip()