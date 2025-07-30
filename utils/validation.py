import json

def validate_context_factors(context_factors):
    contrastive_pairs = []
    for i, factor in enumerate(context_factors):
        if factor['related_factor'] == "N/A":
            continue

        for other_factor in context_factors[i+1:]:
            # A pair of entities should have each other as related entities
            if factor['related_factor'] == other_factor['factor'] and factor['factor'] == other_factor['related_factor']:
                contrastive_pairs.append(f"{factor['factor']} - {other_factor['factor']}")
    
    # If with_contrastive is False, then there should be no contrastive entities
    if len(contrastive_pairs) == 1:
        return True, "Valid context factors"
    else:
        return False, "Invalid number of contrastive pairs found in context factors"

def validate_sessions(sessions, context_factors, n_sessions):
    if len(sessions) != n_sessions:
        return False, "Invalid number of sessions"

    final_factor = sessions[-1]['context_factor']
    final_preference = sessions[-1]['preference']

    for session in sessions:
        # Check that each scenario has the field 'request_with_factor'
        if 'request_with_factor' not in session:
            return False, "Missing 'request_with_factor' field"

    all_factors = [factor['factor'].replace(".", "").strip() for factor in context_factors]
    all_preferences = [factor['preference'].replace(".", "").strip() for factor in context_factors]

    # Find the factor that contrasts with the final factor
    contrastive_factor = None
    for factor in context_factors:
        if 'related_factor' not in factor or factor['related_factor'] == "N/A":
            continue
        if factor['factor'] == final_factor:
            contrastive_factor = factor['related_factor']
        elif factor['related_factor'] == final_factor:
            contrastive_factor = factor['factor']
    
    # Find the preference of the contrastive factor
    contrastive_preference = None
    for factor in context_factors:
        if factor['factor'] == contrastive_factor:
            contrastive_preference = factor['preference']

    # Check validity of changing journeys
    preference_history = []
    contrastive_count = 0
    for session in sessions[:-1]:
        if session['context_factor'].replace(".", "").strip() not in all_factors:
            return False, "Session's factor not found in list of context factors"
        
        # Check if contrastive entity is found
        if session['context_factor'] == contrastive_factor and session['preference'] == contrastive_preference:
            contrastive_count += 1
        elif session['context_factor'] == contrastive_factor and session['preference'] != contrastive_preference:
            return False, "Contrastive entity found in journey but with different preference"

        # Find all scenarios with the final entity
        if session['context_factor'] == final_factor:
            preference_history.append(session['preference'])
        else:            
            if session['preference'].replace(".", "").strip() not in all_preferences:
                return False, "Session's preference not found in list of context factors"

    # In changing journey, the final entity should only have been presented four times (2 consistent, 2 changing)
    if len(preference_history) >  4:
        return False, "Final factor found more than four times in interaction sessions"
    elif len(preference_history) < 4:
        return False, "Final factor found less than four times in interaction sessions"
    elif preference_history[0].replace(".", "").strip() not in all_preferences:
        return False, "Initial preference not found in list of context factors"
    # The first preference should be different from the last preference
    # But the second preference should be the same as the last preference
    elif contrastive_count > 2:
        return False, "Contrastive factor found more than twice in interaction sessions"
    elif contrastive_count == 0:
        return False, "Contrastive factor not found in interaction sessions"
    # Verify that the preferences appears consistently twice, then changes, then appears consistently twice again, and the final session also has the same final preference
    elif (preference_history[0] == preference_history[1]) and (preference_history[1] != preference_history[2]) and (preference_history[2] == preference_history[3]) and (preference_history[3] == final_preference):
        return True, "Valid interaction sessions"
    else:
        return False, "Preference did not change as expected in the interaction sessions"
