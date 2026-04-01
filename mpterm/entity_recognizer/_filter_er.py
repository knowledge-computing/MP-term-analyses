from typing import List, Dict

def select_entities(ner_results:dict, min_score:float=0.1) -> Dict[str, List[str]]:
    """
    Converts tokens back into word-level and filters entities to report based on score

    Parameters
    : ner_results (dict): score for all tokens in sentence
    : min_score (min_score): (optional) minimum confidence score; default=0.1

    Return
    : cleaned_ner (Dict[str, List[str]]): selected entity per sentence
    """
    cleaned_ner = {}

    for sentence, tmp_ners in ner_results.items():
        grouped_entities = []
        current_entity = None

        for result in tmp_ners:
            if result["score"] < min_score:
                if current_entity:
                    if current_entity["score"] >= min_score:
                        grouped_entities.append(current_entity['word'].strip())
                    current_entity = None
                continue

            # Remove sub token marker
            word = result["word"].replace("##", "")
            
            if current_entity and result["entity_group"] == current_entity["entity_group"] and result["start"] == current_entity["end"]:
                current_entity["word"] += word
                current_entity["end"] = result["end"]
                current_entity["score"] = min(current_entity["score"], result["score"])
                
                if current_entity["score"] < min_score:
                    current_entity = None
            else:
                if current_entity and current_entity["score"] >= min_score:
                    grouped_entities.append(current_entity['word'].strip())
                
                current_entity = {
                    "entity_group": result["entity_group"],
                    "word": word,
                    "start": result["start"],
                    "end": result["end"],
                    "score": result["score"]
                }

        # Add the last entity if it meets threshold
        if current_entity and current_entity["score"] >= min_score:
            grouped_entities.append(current_entity['word'].strip())

        cleaned_ner[sentence] = grouped_entities

    return cleaned_ner