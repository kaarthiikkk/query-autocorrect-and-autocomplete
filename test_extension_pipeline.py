# test_extension_pipeline.py - Automated verification of the spelling assistant pipeline
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_api_health():
    print("Testing /api/v1/health...")
    r = requests.get(f"{BASE_URL}/api/v1/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy"
    print(f"✓ Health check OK: {len(data['supported_languages'])} languages supported.")

def test_word(word, expected_typo, expected_best_correction=None):
    payload = {"query": word, "top_k": 5, "user_id": "inline_tester"}
    r = requests.post(f"{BASE_URL}/api/v1/suggest", json=payload)
    assert r.status_code == 200, f"Suggest failed for {word}: {r.status_code}"
    data = r.json()
    is_typo = data.get("error_classification") == "typo"
    spelling_corrections = data.get("spelling_corrections", [])
    ranked_suggestions = data.get("ranked_suggestions", [])
    candidates = [c for c in dict.fromkeys(spelling_corrections + ranked_suggestions) if c.lower() != word.lower()]

    print(f"\nQuery: '{word}' -> Language: {data.get('detected_language')}, Classification: {data.get('error_classification')}")
    print(f"  Spelling corrections: {spelling_corrections}")
    print(f"  Ranked suggestions: {ranked_suggestions}")
    print(f"  Deduped non-identical candidates: {candidates[:4]}")

    if expected_typo:
        assert is_typo or len(candidates) > 0, f"Expected '{word}' to be identified as typo, got classification={data.get('error_classification')}"
        if expected_best_correction:
            assert expected_best_correction.lower() in [c.lower() for c in candidates], \
                f"Expected correction '{expected_best_correction}' not found in candidates: {candidates}"
        print(f"✓ '{word}' successfully identified as typo. Top candidates: {candidates[:3]}")
    else:
        assert not is_typo or len(candidates) == 0, f"Expected '{word}' to be valid word, but got typo={is_typo} with candidates {candidates}"
        print(f"✓ '{word}' successfully recognized as valid (NOT underlined).")

def test_feedback():
    print("\nTesting /api/v1/feedback...")
    payload = {"query": "recieved", "chosen_word": "received", "user_id": "inline_tester"}
    r = requests.post(f"{BASE_URL}/api/v1/feedback", json=payload)
    assert r.status_code == 200, f"Feedback failed: {r.status_code}"
    data = r.json()
    assert data.get("status") == "success"
    print(f"✓ Feedback loop recorded successfully for 'received'. Summary: {data.get('summary')}")

if __name__ == "__main__":
    try:
        test_api_health()
        # Test Case 1 & 2: User prompt example "I recieved your mesage yesterday."
        test_word("recieved", expected_typo=True, expected_best_correction="received")
        test_word("mesage", expected_typo=True, expected_best_correction="message")
        test_word("yesterday", expected_typo=False)

        # Test Case 3: Common English programming typos
        test_word("pythno", expected_typo=True, expected_best_correction="python")
        test_word("writting", expected_typo=True, expected_best_correction="writing")

        # Test Case 4: Multilingual valid words (must not be marked as typos)
        test_word("sapda", expected_typo=False) # Tamil Romanized
        test_word("deutschland", expected_typo=False) # German

        # Test Case 5: Multilingual typos
        test_word("informasion", expected_typo=True, expected_best_correction="informacion")

        # Test Case 6: Feedback reinforcement
        test_feedback()

        print("\n==========================================")
        print("ALL END-TO-END PIPELINE TESTS PASSED 100%!")
        print("==========================================")
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
