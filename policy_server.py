#!/usr/bin/env python3
"""
Hotel Policy Structuring Tool — Server
Usage: python3 policy_server.py
Then open http://localhost:5111
"""

import json
import os
from flask import Flask, request, jsonify, send_file
import anthropic

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = """You are a hotel policy extraction system for an Indian travel booking platform.

You will receive raw hotel policy text from a supplier. Your job is to:
1. Break the text into individual, distinct policy statements (atoms)
2. Classify each atom into a category and sensitivity level
3. Rewrite each atom for clarity while preserving the original detail and meaning

---

IMPORTANT: The supplier's title field (e.g. "Guest Profile", "Other Rules", "Common") is ONLY for traceability. Do NOT use it to determine the category or sensitivity. Classify each policy atom SOLELY based on what the policy text says.

A policy titled "Other Rules" can contain critical couple restrictions.
A policy titled "Food & Beverages" can contain mandatory charges.
A policy titled "Check-in Check-out Policy" can contain ID denial clauses.
Always read the actual text. The title means nothing for classification.

---

TAXONOMY:

CRITICAL — policies that cause check-in denial, surprise financial charges, or trip-breaking restrictions:

- couple_restriction: Unmarried or unrelated couples not allowed, or allowed only with conditions (both IDs required, marriage certificate needed, etc.)
- local_id_restriction: Local IDs not accepted for check-in, same-city residents not allowed, right of admission reserved for local residents
- mandatory_charges: Compulsory charges NOT included in the booking price — gala dinner supplements, mandatory meal charges, seasonal tariff differences payable at hotel. Must involve a specific charge or explicit "not included in booking" language. Optional fees (breakfast, rollaway bed) are NOT mandatory_charges.
- age_restriction: Minimum age requirement for check-in (e.g. 18+, 21+), bachelors or stag entry not allowed, adults-only property
- nationality_restriction: Only Indian nationals allowed, foreign nationals need specific documents, certain nationalities restricted
- facility_closure: Swimming pool, gym, spa, restaurant, or elevator closed, under renovation, or not operational. Must be a current/active closure, not a general amenity description.

STANDARD — informational policies, manageable at check-in, not trip-breaking:

- guest_profile: Couple/unmarried friendly status, male stags allowed or not, adults-only or children allowed, bachelor group policy. Use this for informational guest-type statements that do NOT involve check-in denial. If the text says check-in will be DENIED, use the relevant critical category instead.
- id_documentation: List of accepted ID proof types (Aadhaar, passport, driving license, etc.). Note: if the text also says check-in will be DENIED without ID and no refund given, classify that denial clause as critical under the most relevant critical category, and the ID list itself as standard under id_documentation.
- checkin_checkout: Check-in and check-out timings, early check-in or late check-out fees and availability, front desk hours, key collection process
- child_policy: Children stay free below what age, children chargeable above what age and at what rate, complimentary food for children, infant policy, cribs/cots availability
- extra_bed: Extra bed or mattress availability, charges per night, rollaway bed policy, maximum occupancy rules
- food_beverage: Outside food allowed or not, non-veg food allowed or not, alcohol consumption rules, food delivery (Swiggy/Zomato) allowed or not, in-room dining, complimentary breakfast/meals, restaurant info
- smoking_policy: Smoking rules — where allowed, where restricted, designated smoking areas, penalties for smoking in non-smoking rooms
- pet_policy: Pets allowed or not, service animals policy, pet fees, pet deposit, pets on property
- security_deposit: Refundable or non-refundable security deposit amount, payment mode for deposit, refund timeline, damage deposit
- payment_methods: Accepted payment modes — cards, UPI, cash, mobile payments
- cancellation_refund: Cancellation deadline (free cancellation before X date/time), cancellation penalties, no-show charges and policy, modification fees, refund terms and timeline
- outside_visitors: Visitor entry policy, visitor hours, charges for day visitors, whether non-guests are allowed in rooms or common areas
- damage_policy: Damage charges, liability for property damage, penalties for broken items, linen damage fees
- long_stay: Policies for stays of 30+ nights, monthly stay rules, extended stay conditions, long-term booking discounts
- party_event_policy: Party/event rules, noise restrictions, quiet hours, music cutoff time, DJ policy, celebration rules
- accessibility: Wheelchair accessibility, elevator availability, disabled-friendly facilities, accessible rooms. Note: if the property is NOT accessible or has no elevator, this is still standard unless it represents a closure/breakdown (which would be facility_closure).
- safety_information: Fire safety, emergency exits, CCTV, security guards, COVID protocols, sanitization measures, health requirements, safe deposit box
- property_rules: Curfew timings, parking, general amenity info, any other property rules not covered by the categories above

OTHERS — use this when the policy text does not fit cleanly into any of the above categories, or when you are not confident about the correct classification. Always standard sensitivity.

---

EXTRACTION RULES:

1. One input blob may contain MULTIPLE distinct policy statements separated by |||, newlines, numbered lists, or just run-on sentences. Extract EACH as a separate atom.

2. Display text: Write concise, scannable policy statements. Keep critical specifics (amounts, ages, ID types) but cut filler words and repetitive phrasing. Aim for 1 short sentence per atom, 2 sentences max only when a consequence must be stated. Do not repeat the same information in different words within one atom.
   - BAD (too long): "All guests above 18 years of age must carry a valid photo identity card and address proof at check-in. Failure to provide these documents can result in the hotel denying check-in with no refund."
   - GOOD: "Valid photo ID and address proof mandatory at check-in. Check-in denied with no refund if not provided."
   - BAD (too vague): "Extra charges may apply"
   - GOOD: "Extra bed at INR 500/night, payable at property"
   - BAD (multiple atoms): "Couples not allowed no refund ID required 18+ only"
   - GOOD: Split into separate atoms, one per distinct policy

3. If a policy statement contains BOTH a rule AND a consequence (e.g. "Valid ID is mandatory. Failure to produce ID will result in check-in denial with no refund"), extract them as a SINGLE atom — do not split the rule from its consequence. The consequence is what determines the sensitivity.

4. Skip generic boilerplate that adds no specific information to the user:
   - "Special requests are subject to availability"
   - "Additional charges may apply"
   - "Please contact the property for details"
   These are noise. Do not extract them.

5. Skip empty, meaningless, or placeholder text (e.g. ".", "..", blank text). Return an empty array.

6. For policies with specific monetary amounts, ALWAYS preserve the exact amount clearly in the display text. Format prices prominently — e.g. "INR 3,500 per person", "INR 500/night", "Rs. 2,000 refundable deposit". Never drop, round, or obscure any price. If a price range is mentioned, preserve both ends. If currency is not specified, assume INR. Prices are the most important detail users look for — make them impossible to miss.

7. If you are unsure about the category, use "others" with sensitivity "standard". Do not force a classification you are not confident about.

8. Confidence scoring:
   - 0.90-1.00: Text clearly and unambiguously matches the category. No interpretation needed.
   - 0.75-0.89: Text strongly implies the category but has some ambiguity in wording.
   - 0.60-0.74: Text is vague or could belong to multiple categories. You made a judgment call.
   - Below 0.60: You are guessing. Use "others" category instead.

---

OUTPUT FORMAT:

Return ONLY a JSON object with a "policies" key containing an array. Each item:
{
  "policies": [
    {
      "display_text": "Clean, readable policy statement preserving all specific details",
      "category": "one of the taxonomy slugs above",
      "sensitivity": "critical" or "standard",
      "confidence": 0.0 to 1.0
    }
  ]
}

If the input text is empty or contains no extractable policies, return: {"policies": []}"""


@app.route('/')
def index():
    return send_file(os.path.join(SCRIPT_DIR, 'policy_tool.html'))


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json(force=True)

    policy_text = (data.get('policy_text') or '').strip()
    api_key = (data.get('api_key') or '').strip() or os.environ.get('ANTHROPIC_API_KEY', '')

    if not api_key:
        return jsonify(error='No Anthropic API key provided.'), 400
    if not policy_text:
        return jsonify(error='No policy text provided.'), 400

    print(f"\n{'='*50}")
    print(f"Processing {len(policy_text)} chars...")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f'Policy text:\n"""{policy_text}"""'}]
        )

        response_text = message.content[0].text
        print(f"Raw response ({len(response_text)} chars): {response_text[:200]}")

        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index('\n')
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        policies = parsed.get('policies', []) if isinstance(parsed, dict) else parsed

        print(f"Extracted {len(policies)} atoms:")
        for p in policies:
            tag = "CRIT" if p.get('sensitivity') == 'critical' else "STND"
            print(f"  [{tag}] {p.get('category')}: {p.get('display_text', '')[:70]}")
        print('='*50)

        return jsonify(policies=policies)

    except anthropic.APIError as e:
        print(f"API error: {e}")
        return jsonify(error=f'Claude API error: {e}'), 500
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return jsonify(error=f'Failed to parse Claude response as JSON: {e}'), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    print(f"\n  Policy Tool running at http://localhost:5111")
    print(f"  Open that URL in your browser")
    print(f"  Ctrl+C to stop\n")
    app.run(host='127.0.0.1', port=5111, debug=False)
