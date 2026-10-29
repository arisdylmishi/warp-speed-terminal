import os
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
import stripe
from supabase import create_client

app = Flask(__name__)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

# How many days each Stripe Price ID should grant.
# We'll fill these in together once we're in Stripe looking at the real IDs.
PRICE_TO_DAYS = {
    os.environ.get("PRICE_ID_1M", ""): 30,
    os.environ.get("PRICE_ID_3M", ""): 90,
    os.environ.get("PRICE_ID_6M", ""): 180,
    os.environ.get("PRICE_ID_1Y", ""): 365,
}


@app.route("/", methods=["GET"])
def health():
    # Just so visiting the URL in a browser shows something other than an error.
    return "Warp Speed Terminal webhook is running.", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        # Message didn't really come from Stripe, or was tampered with. Reject it.
        return jsonify(error="invalid signature"), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

                
        session_dict = session.to_dict() if hasattr(session, "to_dict") else session
        email = session_dict.get("client_reference_id") or (
            session_dict.get("customer_details", {}) or {}
        ).get("email")

        days = 30
        try:
            line_items = stripe.checkout.Session.list_line_items(session["id"], limit=1)
            if line_items["data"]:
                price_id = line_items["data"][0]["price"]["id"]
                days = PRICE_TO_DAYS.get(price_id, 30)
        except Exception:
            pass
        if email:
            new_expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            supabase.table("users").update(
                {"status": "active", "expiry_date": new_expiry}
            ).eq("email", email).execute()
            result = supabase.table("users").update(
                {"status": "active", "expiry_date": new_expiry}
            ).eq("email", email).execute()
        
    return jsonify(received=True), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
