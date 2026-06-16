import os
import stripe
from fastapi import FastAPI, Request, HTTPException

# Initialize Stripe with your private API key (stored securely in environment variables)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://streamlit.app")

app = FastAPI(title="SaaS Monetization Engine")

@app.post("/api/v1/billing/create-checkout")
async def create_checkout_session(user_email: str, plan_id: str):
    """
    Generates a secure checkout link. 
    Plan ID matches your Stripe Dashboard Price IDs (e.g., price_H5gg2...).
    """
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=user_email,
            line_items=[{
                'price': plan_id,  # Recurring subscription plan ID
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{FRONTEND_URL}?session_id={{CHECKOUT_SESSION_ID}}&payment=success",
            cancel_url=f"{FRONTEND_URL}?payment=cancelled",
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/billing/webhook")
async def stripe_webhook(request: Request):
    """
    Listens to Stripe server updates to automatically unlock premium 
    access when a customer's payment succeeds.
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook Error: {str(e)}")

    # Handle successful payments
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        customer_email = session.get('customer_details', {}).get('email')
        print(f"💰 Payment Successful! Unlocking Premium features for: {customer_email}")
        # Here you would update your database (e.g., Supabase) setting user_tier = 'premium'

    return {"status": "success"}
